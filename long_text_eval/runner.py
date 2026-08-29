from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

from extreme_eval.audit import sha256_file
from extreme_eval.system_info import collect_system_info, process_memory_bytes
from muddywater.config import load_config
from muddywater.generation_runtime import load_generation_runtime
from muddywater.text_simplification.evaluation import format_prompt
from muddywater.text_simplification.inference import LongTextOptions, TextSimplifier
from muddywater.utils import enable_torch_backends, set_seed

from .audits import run_planning_benchmarks, run_segmentation_audit
from .cases import build_long_documents
from .metrics import compare_modes, score_document, summarize_by, summarize_long_records
from .types import LongDocument


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "inference_text_simplification_portable.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "long_text_chunking_evaluation_20260814"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extreme A/B evaluation for sentence-aware long-text simplification."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-budget", type=int, default=160)
    parser.add_argument("--document-limit", type=int, default=0)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-budget-sweep", action="store_true")
    parser.add_argument("--skip-determinism", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _run_tests() -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "source"), str(PROJECT_ROOT)))
    command = [sys.executable, "-m", "pytest", "tests", "-q", "--disable-warnings"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    match = re.search(r"(\d+) failed, (\d+) passed", output)
    return {
        "command": command,
        "return_code": completed.returncode,
        "passed": int(match.group(2)) if match else None,
        "failed": int(match.group(1)) if match else (0 if completed.returncode == 0 else None),
        "expected_packaging_failure": "pretrain_fineweb_v21_20b_540m_wide.yaml" in output,
        "new_long_text_tests_passed": "test_long_text_simplification" not in output,
        "output_tail": "\n".join(output.splitlines()[-80:]),
    }


def _load_runtime(config_path: Path, threads: int):
    torch.set_num_threads(threads)
    set_seed(20260814)
    enable_torch_backends()
    config = load_config(config_path)
    config["__config_path__"] = str(config_path.resolve())
    config["device"] = "cpu" if not torch.cuda.is_available() else "auto"
    memory_before = process_memory_bytes()
    started = time.perf_counter()
    runtime = load_generation_runtime(config)
    load_seconds = time.perf_counter() - started
    return config, runtime, load_seconds, memory_before, process_memory_bytes()


def _prompt_builder(config: dict[str, Any]):
    task = dict(config.get("text_simplification", {}))

    def build(source: str) -> str:
        return format_prompt(
            source,
            source_label=str(task.get("source_label", "<|im_start|>")),
            target_label=str(task.get("target_label", "<|im_end|>")),
            sanitize=bool(task.get("sanitize_reserved_tags", True)),
        )

    return build


def _make_simplifier(runtime, prompt_builder, budget: int) -> TextSimplifier:
    return TextSimplifier(
        runtime,
        prompt_builder=prompt_builder,
        options=LongTextOptions(
            max_prompt_tokens=budget,
            adaptive_output_tokens=True,
            output_token_ratio=1.25,
            min_new_tokens=32,
            fallback_on_empty=True,
        ),
        generation_overrides={
            "return_full_text": False,
            "skip_special_tokens": False,
        },
    )


def _warm_up(simplifier: TextSimplifier) -> dict[str, Any]:
    result = simplifier.simplify("这是一条用于预热模型的短句。", mode="never")
    return {
        "latency_ms": round(result.elapsed_ms, 6),
        "output": result.text,
        "finish_reason": result.chunks[0].finish_reason,
    }


def _evaluate_ab(
    documents: Sequence[LongDocument],
    direct: TextSimplifier,
    chunked: TextSimplifier,
    *,
    budget: int,
    progress_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct_records: list[dict[str, Any]] = []
    chunked_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, document in enumerate(documents, start=1):
        modes = (("direct", direct), ("chunked", chunked))
        if index % 2 == 0:
            modes = tuple(reversed(modes))
        scored: dict[str, dict[str, Any]] = {}
        for mode, simplifier in modes:
            result = simplifier.simplify(
                document.source,
                mode="never" if mode == "direct" else "always",
            )
            scored[mode] = score_document(
                document,
                result,
                mode=mode,
                chunk_budget=None if mode == "direct" else budget,
            )
        direct_records.append(scored["direct"])
        chunked_records.append(scored["chunked"])
        elapsed = time.perf_counter() - started
        print(
            f"A/B {index}/{len(documents)} {document.document_id} "
            f"tokens={scored['direct']['source_prompt_tokens']} "
            f"direct={scored['direct']['latency_ms']/1000:.2f}s "
            f"chunked={scored['chunked']['latency_ms']/1000:.2f}s "
            f"chunks={scored['chunked']['chunk_count']} elapsed={elapsed:.1f}s",
            flush=True,
        )
        _atomic_json(
            progress_path,
            {
                "phase": "ab",
                "completed": index,
                "total": len(documents),
                "direct_records": direct_records,
                "chunked_records": chunked_records,
            },
        )
    return direct_records, chunked_records


def _select_sweep_documents(
    documents: Sequence[LongDocument],
    direct_records: Sequence[dict[str, Any]],
) -> list[LongDocument]:
    token_by_id = {
        str(record["document_id"]): int(record["source_prompt_tokens"])
        for record in direct_records
    }
    selected: list[LongDocument] = []
    used: set[str] = set()
    for target in (250, 500, 800, 1150):
        candidates = [doc for doc in documents if doc.document_id not in used]
        best = min(candidates, key=lambda doc: abs(token_by_id[doc.document_id] - target))
        selected.append(best)
        used.add(best.document_id)
    return selected


def _budget_sweep(
    documents: Sequence[LongDocument],
    primary_records: Sequence[dict[str, Any]],
    runtime,
    prompt_builder,
    budgets: Sequence[int],
    primary_budget: int,
) -> dict[str, Any]:
    primary_by_id = {str(row["document_id"]): row for row in primary_records}
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        simplifier = _make_simplifier(runtime, prompt_builder, budget)
        for index, document in enumerate(documents, start=1):
            if budget == primary_budget:
                record = primary_by_id[document.document_id]
            else:
                result = simplifier.simplify(document.source, mode="always")
                record = score_document(
                    document,
                    result,
                    mode="chunked",
                    chunk_budget=budget,
                )
            rows.append(record)
            print(
                f"budget {budget} {index}/{len(documents)} {document.document_id} "
                f"latency={record['latency_ms']/1000:.2f}s chunks={record['chunk_count']}",
                flush=True,
            )
    return {
        "document_ids": [document.document_id for document in documents],
        "budgets": list(budgets),
        "records": rows,
        "by_budget": {
            str(budget): summarize_long_records(
                [row for row in rows if int(row["chunk_budget"]) == budget]
            )
            for budget in budgets
        },
    }


def _determinism_check(
    documents: Sequence[LongDocument],
    primary_records: Sequence[dict[str, Any]],
    simplifier: TextSimplifier,
) -> dict[str, Any]:
    primary_by_id = {str(row["document_id"]): str(row["prediction"]) for row in primary_records}
    details = []
    for document in documents:
        outputs = [primary_by_id[document.document_id]]
        for _ in range(2):
            outputs.append(simplifier.simplify(document.source, mode="always").text)
        details.append(
            {
                "document_id": document.document_id,
                "repeats": len(outputs),
                "exact_match": len(set(outputs)) == 1,
                "output_lengths": [len(output) for output in outputs],
            }
        )
        print(f"determinism {document.document_id} exact={details[-1]['exact_match']}", flush=True)
    return {
        "documents": len(details),
        "repeats_per_document": 3,
        "exact_determinism_rate": round(mean(row["exact_match"] for row in details), 6),
        "details": details,
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw"
    progress_path = raw_dir / "progress.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "evaluation": "Murmur 203M sentence-aware long-text chunking extreme evaluation",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "project_root": str(PROJECT_ROOT),
        "system": collect_system_info(PROJECT_ROOT),
        "test_suite": None if args.skip_tests else _run_tests(),
        "methodology": {
            "validation_split_used": False,
            "packaged_train_or_validation_rows_used": False,
            "primary_chunk_budget": args.chunk_budget,
            "ab_order": "Alternated direct-first and chunked-first per document to reduce order bias.",
            "decoding": "Greedy deterministic decoding; KV cache enabled.",
        },
    }

    config_path = Path(args.config).resolve()
    config, runtime, load_seconds, memory_before, memory_after = _load_runtime(
        config_path, args.threads
    )
    prompt_builder = _prompt_builder(config)
    direct = _make_simplifier(runtime, prompt_builder, args.chunk_budget)
    chunked = _make_simplifier(runtime, prompt_builder, args.chunk_budget)
    result["runtime"] = {
        "config": str(config_path),
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_sha256": sha256_file(runtime.checkpoint_path),
        "checkpoint_bytes": runtime.checkpoint_path.stat().st_size,
        "tokenizer": str(runtime.tokenizer_path),
        "tokenizer_sha256": sha256_file(runtime.tokenizer_path),
        "device": str(runtime.device),
        "threads": args.threads,
        "load_seconds": round(load_seconds, 6),
        "memory_before_load": memory_before,
        "memory_after_load": memory_after,
        "model_parameters": sum(parameter.numel() for parameter in runtime.model.parameters()),
        "model_config": runtime.model.config.__dict__,
        "generation_config": runtime.generation_config,
    }
    result["warmup"] = _warm_up(direct)

    print("running segmentation and token-budget audit", flush=True)
    result["segmentation_audit"] = run_segmentation_audit(
        chunked.count_prompt_tokens,
        max_prompt_tokens=args.chunk_budget,
        fuzz_count=2000,
    )
    print("running planning microbenchmarks", flush=True)
    result["planning_benchmarks"] = run_planning_benchmarks(
        chunked.count_prompt_tokens,
        max_prompt_tokens=args.chunk_budget,
    )

    documents = build_long_documents()
    if args.document_limit > 0:
        documents = documents[: args.document_limit]
    result["documents"] = {
        "count": len(documents),
        "provenance": (
            "Four fixed independent repetition regressions plus twelve deterministic composite "
            "documents assembled from fixed authored stress cases. No packaged train/validation rows."
        ),
        "metadata": [
            {
                "document_id": doc.document_id,
                "category": doc.category,
                "sentence_count": doc.sentence_count,
                "layout": doc.layout,
                "length_tier": doc.length_tier,
                "source_chars": len(doc.source),
                "target_chars": len(doc.target),
                "must_keep_items": len(doc.must_keep),
                "tail_keep_items": len(doc.tail_keep),
            }
            for doc in documents
        ],
    }
    print(f"running alternating A/B documents={len(documents)}", flush=True)
    direct_records, chunked_records = _evaluate_ab(
        documents,
        direct,
        chunked,
        budget=args.chunk_budget,
        progress_path=progress_path,
    )
    result["direct"] = {
        "records": direct_records,
        "summary": summarize_long_records(direct_records),
        "by_length_tier": summarize_by(direct_records, "length_tier"),
        "by_category": summarize_by(direct_records, "category"),
    }
    result["chunked"] = {
        "records": chunked_records,
        "summary": summarize_long_records(chunked_records),
        "by_length_tier": summarize_by(chunked_records, "length_tier"),
        "by_category": summarize_by(chunked_records, "category"),
    }
    result["comparison"] = compare_modes(direct_records, chunked_records)

    if not args.skip_budget_sweep:
        sweep_documents = _select_sweep_documents(documents, direct_records)
        result["budget_sweep"] = _budget_sweep(
            sweep_documents,
            chunked_records,
            runtime,
            prompt_builder,
            budgets=(96, args.chunk_budget, 224),
            primary_budget=args.chunk_budget,
        )
    if not args.skip_determinism:
        preferred = ("long_02", "composite_07", "composite_12")
        selected = [doc for document_id in preferred for doc in documents if doc.document_id == document_id]
        result["determinism"] = _determinism_check(selected, chunked_records, chunked)

    result["final_process_memory"] = process_memory_bytes()
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result_path = raw_dir / "long_text_chunking_eval_results.json"
    _atomic_json(result_path, result)
    if progress_path.exists():
        progress_path.unlink()
    print(f"results written: {result_path}", flush=True)
    return result_path


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
