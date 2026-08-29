from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np
import torch

from muddywater.config import load_config
from muddywater.generation_runtime import generate_from_runtime, load_generation_runtime
from muddywater.text_simplification.evaluation import (
    SimplificationPair,
    format_prompt,
    load_pairs,
    score_prediction,
    summarize_predictions,
)
from muddywater.utils import enable_torch_backends, set_seed

from .audit import dataset_audit, sha256_file
from .cases import build_stress_cases
from .metrics import (
    copy_baseline_records,
    paired_comparison,
    perturbation_consistency,
    score_constraints,
    stratified_summaries,
    summarize_constraints,
    summarize_extended,
)
from .system_info import collect_system_info, process_memory_bytes
from .types import EvalCase, GenerationMeasurement


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "inference_text_simplification_portable.yaml"
DEFAULT_VALIDATION = PROJECT_ROOT / "data" / "text_simplification_pass_filtered" / "processed" / "validation.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "text_simplification_pass_filtered" / "processed" / "train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Murmur 203M extreme evaluation suite.")
    parser.add_argument("--output-dir", default="output/extreme_evaluation_20260814")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--validation-limit", type=int, default=0, help="0 evaluates all validation rows.")
    parser.add_argument("--stress-limit", type=int, default=0, help="0 evaluates the full authored stress suite.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-microbench", action="store_true")
    parser.add_argument("--skip-determinism", action="store_true")
    parser.add_argument(
        "--reuse-progress",
        action="store_true",
        help="Reuse validation records already stored in raw/progress.json instead of generating them again.",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _run_tests() -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "source")
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
    return {
        "command": command,
        "return_code": completed.returncode,
        "expected_packaging_failure": "pretrain_fineweb_v21_20b_540m_wide.yaml" in output,
        "summary": "70 passed, 1 failed" if "1 failed, 70 passed" in output else None,
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
    memory_after = process_memory_bytes()
    return config, runtime, load_seconds, memory_before, memory_after


def _generate(runtime, source: str, *, use_cache: bool | None = None, max_new_tokens: int | None = None) -> GenerationMeasurement:
    prompt = format_prompt(source, sanitize=True)
    prompt_tokens = len(runtime.tokenizer.encode(prompt, add_bos=runtime.add_bos, add_eos=False))
    overrides = {
        **runtime.generation_config,
        "return_full_text": False,
        "skip_special_tokens": False,
        "return_details": True,
    }
    if use_cache is not None:
        overrides["use_cache"] = use_cache
    if max_new_tokens is not None:
        overrides["max_new_tokens"] = max_new_tokens
    started = time.perf_counter()
    generated = generate_from_runtime(runtime, prompt=prompt, overrides=overrides)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(generated, dict):
        return GenerationMeasurement(str(generated), "unknown", 0, latency_ms, prompt_tokens)
    return GenerationMeasurement(
        text=str(generated.get("text", "")).strip(),
        finish_reason=str(generated.get("finish_reason", "unknown")),
        generated_tokens=int(generated.get("generated_tokens", 0) or 0),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
    )


def _score_pair(pair: SimplificationPair, measurement: GenerationMeasurement, index: int) -> dict[str, Any]:
    record = score_prediction(
        pair.source,
        pair.target,
        measurement.text,
        finish_reason=measurement.finish_reason,
        latency_ms=measurement.latency_ms,
        generated_tokens=measurement.generated_tokens,
        reserved_tags=("<|im_start|>", "<|im_end|>"),
    )
    record.update({"index": index, "prompt_tokens": measurement.prompt_tokens})
    return record


def _evaluate_validation(runtime, pairs: Sequence[SimplificationPair], progress_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, pair in enumerate(pairs, start=1):
        records.append(_score_pair(pair, _generate(runtime, pair.source), index))
        if index % 100 == 0 or index == len(pairs):
            elapsed = time.perf_counter() - started
            print(
                f"validation {index}/{len(pairs)} | elapsed={elapsed:.1f}s | rate={index / elapsed:.3f} samples/s",
                flush=True,
            )
        if index % 250 == 0:
            _atomic_json(
                progress_path,
                {"phase": "validation", "completed": index, "total": len(pairs), "records": records},
            )
    return records


def _evaluate_stress(runtime, cases: Sequence[EvalCase]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        measurement = _generate(runtime, case.source)
        record = _score_pair(
            SimplificationPair(source=case.source, target=case.target),
            measurement,
            index,
        )
        record.update(
            {
                "case_id": case.case_id,
                "case_category": case.category,
                "perturbation_group": case.perturbation_group,
                "perturbation": case.perturbation,
                "constraints": score_constraints(case, measurement.text),
            }
        )
        records.append(record)
        if index % 25 == 0 or index == len(cases):
            elapsed = time.perf_counter() - started
            print(f"stress {index}/{len(cases)} | elapsed={elapsed:.1f}s", flush=True)
    return records


def _determinism_check(runtime, sources: Sequence[str]) -> dict[str, Any]:
    details = []
    for index, source in enumerate(sources[:8], start=1):
        outputs = [_generate(runtime, source).text for _ in range(3)]
        details.append({"index": index, "source": source, "outputs": outputs, "exact_match": len(set(outputs)) == 1})
    return {
        "cases": len(details),
        "repeats_per_case": 3,
        "exact_determinism_rate": round(mean(row["exact_match"] for row in details), 6) if details else None,
        "details": details,
    }


def _timed_forward(model, input_ids: torch.Tensor, repeats: int) -> list[float]:
    timings = []
    with torch.inference_mode():
        _ = model(input_ids)
        for _ in range(repeats):
            started = time.perf_counter()
            outputs = model(input_ids)
            timings.append((time.perf_counter() - started) * 1000.0)
            del outputs
    return timings


def _microbench(runtime, validation_sources: Sequence[str], default_threads: int) -> dict[str, Any]:
    model = runtime.model
    device = runtime.device
    vocab_size = runtime.tokenizer.vocab_size
    forward_rows = []
    for length in (16, 32, 64, 128, 256, 512, 896):
        repeats = 3 if length <= 256 else 2
        ids = torch.randint(0, vocab_size, (1, length), dtype=torch.long, device=device)
        timings = _timed_forward(model, ids, repeats)
        forward_rows.append(
            {
                "batch_size": 1,
                "sequence_length": length,
                "repeats": repeats,
                "latency_ms": round(mean(timings), 6),
                "tokens_per_second": round(length / (mean(timings) / 1000.0), 6),
            }
        )
        print(f"microbench forward seq={length} done", flush=True)

    batch_rows = []
    for batch_size in (1, 2, 4):
        ids = torch.randint(0, vocab_size, (batch_size, 128), dtype=torch.long, device=device)
        timings = _timed_forward(model, ids, 3)
        batch_rows.append(
            {
                "batch_size": batch_size,
                "sequence_length": 128,
                "latency_ms": round(mean(timings), 6),
                "tokens_per_second": round((batch_size * 128) / (mean(timings) / 1000.0), 6),
            }
        )

    thread_rows = []
    for threads in (1, 2, 4, 8, 12):
        torch.set_num_threads(threads)
        ids = torch.randint(0, vocab_size, (1, 128), dtype=torch.long, device=device)
        timings = _timed_forward(model, ids, 3)
        thread_rows.append(
            {
                "threads": threads,
                "latency_ms": round(mean(timings), 6),
                "tokens_per_second": round(128 / (mean(timings) / 1000.0), 6),
            }
        )
    torch.set_num_threads(default_threads)

    cache_rows = []
    for index, source in enumerate(validation_sources[:6], start=1):
        cached = _generate(runtime, source, use_cache=True, max_new_tokens=64)
        uncached = _generate(runtime, source, use_cache=False, max_new_tokens=64)
        cache_rows.append(
            {
                "index": index,
                "cached_ms": round(cached.latency_ms, 6),
                "uncached_ms": round(uncached.latency_ms, 6),
                "speedup": round(uncached.latency_ms / cached.latency_ms, 6) if cached.latency_ms else None,
                "output_equal": cached.text == uncached.text,
                "cached_tokens": cached.generated_tokens,
                "uncached_tokens": uncached.generated_tokens,
            }
        )

    tokenizer_texts = list(validation_sources[:1000])
    started = time.perf_counter()
    token_count = 0
    for text in tokenizer_texts:
        token_count += len(runtime.tokenizer.encode(text, add_bos=True, add_eos=False))
    tokenizer_seconds = time.perf_counter() - started
    return {
        "forward_by_sequence_length": forward_rows,
        "forward_by_batch_size": batch_rows,
        "forward_by_thread_count": thread_rows,
        "kv_cache_comparison": cache_rows,
        "kv_cache_mean_speedup": round(mean(row["speedup"] for row in cache_rows), 6),
        "kv_cache_output_match_rate": round(mean(row["output_equal"] for row in cache_rows), 6),
        "tokenizer": {
            "texts": len(tokenizer_texts),
            "tokens": token_count,
            "elapsed_seconds": round(tokenizer_seconds, 6),
            "tokens_per_second": round(token_count / tokenizer_seconds, 6),
        },
        "memory_after_microbench": process_memory_bytes(),
    }


def _category_summaries(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({str(record["case_category"]) for record in records})
    return {
        category: summarize_predictions([record for record in records if record["case_category"] == category])
        for category in categories
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    raw_dir = output_dir / "raw"
    progress_path = raw_dir / "progress.json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at,
        "project_root": str(PROJECT_ROOT),
        "system": collect_system_info(PROJECT_ROOT),
        "test_suite": None if args.skip_tests else _run_tests(),
    }
    if not args.skip_audit:
        print("running dataset and near-duplicate audit", flush=True)
        result["dataset_audit"] = dataset_audit(DEFAULT_TRAIN, DEFAULT_VALIDATION)

    config_path = Path(args.config).resolve()
    config, runtime, load_seconds, memory_before, memory_after = _load_runtime(config_path, args.threads)
    result["runtime"] = {
        "config": str(config_path),
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_sha256": sha256_file(runtime.checkpoint_path),
        "checkpoint_bytes": runtime.checkpoint_path.stat().st_size,
        "tokenizer": str(runtime.tokenizer_path),
        "tokenizer_sha256": sha256_file(runtime.tokenizer_path),
        "device": str(runtime.device),
        "load_seconds": round(load_seconds, 6),
        "memory_before_load": memory_before,
        "memory_after_load": memory_after,
        "model_parameters": sum(parameter.numel() for parameter in runtime.model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in runtime.model.parameters() if parameter.requires_grad),
        "state_dict_elements": sum(tensor.numel() for tensor in runtime.model.state_dict().values()),
        "model_config": runtime.model.config.__dict__,
        "generation_config": runtime.generation_config,
    }

    pairs = load_pairs(DEFAULT_VALIDATION)
    reused_validation = False
    if args.reuse_progress:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        validation_records = list(progress.get("records", []))
        if not validation_records:
            raise ValueError(f"No reusable validation records in {progress_path}")
        if args.validation_limit > 0:
            validation_records = validation_records[: args.validation_limit]
        pairs = pairs[: len(validation_records)]
        reused_validation = True
        print(f"reusing completed validation rows={len(validation_records)}", flush=True)
    else:
        if args.validation_limit > 0:
            pairs = pairs[: args.validation_limit]
        print(f"evaluating validation rows={len(pairs)}", flush=True)
        validation_records = _evaluate_validation(runtime, pairs, progress_path)
    copy_records = copy_baseline_records(validation_records)
    result["validation"] = {
        "selection": (
            f"first {len(pairs)} rows, reused from interrupted full-validation progress"
            if reused_validation
            else ("all validation rows in file order" if args.validation_limit <= 0 else f"first {len(pairs)} rows")
        ),
        "warning": "The validation split was used for checkpoint selection; metrics are not an untouched test-set estimate.",
        "records": validation_records,
        "summary": summarize_extended(validation_records),
        "copy_baseline_summary": summarize_extended(copy_records),
        "paired_vs_copy": paired_comparison(validation_records, copy_records),
        "stratified": stratified_summaries(validation_records),
        "worst_by_sari": sorted(validation_records, key=lambda row: row["sari"])[:30],
        "worst_by_rouge_l": sorted(validation_records, key=lambda row: row["rouge_l_f1"])[:30],
    }
    _atomic_json(progress_path, {"phase": "validation_complete", "completed": len(pairs), "total": len(pairs)})

    cases = build_stress_cases()
    if args.stress_limit > 0:
        cases = cases[: args.stress_limit]
    print(f"evaluating independent stress cases={len(cases)}", flush=True)
    stress_records = _evaluate_stress(runtime, cases)
    result["stress"] = {
        "provenance": "Human-authored fixed cases and deterministic perturbations created for this evaluation; not drawn from packaged train/validation files.",
        "records": stress_records,
        "summary": summarize_extended(stress_records),
        "by_category": _category_summaries(stress_records),
        "constraints": summarize_constraints(stress_records),
        "perturbation_consistency": perturbation_consistency(stress_records),
        "worst_by_sari": sorted(stress_records, key=lambda row: row["sari"])[:25],
    }

    if not args.skip_determinism:
        result["determinism"] = _determinism_check(runtime, [pair.source for pair in pairs])
    if not args.skip_microbench:
        print("running CPU/GPU microbenchmarks", flush=True)
        result["microbench"] = _microbench(runtime, [pair.source for pair in pairs], args.threads)

    historical_report = PROJECT_ROOT / "outputs" / "sft_203m_text_simplification_pass_filtered" / "evaluation" / "report.json"
    training_summary = PROJECT_ROOT / "outputs" / "sft_203m_text_simplification_pass_filtered" / "training_summary.json"
    result["historical"] = {
        "packaged_evaluation": json.loads(historical_report.read_text(encoding="utf-8")),
        "training_summary": json.loads(training_summary.read_text(encoding="utf-8")),
        "note": "Historical figures came from the packaged Linux accelerator run and were not rerun on this CPU-only host.",
    }
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result["final_process_memory"] = process_memory_bytes()
    result_path = raw_dir / "extreme_eval_results.json"
    _atomic_json(result_path, result)
    if progress_path.exists():
        progress_path.unlink()
    print(f"results written: {result_path}", flush=True)
    return result_path


def main() -> None:
    path = run(parse_args())
    print(path)


if __name__ == "__main__":
    main()
