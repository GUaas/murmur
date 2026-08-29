from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .adapters import GenerationMeasurement, ModelAdapter, build_adapter
from .paths import EvaluationPaths
from .protocol import load_stress_cases, load_validation_sample, protocol_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one simplification model release.")
    parser.add_argument("--model", choices=("v1", "v2"), required=True)
    parser.add_argument("--output-dir", default="model_comparison_results")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _system_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
    }


def _score_record(
    paths: EvaluationPaths,
    source: str,
    target: str,
    measurement: GenerationMeasurement,
) -> dict[str, Any]:
    from muddywater.text_simplification.evaluation import score_prediction

    record = score_prediction(
        source,
        target,
        measurement.text,
        finish_reason=measurement.finish_reason,
        latency_ms=measurement.latency_ms,
        generated_tokens=measurement.generated_tokens,
        reserved_tags=("<|im_start|>", "<|im_end|>"),
    )
    record.update(
        {
            "prompt_tokens": measurement.prompt_tokens,
            "used_chunking": measurement.used_chunking,
            "chunk_count": measurement.chunk_count,
            "fallback_count": measurement.fallback_count,
        }
    )
    return record


def _evaluate_stress(
    adapter: ModelAdapter,
    paths: EvaluationPaths,
    cases: Sequence[Any],
    existing: list[dict[str, Any]],
    save_progress: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None],
    validation_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from extreme_eval.metrics import score_constraints

    records = list(existing)
    started = time.perf_counter()
    for index, case in enumerate(cases[len(records) :], start=len(records) + 1):
        measurement = adapter.simplify(case.source)
        record = _score_record(paths, case.source, case.target, measurement)
        record.update(
            {
                "index": index,
                "case_id": case.case_id,
                "case_category": case.category,
                "perturbation_group": case.perturbation_group,
                "perturbation": case.perturbation,
                "constraints": score_constraints(case, measurement.text),
            }
        )
        records.append(record)
        if index % 10 == 0 or index == len(cases):
            elapsed = time.perf_counter() - started
            print(
                f"{adapter.model_id} stress {index}/{len(cases)} | "
                f"segment_elapsed={elapsed:.1f}s",
                flush=True,
            )
        if index % 10 == 0:
            save_progress(records, validation_records)
    return records


def _evaluate_validation(
    adapter: ModelAdapter,
    paths: EvaluationPaths,
    items: Sequence[Any],
    existing: list[dict[str, Any]],
    save_progress: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None],
    stress_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = list(existing)
    started = time.perf_counter()
    for position, item in enumerate(items[len(records) :], start=len(records) + 1):
        measurement = adapter.simplify(item.source)
        record = _score_record(paths, item.source, item.target, measurement)
        record.update({"position": position, "dataset_index": item.dataset_index})
        records.append(record)
        if position % 25 == 0 or position == len(items):
            elapsed = time.perf_counter() - started
            print(
                f"{adapter.model_id} validation {position}/{len(items)} | "
                f"segment_elapsed={elapsed:.1f}s",
                flush=True,
            )
        if position % 25 == 0:
            save_progress(stress_records, records)
    return records


def _summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from extreme_eval.metrics import summarize_extended

    summary = summarize_extended(records)
    summary["chunking_rate"] = round(mean(bool(row["used_chunking"]) for row in records), 6)
    summary["mean_chunk_count"] = round(mean(int(row["chunk_count"]) for row in records), 6)
    summary["fallback_total"] = sum(int(row["fallback_count"]) for row in records)
    return summary


def _stress_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from extreme_eval.metrics import (
        copy_baseline_records,
        paired_comparison,
        perturbation_consistency,
        summarize_constraints,
    )

    categories = sorted({str(record["case_category"]) for record in records})
    return {
        "summary": _summarize_records(records),
        "by_category": {
            category: _summarize_records(
                [record for record in records if record["case_category"] == category]
            )
            for category in categories
        },
        "constraints": summarize_constraints(records),
        "perturbation_consistency": perturbation_consistency(records),
        "paired_vs_copy": paired_comparison(records, copy_baseline_records(records)),
    }


def _determinism(adapter: ModelAdapter, sources: Sequence[str]) -> dict[str, Any]:
    details = []
    for index, source in enumerate(sources[:8], start=1):
        outputs = [adapter.simplify(source).text for _ in range(3)]
        details.append(
            {
                "index": index,
                "outputs": outputs,
                "exact_match": len(set(outputs)) == 1,
            }
        )
        print(f"{adapter.model_id} determinism {index}/8", flush=True)
    return {
        "cases": len(details),
        "repeats_per_case": 3,
        "exact_determinism_rate": round(mean(row["exact_match"] for row in details), 6),
        "details": details,
    }


def run(args: argparse.Namespace) -> Path:
    paths = EvaluationPaths.discover(args.output_dir)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    result_path = paths.raw_dir / f"{args.model}_results.json"
    progress_path = paths.raw_dir / f"{args.model}_progress.json"
    if args.resume and result_path.is_file():
        print(f"completed result already exists: {result_path}", flush=True)
        return result_path

    stress_cases = load_stress_cases(paths)
    validation_items = load_validation_sample(paths, args.validation_size)
    progress = {"stress_records": [], "validation_records": []}
    if args.resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        print(
            f"resuming {args.model}: stress={len(progress['stress_records'])}, "
            f"validation={len(progress['validation_records'])}",
            flush=True,
        )

    def save_progress(
        stress_records: list[dict[str, Any]],
        validation_records: list[dict[str, Any]],
    ) -> None:
        _atomic_json(
            progress_path,
            {
                "model_id": args.model,
                "protocol": protocol_manifest(stress_cases, validation_items),
                "stress_records": stress_records,
                "validation_records": validation_records,
            },
        )

    print(f"loading {args.model}", flush=True)
    adapter = build_adapter(args.model, paths, args.threads)
    metadata = adapter.metadata()
    warmup = adapter.simplify("天气很好，我们按原计划出发。")
    print(f"{args.model} warmup={warmup.latency_ms:.1f}ms", flush=True)
    started = time.perf_counter()
    stress_records = _evaluate_stress(
        adapter,
        paths,
        stress_cases,
        list(progress.get("stress_records", [])),
        save_progress,
        list(progress.get("validation_records", [])),
    )
    validation_records = _evaluate_validation(
        adapter,
        paths,
        validation_items,
        list(progress.get("validation_records", [])),
        save_progress,
        stress_records,
    )
    determinism_sources = [
        case.source
        for case in stress_cases
        if case.category not in {"perturbation", "identity", "injection", "long_context"}
    ]
    determinism = _determinism(adapter, determinism_sources)
    elapsed_seconds = time.perf_counter() - started

    result = {
        "schema_version": 1,
        "model_id": args.model,
        "display_name": adapter.display_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": _system_info(),
        "protocol": protocol_manifest(stress_cases, validation_items),
        "runtime": metadata,
        "warmup": asdict(warmup),
        "stress": {
            "records": stress_records,
            **_stress_summary(stress_records),
        },
        "validation": {
            "records": validation_records,
            "summary": _summarize_records(validation_records),
            "warning": (
                "Version 2 used this validation split for checkpoint selection; "
                "these metrics are excluded from the overall score."
            ),
        },
        "determinism": determinism,
        "evaluation_elapsed_seconds": round(elapsed_seconds, 6),
        "final_memory": _final_memory(),
    }
    _atomic_json(result_path, result)
    if progress_path.exists():
        progress_path.unlink()
    print(f"wrote {result_path}", flush=True)
    return result_path


def _final_memory() -> dict[str, int | None]:
    try:
        from extreme_eval.system_info import process_memory_bytes

        return process_memory_bytes()
    except Exception:
        return {"working_set": None, "peak_working_set": None, "private_usage": None}


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
