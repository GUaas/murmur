from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

from muddywater.text_simplification.evaluation import (
    rouge_l_f1,
    score_prediction,
    summarize_predictions,
)

from .types import EvalCase


REFERENCE_METRICS = ("rouge_l_f1", "chrf", "sari")


def percentile_summary(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": round(float(array.mean()), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p90": round(float(np.percentile(array, 90)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "p99": round(float(np.percentile(array, 99)), 6),
        "max": round(float(array.max()), 6),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 20260814,
    resamples: int = 2000,
) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "lower_95": None, "upper_95": None, "resamples": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 100
    for start in range(0, resamples, chunk):
        width = min(chunk, resamples - start)
        indices = rng.integers(0, array.size, size=(width, array.size))
        means[start : start + width] = array[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "mean": round(float(array.mean()), 6),
        "lower_95": round(float(lower), 6),
        "upper_95": round(float(upper), 6),
        "resamples": resamples,
    }


def summarize_extended(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_predictions(records)
    summary["latency_ms_percentiles"] = percentile_summary(
        float(record["latency_ms"]) for record in records
    )
    total_seconds = sum(float(record["latency_ms"]) for record in records) / 1000.0
    total_tokens = sum(int(record["generated_tokens"]) for record in records)
    summary["decode_tokens_per_second"] = round(total_tokens / total_seconds, 6) if total_seconds else None
    summary["bootstrap_95_ci"] = {
        key: bootstrap_mean_ci([float(record[key]) for record in records])
        for key in REFERENCE_METRICS
    }
    return summary


def copy_baseline_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        score_prediction(
            str(record["source"]),
            str(record["target"]),
            str(record["source"]),
            finish_reason="baseline_copy",
            latency_ms=0.0,
            generated_tokens=0,
            reserved_tags=("<|im_start|>", "<|im_end|>"),
        )
        for record in records
    ]


def paired_comparison(
    model_records: Sequence[dict[str, Any]],
    baseline_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in REFERENCE_METRICS:
        deltas = np.asarray(
            [float(model[key]) - float(base[key]) for model, base in zip(model_records, baseline_records)],
            dtype=np.float64,
        )
        result[key] = {
            "mean_delta": round(float(deltas.mean()), 6),
            "win_rate": round(float((deltas > 1e-12).mean()), 6),
            "tie_rate": round(float((np.abs(deltas) <= 1e-12).mean()), 6),
            "loss_rate": round(float((deltas < -1e-12).mean()), 6),
        }
    return result


def _length_bucket(chars: int) -> str:
    if chars <= 32:
        return "01_<=32"
    if chars <= 64:
        return "02_33-64"
    if chars <= 128:
        return "03_65-128"
    if chars <= 256:
        return "04_129-256"
    return "05_>256"


def stratified_summaries(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "source_length": defaultdict(list),
        "reference_action": defaultdict(list),
        "contains_number": defaultdict(list),
    }
    for record in records:
        groups["source_length"][_length_bucket(int(record["source_chars"]))].append(record)
        source = str(record["source"]).strip()
        target = str(record["target"]).strip()
        if source == target:
            action = "identity"
        elif int(record["target_chars"]) > int(record["source_chars"]):
            action = "expand"
        else:
            action = "simplify"
        groups["reference_action"][action].append(record)
        groups["contains_number"]["yes" if record["number_recall"] is not None else "no"].append(record)
    return {
        dimension: {bucket: summarize_predictions(items) for bucket, items in sorted(buckets.items())}
        for dimension, buckets in groups.items()
    }


def score_constraints(case: EvalCase, prediction: str) -> dict[str, Any]:
    missing = [item for item in case.must_keep if item not in prediction]
    forbidden_hits = [item for item in case.forbidden_exact if prediction.strip() == item.strip()]
    identity_pass = None
    if case.expect_unchanged:
        identity_pass = prediction.strip() == case.source.strip()
    return {
        "must_keep_total": len(case.must_keep),
        "must_keep_missing": missing,
        "must_keep_pass": not missing,
        "forbidden_exact_hits": forbidden_hits,
        "injection_pass": not forbidden_hits,
        "identity_pass": identity_pass,
    }


def summarize_constraints(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    with_keep = [record for record in records if record["constraints"]["must_keep_total"]]
    injection = [record for record in records if record["case_category"] == "injection"]
    identity = [record for record in records if record["constraints"]["identity_pass"] is not None]
    return {
        "must_keep_case_count": len(with_keep),
        "must_keep_pass_rate": round(mean(r["constraints"]["must_keep_pass"] for r in with_keep), 6) if with_keep else None,
        "injection_case_count": len(injection),
        "injection_forbidden_exact_pass_rate": round(mean(r["constraints"]["injection_pass"] for r in injection), 6) if injection else None,
        "identity_case_count": len(identity),
        "identity_exact_preservation_rate": round(mean(r["constraints"]["identity_pass"] for r in identity), 6) if identity else None,
    }


def perturbation_consistency(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(record["case_id"]): record for record in records}
    scores: dict[str, list[float]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    for record in records:
        group = record.get("perturbation_group")
        if not group or group not in by_id:
            continue
        base = by_id[group]
        score = rouge_l_f1(str(record["prediction"]), str(base["prediction"]))
        name = str(record.get("perturbation") or "unknown")
        scores[name].append(score)
        details.append({"case_id": record["case_id"], "base_id": group, "perturbation": name, "output_rouge_l": round(score, 6)})
    return {
        "overall": percentile_summary(score for values in scores.values() for score in values),
        "by_perturbation": {name: percentile_summary(values) for name, values in sorted(scores.items())},
        "details": details,
    }

