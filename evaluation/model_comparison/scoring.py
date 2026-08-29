from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

from .constraints import semantic_constraint_audit


DIMENSION_WEIGHTS = {
    "简化质量": 30,
    "事实与关键信息保持": 20,
    "简化力度": 10,
    "鲁棒性": 15,
    "长文本能力": 10,
    "推理效率": 10,
    "稳定性": 5,
}

CORE_EXCLUDED_CATEGORIES = {"perturbation", "identity", "injection", "long_context"}


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    materialized = list(values)
    return mean(materialized) if materialized else default


def _core_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for record in result["stress"]["records"]
        if record["case_category"] not in CORE_EXCLUDED_CATEGORIES
    ]


def _category_records(result: dict[str, Any], category: str) -> list[dict[str, Any]]:
    return [
        record
        for record in result["stress"]["records"]
        if record["case_category"] == category
    ]


def _f1(precision: float | None, recall: float | None) -> float:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _quality_score(result: dict[str, Any]) -> tuple[float, dict[str, float]]:
    records = _core_records(result)
    sari = _safe_mean(float(row["sari"]) for row in records)
    rouge = _safe_mean(float(row["rouge_l_f1"]) for row in records)
    chrf = _safe_mean(float(row["chrf"]) for row in records)
    score = 100 * (0.50 * sari + 0.30 * rouge + 0.20 * chrf)
    return score, {"sari": sari, "rouge_l": rouge, "chrf": chrf, "count": len(records)}


def _factual_score(result: dict[str, Any]) -> tuple[float, dict[str, float]]:
    number_records = _category_records(result, "numbers")
    precision_values = [row["number_precision"] for row in number_records if row["number_precision"] is not None]
    recall_values = [row["number_recall"] for row in number_records if row["number_recall"] is not None]
    number_precision = _safe_mean(float(value) for value in precision_values)
    number_recall = _safe_mean(float(value) for value in recall_values)
    number_f1 = _f1(number_precision, number_recall)
    constrained = [
        row
        for row in result["stress"]["records"]
        if row["constraints"]["must_keep_total"]
        and row["case_category"] not in {"long_context", "perturbation"}
    ]
    audits = [semantic_constraint_audit(row) for row in constrained]
    strict_keep_rate = _safe_mean(float(row["constraints"]["must_keep_pass"]) for row in constrained)
    keep_rate = _safe_mean(float(audit["semantic_pass"]) for audit in audits)
    score = 100 * (0.50 * number_f1 + 0.50 * keep_rate)
    return score, {
        "number_precision": number_precision,
        "number_recall": number_recall,
        "number_f1": number_f1,
        "must_keep_pass_rate": keep_rate,
        "strict_must_keep_pass_rate": strict_keep_rate,
        "normalization_or_alias_recoveries": sum(
            len(audit["recovered_by_normalization_or_alias"]) for audit in audits
        ),
        "constrained_count": len(constrained),
    }


def _simplification_score(result: dict[str, Any]) -> tuple[float, dict[str, float]]:
    records = _core_records(result)
    alignments = [
        max(
            0.0,
            1.0
            - abs(float(row["compression_ratio"]) - float(row["reference_compression_ratio"]))
            / max(float(row["reference_compression_ratio"]), 0.25),
        )
        for row in records
    ]
    alignment = _safe_mean(alignments)
    copy_avoidance = 1.0 - _safe_mean(float(row["unchanged_copy"]) for row in records)
    score = 100 * (0.60 * alignment + 0.40 * copy_avoidance)
    return score, {
        "compression_alignment": alignment,
        "copy_avoidance": copy_avoidance,
        "mean_compression_ratio": _safe_mean(float(row["compression_ratio"]) for row in records),
        "reference_compression_ratio": _safe_mean(
            float(row["reference_compression_ratio"]) for row in records
        ),
    }


def _robustness_score(result: dict[str, Any]) -> tuple[float, dict[str, float]]:
    perturbation = float(result["stress"]["perturbation_consistency"]["overall"]["mean"] or 0.0)
    constraints = result["stress"]["constraints"]
    identity = float(constraints["identity_exact_preservation_rate"] or 0.0)
    injection = float(constraints["injection_forbidden_exact_pass_rate"] or 0.0)
    score = 100 * (0.50 * perturbation + 0.25 * identity + 0.25 * injection)
    return score, {
        "perturbation_consistency": perturbation,
        "identity_preservation": identity,
        "injection_pass": injection,
    }


def _long_text_score(result: dict[str, Any]) -> tuple[float, dict[str, float]]:
    records = _category_records(result, "long_context")
    sari = _safe_mean(float(row["sari"]) for row in records)
    audits = [semantic_constraint_audit(row) for row in records]
    keep = _safe_mean(float(audit["semantic_pass"]) for audit in audits)
    strict_keep = _safe_mean(float(row["constraints"]["must_keep_pass"]) for row in records)
    nonempty = 1.0 - _safe_mean(float(row["empty"]) for row in records)
    compression_efficiency = _safe_mean(
        min(
            1.0,
            float(row["reference_compression_ratio"])
            / max(float(row["compression_ratio"]), 1e-9),
        )
        for row in records
    )
    repetition_quality = _safe_mean(
        max(0.0, 1.0 - min(1.0, float(row["repetition_ratio"]) * 2.0))
        for row in records
    )
    score = 100 * (
        0.25 * sari
        + 0.25 * keep
        + 0.25 * compression_efficiency
        + 0.20 * repetition_quality
        + 0.05 * nonempty
    )
    return score, {
        "sari": sari,
        "must_keep_pass_rate": keep,
        "strict_must_keep_pass_rate": strict_keep,
        "nonempty_rate": nonempty,
        "compression_efficiency": compression_efficiency,
        "repetition_quality": repetition_quality,
        "mean_chunk_count": _safe_mean(float(row["chunk_count"]) for row in records),
    }


def _stability_score(result: dict[str, Any]) -> tuple[float, dict[str, float]]:
    records = [
        row for row in result["stress"]["records"] if row["case_category"] != "long_context"
    ]
    determinism = float(result["determinism"]["exact_determinism_rate"])
    nonempty = 1.0 - _safe_mean(float(row["empty"]) for row in records)
    no_leak = 1.0 - _safe_mean(float(bool(row["special_token_leaks"])) for row in records)
    repetition_quality = max(
        0.0,
        1.0 - min(1.0, _safe_mean(float(row["repetition_ratio"]) for row in records) * 5.0),
    )
    score = 100 * (
        0.50 * determinism + 0.20 * nonempty + 0.15 * no_leak + 0.15 * repetition_quality
    )
    return score, {
        "determinism": determinism,
        "nonempty_rate": nonempty,
        "no_special_token_leak": no_leak,
        "repetition_quality": repetition_quality,
    }


def _performance_values(result: dict[str, Any]) -> dict[str, float]:
    records = result["validation"]["records"]
    latencies = np.asarray([float(row["latency_ms"]) for row in records], dtype=np.float64)
    total_seconds = float(latencies.sum()) / 1000.0
    generated_tokens = sum(int(row["generated_tokens"]) for row in records)
    peak_working_set = result["final_memory"].get("peak_working_set")
    return {
        "weight_mib": float(result["runtime"]["weight_bytes"]) / (1024 * 1024),
        "load_seconds": float(result["runtime"]["load_seconds"]),
        "peak_working_set_mib": float(peak_working_set or 0) / (1024 * 1024),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "decode_tokens_per_second": generated_tokens / total_seconds if total_seconds else 0.0,
        "performance_samples": len(records),
    }


def _efficiency_scores(results: Sequence[dict[str, Any]]) -> dict[str, tuple[float, dict[str, float]]]:
    values = {result["model_id"]: _performance_values(result) for result in results}
    best_low = {
        key: min(model_values[key] for model_values in values.values())
        for key in (
            "weight_mib",
            "load_seconds",
            "peak_working_set_mib",
            "latency_p50_ms",
            "latency_p95_ms",
        )
    }
    best_throughput = max(model_values["decode_tokens_per_second"] for model_values in values.values())
    output: dict[str, tuple[float, dict[str, float]]] = {}
    for model_id, model_values in values.items():
        component_scores = {
            "weight": best_low["weight_mib"] / model_values["weight_mib"],
            "load": best_low["load_seconds"] / model_values["load_seconds"],
            "memory": best_low["peak_working_set_mib"] / model_values["peak_working_set_mib"],
            "p50": best_low["latency_p50_ms"] / model_values["latency_p50_ms"],
            "p95": best_low["latency_p95_ms"] / model_values["latency_p95_ms"],
            "throughput": (
                model_values["decode_tokens_per_second"] / best_throughput
                if best_throughput
                else 0.0
            ),
        }
        score = 100 * (
            0.10 * component_scores["weight"]
            + 0.10 * component_scores["load"]
            + 0.15 * component_scores["memory"]
            + 0.20 * component_scores["p50"]
            + 0.25 * component_scores["p95"]
            + 0.20 * component_scores["throughput"]
        )
        output[model_id] = (score, {**model_values, **{f"relative_{k}": v for k, v in component_scores.items()}})
    return output


def score_models(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    efficiency = _efficiency_scores(results)
    scored: dict[str, Any] = {}
    for result in results:
        dimensions = {
            "简化质量": _quality_score(result),
            "事实与关键信息保持": _factual_score(result),
            "简化力度": _simplification_score(result),
            "鲁棒性": _robustness_score(result),
            "长文本能力": _long_text_score(result),
            "推理效率": efficiency[result["model_id"]],
            "稳定性": _stability_score(result),
        }
        overall = sum(
            dimensions[name][0] * weight / 100.0
            for name, weight in DIMENSION_WEIGHTS.items()
        )
        scored[result["model_id"]] = {
            "display_name": result["display_name"],
            "overall": round(overall, 3),
            "dimensions": {
                name: {"score": round(payload[0], 3), "details": payload[1]}
                for name, payload in dimensions.items()
            },
        }
    return {"weights": DIMENSION_WEIGHTS, "models": scored}


def paired_statistics(
    v1: dict[str, Any],
    v2: dict[str, Any],
    *,
    resamples: int = 10000,
    seed: int = 20260815,
) -> dict[str, Any]:
    left = {row["case_id"]: row for row in _core_records(v1)}
    right = {row["case_id"]: row for row in _core_records(v2)}
    ids = sorted(set(left) & set(right))
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {"case_count": len(ids), "metrics": {}}
    for metric in ("sari", "rouge_l_f1", "chrf"):
        deltas = np.asarray(
            [float(right[case_id][metric]) - float(left[case_id][metric]) for case_id in ids],
            dtype=np.float64,
        )
        indices = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
        bootstrap = deltas[indices].mean(axis=1)
        lower, upper = np.percentile(bootstrap, [2.5, 97.5])
        output["metrics"][metric] = {
            "v2_minus_v1": round(float(deltas.mean()), 6),
            "lower_95": round(float(lower), 6),
            "upper_95": round(float(upper), 6),
            "v2_win_rate": round(float((deltas > 1e-12).mean()), 6),
            "tie_rate": round(float((np.abs(deltas) <= 1e-12).mean()), 6),
            "v1_win_rate": round(float((deltas < -1e-12).mean()), 6),
            "bootstrap_probability_v2_better": round(float((bootstrap > 0).mean()), 6),
            "resamples": resamples,
        }
    return output
