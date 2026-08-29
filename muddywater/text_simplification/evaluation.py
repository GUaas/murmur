from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .prompting import format_prompt, sanitize_reserved_tags


NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")


@dataclass(frozen=True)
class SimplificationPair:
    source: str
    target: str


def load_pairs(path: str | Path) -> list[SimplificationPair]:
    records: list[SimplificationPair] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            source = str(payload.get("source", "")).strip()
            target = str(payload.get("target", "")).strip()
            if not source or not target:
                raise ValueError(f"Missing source/target at line {line_no}: {path}")
            records.append(SimplificationPair(source=source, target=target))
    if not records:
        raise ValueError(f"No source-target records found: {path}")
    return records


def compact_chars(text: str) -> list[str]:
    return [char for char in str(text) if not char.isspace()]


def _safe_f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = compact_chars(prediction)
    expected = compact_chars(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = lcs_length(predicted, expected)
    return _safe_f1(overlap / len(predicted), overlap / len(expected))


def _ngrams(text: str, n: int) -> list[str]:
    chars = compact_chars(text)
    return ["".join(chars[index : index + n]) for index in range(len(chars) - n + 1)]


def chrf_score(prediction: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    precision_values: list[float] = []
    recall_values: list[float] = []
    for order in range(1, max_order + 1):
        predicted = Counter(_ngrams(prediction, order))
        expected = Counter(_ngrams(reference, order))
        if not predicted and not expected:
            continue
        overlap = sum((predicted & expected).values())
        precision_values.append(overlap / sum(predicted.values()) if predicted else 0.0)
        recall_values.append(overlap / sum(expected.values()) if expected else 0.0)
    if not precision_values:
        return float(not compact_chars(prediction) and not compact_chars(reference))
    precision = mean(precision_values)
    recall = mean(recall_values)
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    return 0.0 if denominator == 0.0 else (1 + beta_squared) * precision * recall / denominator


def _set_f1(system: set[str], reference: set[str]) -> float:
    if not system and not reference:
        return 1.0
    overlap = len(system & reference)
    precision = overlap / len(system) if system else 0.0
    recall = overlap / len(reference) if reference else 0.0
    return _safe_f1(precision, recall)


def sari_score(source: str, prediction: str, reference: str, max_order: int = 4) -> float:
    """Character n-gram SARI for Chinese, averaged over add/keep/delete."""

    order_scores: list[float] = []
    for order in range(1, max_order + 1):
        source_grams = set(_ngrams(source, order))
        predicted_grams = set(_ngrams(prediction, order))
        reference_grams = set(_ngrams(reference, order))

        add_score = _set_f1(predicted_grams - source_grams, reference_grams - source_grams)
        keep_score = _set_f1(predicted_grams & source_grams, reference_grams & source_grams)
        deleted = source_grams - predicted_grams
        should_delete = source_grams - reference_grams
        if not deleted and not should_delete:
            delete_precision = 1.0
        else:
            delete_precision = len(deleted & should_delete) / len(deleted) if deleted else 0.0
        order_scores.append((add_score + keep_score + delete_precision) / 3.0)
    return mean(order_scores) if order_scores else 0.0


def repetition_ratio(text: str, n: int = 4) -> float:
    grams = _ngrams(text, n)
    return 0.0 if not grams else (len(grams) - len(set(grams))) / len(grams)


def numeric_preservation(prediction: str, reference: str) -> tuple[float | None, float | None]:
    predicted = Counter(NUMBER_PATTERN.findall(prediction))
    expected = Counter(NUMBER_PATTERN.findall(reference))
    if not expected:
        return None, None
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values())
    return precision, recall


def score_prediction(
    source: str,
    target: str,
    prediction: str,
    *,
    finish_reason: str,
    latency_ms: float,
    generated_tokens: int,
    reserved_tags: Iterable[str],
) -> dict[str, Any]:
    source_chars = compact_chars(source)
    target_chars = compact_chars(target)
    predicted_chars = compact_chars(prediction)
    number_precision, number_recall = numeric_preservation(prediction, target)
    return {
        "source": source,
        "target": target,
        "prediction": prediction,
        "source_chars": len(source_chars),
        "target_chars": len(target_chars),
        "prediction_chars": len(predicted_chars),
        "compression_ratio": len(predicted_chars) / len(source_chars) if source_chars else 0.0,
        "reference_compression_ratio": len(target_chars) / len(source_chars) if source_chars else 0.0,
        "rouge_l_f1": rouge_l_f1(prediction, target),
        "chrf": chrf_score(prediction, target),
        "sari": sari_score(source, prediction, target),
        "exact_match": prediction.strip() == target.strip(),
        "unchanged_copy": prediction.strip() == source.strip(),
        "empty": not bool(prediction.strip()),
        "repetition_ratio": repetition_ratio(prediction),
        "number_precision": number_precision,
        "number_recall": number_recall,
        "finish_reason": finish_reason,
        "eos_or_stop_hit": finish_reason in {"eos", "stop_string"},
        "latency_ms": latency_ms,
        "generated_tokens": generated_tokens,
        "special_token_leaks": [tag for tag in reserved_tags if tag and tag in prediction],
    }


def summarize_predictions(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}

    def average(key: str) -> float:
        return round(mean(float(record[key]) for record in records), 6)

    numeric_precision = [record["number_precision"] for record in records if record["number_precision"] is not None]
    numeric_recall = [record["number_recall"] for record in records if record["number_recall"] is not None]
    return {
        "count": len(records),
        "rouge_l_f1": average("rouge_l_f1"),
        "chrf": average("chrf"),
        "sari": average("sari"),
        "compression_ratio": average("compression_ratio"),
        "reference_compression_ratio": average("reference_compression_ratio"),
        "exact_match_rate": round(sum(record["exact_match"] for record in records) / len(records), 6),
        "unchanged_copy_rate": round(sum(record["unchanged_copy"] for record in records) / len(records), 6),
        "empty_rate": round(sum(record["empty"] for record in records) / len(records), 6),
        "eos_or_stop_hit_rate": round(sum(record["eos_or_stop_hit"] for record in records) / len(records), 6),
        "special_token_leak_rate": round(sum(bool(record["special_token_leaks"]) for record in records) / len(records), 6),
        "repetition_ratio": average("repetition_ratio"),
        "number_precision": round(mean(numeric_precision), 6) if numeric_precision else None,
        "number_recall": round(mean(numeric_recall), 6) if numeric_recall else None,
        "numbered_sample_count": len(numeric_recall),
        "avg_latency_ms": average("latency_ms"),
        "avg_generated_tokens": average("generated_tokens"),
    }


def metric_delta(finetuned: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ("rouge_l_f1", "chrf", "sari", "exact_match_rate", "eos_or_stop_hit_rate")
    result: dict[str, float] = {}
    for key in keys:
        if isinstance(finetuned.get(key), (int, float)) and isinstance(baseline.get(key), (int, float)):
            result[key] = round(float(finetuned[key]) - float(baseline[key]), 6)
    return result


def finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value
