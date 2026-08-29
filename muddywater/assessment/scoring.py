from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .types import ChoiceExample, TextProbe


@dataclass
class _PreparedContinuation:
    example_index: int
    option_index: int
    input_ids: list[int]
    target_positions: list[int]
    truncated_tokens: int


def _prepare_continuation(
    tokenizer: Any,
    context: str,
    continuation: str,
    max_seq_len: int,
    example_index: int,
    option_index: int,
) -> _PreparedContinuation:
    full_text = context + continuation
    ids, spans = tokenizer.encode_with_char_spans(
        full_text,
        add_bos=True,
        add_eos=False,
    )
    boundary = len(context)
    target_positions = [
        index
        for index, span in enumerate(spans)
        if span is not None and int(span[1]) > boundary
    ]
    if not target_positions:
        raise ValueError(
            f"Continuation produced no scoreable tokens: {continuation!r}"
        )

    truncated_tokens = max(0, len(ids) - int(max_seq_len))
    if truncated_tokens:
        ids = ids[truncated_tokens:]
        target_positions = [
            index - truncated_tokens
            for index in target_positions
            if index - truncated_tokens > 0
        ]
    if not target_positions:
        raise ValueError("All continuation tokens were lost to context truncation")
    return _PreparedContinuation(
        example_index=example_index,
        option_index=option_index,
        input_ids=[int(value) for value in ids],
        target_positions=target_positions,
        truncated_tokens=truncated_tokens,
    )


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


@torch.inference_mode()
def score_choice_examples(
    model: Any,
    tokenizer: Any,
    examples: list[ChoiceExample],
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Score choices with raw and per-token normalized log likelihood."""

    if not examples:
        return {"num_examples": 0, "predictions": []}
    model.eval()
    max_seq_len = int(model.config.max_seq_len)
    prepared: list[_PreparedContinuation] = []
    for example_index, example in enumerate(examples):
        for option_index, option in enumerate(example.options):
            prepared.append(
                _prepare_continuation(
                    tokenizer,
                    example.context,
                    option,
                    max_seq_len=max_seq_len,
                    example_index=example_index,
                    option_index=option_index,
                )
            )

    scores: dict[tuple[int, int], dict[str, float | int]] = {}
    start_time = time.perf_counter()
    forwarded_tokens = 0
    for batch in _chunks(prepared, max(1, int(batch_size))):
        longest = max(len(item.input_ids) for item in batch)
        input_ids = torch.full(
            (len(batch), longest),
            int(tokenizer.pad_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(batch), longest),
            dtype=torch.bool,
            device=device,
        )
        for row_index, item in enumerate(batch):
            length = len(item.input_ids)
            input_ids[row_index, :length] = torch.tensor(
                item.input_ids,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row_index, :length] = True
            forwarded_tokens += length

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"].float()
        for row_index, item in enumerate(batch):
            positions = torch.tensor(
                item.target_positions,
                dtype=torch.long,
                device=device,
            )
            prediction_positions = positions - 1
            target_ids = input_ids[row_index, positions]
            selected_logits = logits[row_index, prediction_positions]
            target_logits = selected_logits.gather(1, target_ids[:, None]).squeeze(1)
            log_probabilities = target_logits - torch.logsumexp(
                selected_logits,
                dim=-1,
            )
            raw_score = float(log_probabilities.sum().item())
            token_count = int(log_probabilities.numel())
            scores[(item.example_index, item.option_index)] = {
                "log_likelihood": raw_score,
                "mean_log_likelihood": raw_score / max(1, token_count),
                "scored_tokens": token_count,
                "sequence_tokens": len(item.input_ids),
                "truncated_tokens": item.truncated_tokens,
            }
        del logits, outputs

    elapsed = time.perf_counter() - start_time
    predictions: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        option_scores = [scores[(example_index, index)] for index in range(len(example.options))]
        raw_values = [float(item["log_likelihood"]) for item in option_scores]
        norm_values = [float(item["mean_log_likelihood"]) for item in option_scores]
        raw_prediction = max(range(len(raw_values)), key=raw_values.__getitem__)
        norm_prediction = max(range(len(norm_values)), key=norm_values.__getitem__)
        primary_values = norm_values if example.primary_metric == "accuracy_norm" else raw_values
        primary_prediction = (
            norm_prediction if example.primary_metric == "accuracy_norm" else raw_prediction
        )
        probabilities = _softmax(primary_values)
        predictions.append(
            {
                "task": example.task,
                "example_id": example.example_id,
                "category": example.category,
                "answer_index": int(example.answer_index),
                "prediction": raw_prediction,
                "prediction_norm": norm_prediction,
                "primary_prediction": primary_prediction,
                "primary_metric": example.primary_metric,
                "correct": raw_prediction == int(example.answer_index),
                "correct_norm": norm_prediction == int(example.answer_index),
                "primary_correct": primary_prediction == int(example.answer_index),
                "confidence": max(probabilities),
                "correct_probability": probabilities[int(example.answer_index)],
                "option_scores": option_scores,
                "metadata": example.metadata,
            }
        )

    return summarize_choice_predictions(
        predictions,
        elapsed_seconds=elapsed,
        forwarded_tokens=forwarded_tokens,
    )


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _expected_calibration_error(predictions: list[dict[str, Any]], bins: int = 10) -> float:
    total = len(predictions)
    if not total:
        return float("nan")
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            item
            for item in predictions
            if lower <= float(item["confidence"]) <= upper
            and (bin_index == bins - 1 or float(item["confidence"]) < upper)
        ]
        if not members:
            continue
        accuracy = sum(bool(item["primary_correct"]) for item in members) / len(members)
        confidence = sum(float(item["confidence"]) for item in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def _mean_centered_accuracy(
    predictions: list[dict[str, Any]],
    score_field: str,
) -> dict[str, Any]:
    """Remove each option index's unsupervised mean score before argmax.

    This is a diagnostic for label/position priors, not an official task metric.
    It uses no answer labels when estimating the centering constants.
    """

    if not predictions:
        return {"accuracy": float("nan"), "option_score_means": []}
    max_options = max(len(item["option_scores"]) for item in predictions)
    option_means = []
    for option_index in range(max_options):
        values = [
            float(item["option_scores"][option_index][score_field])
            for item in predictions
            if option_index < len(item["option_scores"])
        ]
        option_means.append(sum(values) / len(values))
    predictions_after_centering = []
    correct = 0
    distribution: dict[int, int] = defaultdict(int)
    for item in predictions:
        values = [
            float(option[score_field]) - option_means[index]
            for index, option in enumerate(item["option_scores"])
        ]
        prediction = max(range(len(values)), key=values.__getitem__)
        predictions_after_centering.append(prediction)
        distribution[prediction] += 1
        correct += prediction == int(item["answer_index"])
    return {
        "accuracy": correct / len(predictions),
        "wilson_95": wilson_interval(correct, len(predictions)),
        "option_score_means": option_means,
        "prediction_distribution": dict(sorted(distribution.items())),
        "uses_answer_labels_for_centering": False,
        "official_metric": False,
    }


def summarize_choice_predictions(
    predictions: list[dict[str, Any]],
    elapsed_seconds: float,
    forwarded_tokens: int,
) -> dict[str, Any]:
    total = len(predictions)
    raw_correct = sum(bool(item["correct"]) for item in predictions)
    norm_correct = sum(bool(item["correct_norm"]) for item in predictions)
    primary_correct = sum(bool(item["primary_correct"]) for item in predictions)
    chance = (
        sum(1.0 / len(item["option_scores"]) for item in predictions) / total
        if total
        else float("nan")
    )
    primary_accuracy = primary_correct / total if total else float("nan")

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        by_category[str(item["category"])].append(item)
    category_summary = {}
    for category, items in sorted(by_category.items()):
        successes = sum(bool(item["primary_correct"]) for item in items)
        category_summary[category] = {
            "num_examples": len(items),
            "primary_accuracy": successes / len(items),
            "wilson_95": wilson_interval(successes, len(items)),
        }

    return {
        "num_examples": total,
        "primary_metric": predictions[0]["primary_metric"] if predictions else None,
        "accuracy": raw_correct / total if total else float("nan"),
        "accuracy_norm": norm_correct / total if total else float("nan"),
        "primary_accuracy": primary_accuracy,
        "primary_wilson_95": wilson_interval(primary_correct, total),
        "chance_accuracy": chance,
        "chance_adjusted_percent": (
            100.0 * (primary_accuracy - chance) / (1.0 - chance)
            if total and chance < 1.0
            else float("nan")
        ),
        "mean_correct_option_probability": (
            sum(float(item["correct_probability"]) for item in predictions) / total
            if total
            else float("nan")
        ),
        "expected_calibration_error_10_bin": _expected_calibration_error(predictions),
        "mean_centered_raw_diagnostic": _mean_centered_accuracy(
            predictions,
            "log_likelihood",
        ),
        "mean_centered_norm_diagnostic": _mean_centered_accuracy(
            predictions,
            "mean_log_likelihood",
        ),
        "truncated_option_count": sum(
            int(option["truncated_tokens"]) > 0
            for item in predictions
            for option in item["option_scores"]
        ),
        "elapsed_seconds": elapsed_seconds,
        "forwarded_tokens": int(forwarded_tokens),
        "forwarded_tokens_per_second": forwarded_tokens / max(elapsed_seconds, 1e-9),
        "by_category": category_summary,
        "predictions": predictions,
    }


@torch.inference_mode()
def score_text_probes(
    model: Any,
    tokenizer: Any,
    probes: list[TextProbe],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    results = []
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    for probe in probes:
        ids = tokenizer.encode(probe.text, add_bos=True, add_eos=True)
        truncated_tokens = max(0, len(ids) - (int(model.config.max_seq_len) + 1))
        if truncated_tokens:
            ids = ids[truncated_tokens:]
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        labels = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        outputs = model(input_ids, labels=labels, return_logits=False)
        loss_sum = float(outputs["ce_loss_sum"].item())
        token_count = int(outputs["loss_token_count"].item())
        byte_count = len(probe.text.encode("utf-8"))
        ce = loss_sum / max(1, token_count)
        results.append(
            {
                "probe_id": probe.probe_id,
                "category": probe.category,
                "language": probe.language,
                "characters": len(probe.text),
                "utf8_bytes": byte_count,
                "tokens": token_count,
                "characters_per_token": len(probe.text) / max(1, token_count),
                "bytes_per_token": byte_count / max(1, token_count),
                "ce_loss": ce,
                "perplexity": math.exp(min(ce, 50.0)),
                "bits_per_token": ce / math.log(2.0),
                "bits_per_byte": loss_sum / max(1, byte_count) / math.log(2.0),
                "truncated_tokens": truncated_tokens,
                "unknown_tokens": int(tokenizer.count_unknown_tokens(probe.text)),
            }
        )
        total_loss += loss_sum
        total_tokens += token_count
        total_bytes += byte_count
    aggregate_ce = total_loss / max(1, total_tokens)
    return {
        "num_probes": len(results),
        "aggregate": {
            "tokens": total_tokens,
            "utf8_bytes": total_bytes,
            "ce_loss": aggregate_ce,
            "perplexity": math.exp(min(aggregate_ce, 50.0)),
            "bits_per_token": aggregate_ce / math.log(2.0),
            "bits_per_byte": total_loss / max(1, total_bytes) / math.log(2.0),
        },
        "probes": results,
    }


def tokenizer_probe_stats(tokenizer: Any, probes: list[TextProbe]) -> dict[str, Any]:
    output = []
    for probe in probes:
        ids = tokenizer.encode(probe.text, add_bos=False, add_eos=False)
        byte_piece_count = sum(
            str(tokenizer.processor.IdToPiece(int(token_id))).startswith("<0x")
            for token_id in ids
        )
        whitespace_words = len(probe.text.split())
        output.append(
            {
                "probe_id": probe.probe_id,
                "category": probe.category,
                "language": probe.language,
                "characters": len(probe.text),
                "utf8_bytes": len(probe.text.encode("utf-8")),
                "whitespace_words": whitespace_words,
                "tokens": len(ids),
                "characters_per_token": len(probe.text) / max(1, len(ids)),
                "tokens_per_word": len(ids) / max(1, whitespace_words),
                "byte_fallback_tokens": byte_piece_count,
                "byte_fallback_ratio": byte_piece_count / max(1, len(ids)),
                "unknown_tokens": int(tokenizer.count_unknown_tokens(probe.text)),
                "round_trip_exact": tokenizer.decode(ids) == probe.text,
            }
        )
    return {"num_probes": len(output), "probes": output}
