from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Sequence

from extreme_eval.metrics import paired_comparison, percentile_summary, summarize_extended
from muddywater.text_simplification.evaluation import score_prediction
from muddywater.text_simplification.inference import SimplificationResult

from .types import LongDocument


def _constraint_result(expected: Iterable[str], prediction: str) -> dict[str, Any]:
    items = tuple(expected)
    missing = [item for item in items if item not in prediction]
    return {
        "total": len(items),
        "kept": len(items) - len(missing),
        "missing": missing,
        "pass": not missing,
    }


def score_document(
    document: LongDocument,
    result: SimplificationResult,
    *,
    mode: str,
    chunk_budget: int | None,
) -> dict[str, Any]:
    finish_reasons = [chunk.finish_reason for chunk in result.chunks]
    all_finished = all(reason in {"eos", "stop_string"} for reason in finish_reasons)
    aggregate_finish = "eos" if all_finished else "length"
    generated_tokens = sum(chunk.generated_tokens for chunk in result.chunks)
    record = score_prediction(
        document.source,
        document.target,
        result.text,
        finish_reason=aggregate_finish,
        latency_ms=result.elapsed_ms,
        generated_tokens=generated_tokens,
        reserved_tags=("<|im_start|>", "<|im_end|>"),
    )
    constraints = _constraint_result(document.must_keep, result.text)
    tail = _constraint_result(document.tail_keep, result.text)
    record.update(
        {
            "document_id": document.document_id,
            "category": document.category,
            "length_tier": document.length_tier,
            "layout": document.layout,
            "sentence_count": document.sentence_count,
            "provenance": document.provenance,
            "mode": mode,
            "chunk_budget": chunk_budget,
            "source_prompt_tokens": result.source_prompt_tokens,
            "chunk_count": len(result.chunks),
            "max_chunk_prompt_tokens": max((chunk.prompt_tokens for chunk in result.chunks), default=0),
            "chunk_prompt_tokens": [chunk.prompt_tokens for chunk in result.chunks],
            "chunk_finish_reasons": finish_reasons,
            "all_chunks_finished": all_finished,
            "length_finished_chunks": sum(reason == "length" for reason in finish_reasons),
            "fallback_chunks": sum(chunk.used_fallback for chunk in result.chunks),
            "generation_tokens_per_second": round(
                generated_tokens / (result.elapsed_ms / 1000.0), 6
            )
            if result.elapsed_ms
            else None,
            "constraints": constraints,
            "tail_constraints": tail,
            "newline_count_source": document.source.count("\n"),
            "newline_count_output": result.text.count("\n"),
            "newline_layout_exact": document.source.count("\n") == result.text.count("\n"),
        }
    )
    return record


def summarize_long_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_extended(records)
    constraint_total = sum(int(row["constraints"]["total"]) for row in records)
    constraint_kept = sum(int(row["constraints"]["kept"]) for row in records)
    tail_total = sum(int(row["tail_constraints"]["total"]) for row in records)
    tail_kept = sum(int(row["tail_constraints"]["kept"]) for row in records)
    total_seconds = sum(float(row["latency_ms"]) for row in records) / 1000.0
    total_generated = sum(int(row["generated_tokens"]) for row in records)
    summary.update(
        {
            "constraint_document_pass_rate": round(
                mean(bool(row["constraints"]["pass"]) for row in records), 6
            ),
            "constraint_item_recall": round(constraint_kept / constraint_total, 6)
            if constraint_total
            else None,
            "constraint_items": constraint_total,
            "tail_document_pass_rate": round(
                mean(bool(row["tail_constraints"]["pass"]) for row in records), 6
            ),
            "tail_item_recall": round(tail_kept / tail_total, 6) if tail_total else None,
            "tail_items": tail_total,
            "all_chunks_finished_rate": round(
                mean(bool(row["all_chunks_finished"]) for row in records), 6
            ),
            "newline_layout_exact_rate": round(
                mean(bool(row["newline_layout_exact"]) for row in records), 6
            ),
            "fallback_chunk_total": sum(int(row["fallback_chunks"]) for row in records),
            "length_finished_chunk_total": sum(
                int(row["length_finished_chunks"]) for row in records
            ),
            "chunk_count_percentiles": percentile_summary(
                int(row["chunk_count"]) for row in records
            ),
            "prompt_tokens_percentiles": percentile_summary(
                int(row["source_prompt_tokens"]) for row in records
            ),
            "corpus_generated_tokens_per_second": round(total_generated / total_seconds, 6)
            if total_seconds
            else None,
            "total_latency_seconds": round(total_seconds, 6),
        }
    )
    return summary


def summarize_by(records: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row[key])].append(row)
    return {name: summarize_long_records(rows) for name, rows in sorted(groups.items())}


def compare_modes(
    direct_records: Sequence[dict[str, Any]],
    chunked_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    direct_by_id = {str(row["document_id"]): row for row in direct_records}
    chunked_by_id = {str(row["document_id"]): row for row in chunked_records}
    ids = [document_id for document_id in direct_by_id if document_id in chunked_by_id]
    paired_direct = [direct_by_id[document_id] for document_id in ids]
    paired_chunked = [chunked_by_id[document_id] for document_id in ids]
    details = []
    for document_id, direct, chunked in zip(ids, paired_direct, paired_chunked):
        details.append(
            {
                "document_id": document_id,
                "prompt_tokens": direct["source_prompt_tokens"],
                "direct_latency_ms": direct["latency_ms"],
                "chunked_latency_ms": chunked["latency_ms"],
                "latency_speedup": round(
                    float(direct["latency_ms"]) / float(chunked["latency_ms"]), 6
                )
                if chunked["latency_ms"]
                else None,
                "rouge_l_delta": round(
                    float(chunked["rouge_l_f1"]) - float(direct["rouge_l_f1"]), 6
                ),
                "chrf_delta": round(float(chunked["chrf"]) - float(direct["chrf"]), 6),
                "sari_delta": round(float(chunked["sari"]) - float(direct["sari"]), 6),
                "repetition_delta": round(
                    float(chunked["repetition_ratio"]) - float(direct["repetition_ratio"]), 6
                ),
                "number_recall_delta": (
                    round(float(chunked["number_recall"]) - float(direct["number_recall"]), 6)
                    if direct["number_recall"] is not None and chunked["number_recall"] is not None
                    else None
                ),
                "direct_tail_pass": direct["tail_constraints"]["pass"],
                "chunked_tail_pass": chunked["tail_constraints"]["pass"],
                "direct_all_finished": direct["all_chunks_finished"],
                "chunked_all_finished": chunked["all_chunks_finished"],
                "chunks": chunked["chunk_count"],
            }
        )
    return {
        "paired_count": len(ids),
        "reference_metrics": paired_comparison(paired_chunked, paired_direct),
        "mean_latency_speedup": round(
            mean(float(row["latency_speedup"]) for row in details), 6
        ),
        "median_latency_speedup": percentile_summary(
            float(row["latency_speedup"]) for row in details
        )["p50"],
        "chunked_faster_rate": round(
            mean(float(row["latency_speedup"]) > 1.0 for row in details), 6
        ),
        "tail_recovery_count": sum(
            not row["direct_tail_pass"] and row["chunked_tail_pass"] for row in details
        ),
        "finish_recovery_count": sum(
            not row["direct_all_finished"] and row["chunked_all_finished"] for row in details
        ),
        "details": details,
    }
