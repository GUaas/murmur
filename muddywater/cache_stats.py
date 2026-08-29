from __future__ import annotations

from typing import Any

import numpy as np


def document_lengths(doc_starts: np.ndarray, num_tokens: int) -> np.ndarray:
    """Return token lengths for documents described by absolute start offsets."""
    starts = np.asarray(doc_starts, dtype=np.uint64)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("doc_starts must be a non-empty 1D array")
    if int(starts[0]) != 0:
        raise ValueError("first document start must be 0")
    if np.any(starts[1:] <= starts[:-1]):
        raise ValueError("doc_starts must be strictly increasing")
    total = int(num_tokens)
    if int(starts[-1]) >= total:
        raise ValueError("last document start must be inside the token cache")
    ends = np.concatenate((starts[1:], np.asarray([total], dtype=np.uint64)))
    return (ends - starts).astype(np.int64, copy=False)


def summarize_document_coverage(
    doc_starts: np.ndarray,
    num_tokens: int,
    window_size: int,
) -> dict[str, Any]:
    """Summarize how many documents/tokens are eligible for full-window sampling."""
    lengths = document_lengths(doc_starts, num_tokens=num_tokens)
    total_documents = int(lengths.size)
    total_tokens = int(lengths.sum())
    eligible = lengths >= int(window_size)
    eligible_documents = int(eligible.sum())
    eligible_tokens = int(lengths[eligible].sum())
    skipped_documents = total_documents - eligible_documents
    skipped_tokens = total_tokens - eligible_tokens
    return {
        "total_documents": total_documents,
        "eligible_documents": eligible_documents,
        "skipped_documents": skipped_documents,
        "total_tokens": total_tokens,
        "eligible_tokens": eligible_tokens,
        "skipped_tokens": skipped_tokens,
        "single_doc_token_coverage": (
            round(eligible_tokens / total_tokens, 8) if total_tokens else None
        ),
        "single_doc_document_coverage": (
            round(eligible_documents / total_documents, 8) if total_documents else None
        ),
        "min_document_tokens": int(lengths.min()) if total_documents else None,
        "max_document_tokens": int(lengths.max()) if total_documents else None,
        "mean_document_tokens": (
            round(float(lengths.mean()), 4) if total_documents else None
        ),
    }


def summarize_dataset_document_coverage(dataset) -> dict[str, Any] | None:
    """Summarize document coverage across a TokenBlockDataset or sharded variant."""
    shards = getattr(dataset, "shards", [dataset])
    summaries = []
    for shard in shards:
        doc_starts = getattr(shard, "doc_starts", None)
        if doc_starts is None:
            continue
        summaries.append(
            summarize_document_coverage(
                doc_starts=doc_starts,
                num_tokens=int(shard.num_tokens),
                window_size=int(shard.window_size),
            )
        )
    if not summaries:
        return None

    total_documents = sum(int(item["total_documents"]) for item in summaries)
    eligible_documents = sum(int(item["eligible_documents"]) for item in summaries)
    skipped_documents = sum(int(item["skipped_documents"]) for item in summaries)
    total_tokens = sum(int(item["total_tokens"]) for item in summaries)
    eligible_tokens = sum(int(item["eligible_tokens"]) for item in summaries)
    skipped_tokens = sum(int(item["skipped_tokens"]) for item in summaries)
    min_values = [item["min_document_tokens"] for item in summaries if item["min_document_tokens"] is not None]
    max_values = [item["max_document_tokens"] for item in summaries if item["max_document_tokens"] is not None]
    return {
        "total_documents": total_documents,
        "eligible_documents": eligible_documents,
        "skipped_documents": skipped_documents,
        "total_tokens": total_tokens,
        "eligible_tokens": eligible_tokens,
        "skipped_tokens": skipped_tokens,
        "single_doc_token_coverage": (
            round(eligible_tokens / total_tokens, 8) if total_tokens else None
        ),
        "single_doc_document_coverage": (
            round(eligible_documents / total_documents, 8) if total_documents else None
        ),
        "min_document_tokens": min(min_values) if min_values else None,
        "max_document_tokens": max(max_values) if max_values else None,
    }


def sample_window_token_coverage(
    sample_starts: np.ndarray,
    num_tokens: int,
    window_size: int,
) -> dict[str, Any]:
    """Summarize unique token coverage from fixed-width sample windows."""
    starts = np.asarray(sample_starts, dtype=np.int64)
    total = int(num_tokens)
    width = int(window_size)
    if total <= 0 or width <= 0:
        return {
            "num_windows": int(starts.size),
            "sampled_tokens": 0,
            "unsampled_tokens": max(0, total),
            "sampled_token_coverage": 0.0 if total > 0 else None,
            "total_window_tokens": 0,
            "repeated_window_tokens": 0,
            "avg_window_overlap_ratio": 0.0,
        }
    if starts.size == 0:
        return {
            "num_windows": 0,
            "sampled_tokens": 0,
            "unsampled_tokens": total,
            "sampled_token_coverage": 0.0,
            "total_window_tokens": 0,
            "repeated_window_tokens": 0,
            "avg_window_overlap_ratio": 0.0,
        }

    intervals = sorted((max(0, int(start)), min(total, int(start) + width)) for start in starts)
    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if end <= start:
            continue
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        covered += max(0, current_end - current_start)
        current_start, current_end = start, end
    covered += max(0, current_end - current_start)

    total_window_tokens = int(starts.size) * width
    repeated_window_tokens = max(0, total_window_tokens - covered)
    return {
        "num_windows": int(starts.size),
        "sampled_tokens": int(covered),
        "unsampled_tokens": max(0, total - int(covered)),
        "sampled_token_coverage": round(covered / total, 8) if total else None,
        "total_window_tokens": total_window_tokens,
        "repeated_window_tokens": repeated_window_tokens,
        "avg_window_overlap_ratio": (
            round(repeated_window_tokens / total_window_tokens, 8)
            if total_window_tokens
            else 0.0
        ),
    }


def summarize_sample_coverage(dataset) -> dict[str, Any] | None:
    """Summarize actual token coverage from dataset sample windows."""
    shards = getattr(dataset, "shards", [dataset])
    summaries = []
    for shard in shards:
        sample_starts = getattr(shard, "sample_starts", None)
        if sample_starts is None:
            continue
        summaries.append(
            sample_window_token_coverage(
                sample_starts=sample_starts,
                num_tokens=int(shard.num_tokens),
                window_size=int(shard.window_size),
            )
        )
    if not summaries:
        return None

    num_windows = sum(int(item["num_windows"]) for item in summaries)
    sampled_tokens = sum(int(item["sampled_tokens"]) for item in summaries)
    unsampled_tokens = sum(int(item["unsampled_tokens"]) for item in summaries)
    total_tokens = sampled_tokens + unsampled_tokens
    total_window_tokens = sum(int(item["total_window_tokens"]) for item in summaries)
    repeated_window_tokens = sum(int(item["repeated_window_tokens"]) for item in summaries)
    return {
        "num_windows": num_windows,
        "sampled_tokens": sampled_tokens,
        "unsampled_tokens": unsampled_tokens,
        "sampled_token_coverage": (
            round(sampled_tokens / total_tokens, 8) if total_tokens else None
        ),
        "total_window_tokens": total_window_tokens,
        "repeated_window_tokens": repeated_window_tokens,
        "avg_window_overlap_ratio": (
            round(repeated_window_tokens / total_window_tokens, 8)
            if total_window_tokens
            else 0.0
        ),
    }


def estimate_dense_attention_mask_bytes(
    batch_size: int,
    seq_len: int,
    bytes_per_value: int = 1,
) -> int:
    """Estimate per-rank memory for a dense [B,1,T,T] boolean attention mask."""
    return max(0, int(batch_size)) * max(0, int(seq_len)) ** 2 * max(1, int(bytes_per_value))


def human_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.2f} TB"
