from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class PackedLabeledWindow:
    """One padded SFT window plus boundaries needed for safe attention.

    ``token_ids`` includes the extra next-token target position, so all three
    arrays have ``max_seq_len + 1`` entries.  ``document_ids`` assigns a
    different id to every packed fragment and uses ``-1`` for padding.
    """

    token_ids: list[int]
    label_mask: list[int]
    document_ids: list[int]
    source_fragments: int
    real_tokens: int


def stable_hash_fraction(text: str, seed: int = 42) -> float:
    """Map text to a stable [0, 1) fraction for order-independent splits."""
    payload = f"{int(seed)}\0{text}".encode("utf-8", errors="surrogatepass")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value / float(1 << 64)


def route_to_validation(
    text: str,
    val_ratio: float,
    seed: int,
    split_mode: str = "hash",
    rng: random.Random | None = None,
) -> bool:
    """Return whether a document should be written to validation."""
    if val_ratio <= 0:
        return False
    if val_ratio >= 1:
        return True

    mode = split_mode.lower()
    if mode == "hash":
        return stable_hash_fraction(text, seed=seed) < val_ratio
    if mode == "random":
        if rng is None:
            raise ValueError("rng is required when split_mode='random'")
        return rng.random() < val_ratio
    raise ValueError("split_mode must be one of: hash, random")


def grouped_hash_validation_mask(
    group_keys: Iterable[str],
    val_ratio: float,
    seed: int = 42,
) -> list[bool]:
    """Assign complete groups to train/validation with deterministic fallback.

    A per-row fallback can put one copy of duplicated text in validation while
    leaving another copy in training.  This helper always makes the decision
    once per key.  When hashing selects no validation group (or no training
    group), an entire lowest-hash (or highest-hash) group is moved instead.
    With fewer than two distinct groups a leak-free validation split is not
    possible, so all rows stay in training.
    """

    keys = [str(key) for key in group_keys]
    if not keys or float(val_ratio) <= 0:
        return [False] * len(keys)

    unique_keys = list(dict.fromkeys(keys))
    if len(unique_keys) < 2:
        return [False] * len(keys)

    fractions = {
        key: stable_hash_fraction(key, seed=seed)
        for key in unique_keys
    }
    assignments = {
        key: fraction < float(val_ratio)
        for key, fraction in fractions.items()
    }
    if not any(assignments.values()):
        assignments[min(unique_keys, key=lambda key: (fractions[key], key))] = True
    if all(assignments.values()):
        assignments[max(unique_keys, key=lambda key: (fractions[key], key))] = False
    return [assignments[key] for key in keys]


def validate_lm_stride(max_seq_len: int, stride: int | None) -> int:
    """Return a stride that cannot leave next-token targets uncovered."""

    max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    resolved = max_seq_len if stride is None else int(stride)
    if resolved <= 0:
        raise ValueError("stride must be positive")
    if resolved > max_seq_len:
        raise ValueError(
            f"stride ({resolved}) cannot exceed max_seq_len ({max_seq_len}); "
            "a larger stride silently leaves next-token targets uncovered"
        )
    return resolved


def _iter_labeled_fragments(
    token_ids: list[int],
    label_mask: list[int],
    window_size: int,
    stride: int,
) -> Iterable[tuple[list[int], list[int]]]:
    if len(token_ids) != len(label_mask):
        raise ValueError("token_ids and label_mask must have identical lengths")
    if len(token_ids) < 2:
        return
    for start in range(0, len(token_ids) - 1, stride):
        fragment_ids = list(token_ids[start : start + window_size])
        fragment_mask = list(label_mask[start : start + window_size])
        if len(fragment_ids) < 2:
            break
        # A packed boundary must never train the preceding document to predict
        # the first token of this fragment, even when add_bos=False.
        fragment_mask[0] = 0
        yield fragment_ids, fragment_mask
        if start + window_size >= len(token_ids):
            break


def pack_labeled_token_sequences(
    sequences: Iterable[tuple[list[int], list[int]]],
    *,
    max_seq_len: int,
    pad_id: int,
    stride: int | None = None,
) -> tuple[list[PackedLabeledWindow], dict[str, int | float]]:
    """Pack assistant-labeled token sequences without cross-example leakage.

    Long sequences are windowed with an overlap that preserves every
    next-token target.  Short fragments share a window, while document ids and
    a zeroed boundary label keep examples independent.  The caller should pass
    the returned ``document_ids[:-1]`` to the model together with its padding
    mask.
    """

    max_seq_len = int(max_seq_len)
    window_size = max_seq_len + 1
    resolved_stride = validate_lm_stride(max_seq_len, stride)
    materialized = [(list(ids), list(mask)) for ids, mask in sequences]

    windows: list[PackedLabeledWindow] = []
    buffer_ids: list[int] = []
    buffer_mask: list[int] = []
    buffer_docs: list[int] = []
    buffer_fragments = 0
    fragment_count = 0
    skipped_empty_windows = 0
    non_padding_tokens = 0
    supervised_tokens = 0
    packed_windows = 0

    def flush() -> None:
        nonlocal buffer_ids, buffer_mask, buffer_docs, buffer_fragments
        nonlocal skipped_empty_windows, non_padding_tokens, supervised_tokens
        nonlocal packed_windows
        if not buffer_ids:
            return
        real_tokens = len(buffer_ids)
        supervised = sum(buffer_mask[1:])
        if supervised <= 0:
            skipped_empty_windows += 1
        else:
            pad_len = window_size - real_tokens
            windows.append(
                PackedLabeledWindow(
                    token_ids=buffer_ids + [int(pad_id)] * pad_len,
                    label_mask=buffer_mask + [0] * pad_len,
                    document_ids=buffer_docs + [-1] * pad_len,
                    source_fragments=buffer_fragments,
                    real_tokens=real_tokens,
                )
            )
            non_padding_tokens += real_tokens
            supervised_tokens += supervised
            if buffer_fragments > 1:
                packed_windows += 1
        buffer_ids = []
        buffer_mask = []
        buffer_docs = []
        buffer_fragments = 0

    for ids, mask in materialized:
        for fragment_ids, fragment_mask in _iter_labeled_fragments(
            ids,
            mask,
            window_size=window_size,
            stride=resolved_stride,
        ):
            fragment_count += 1
            if buffer_ids and len(buffer_ids) + len(fragment_ids) > window_size:
                flush()
            document_id = buffer_fragments
            buffer_ids.extend(fragment_ids)
            buffer_mask.extend(fragment_mask)
            buffer_docs.extend([document_id] * len(fragment_ids))
            buffer_fragments += 1
            if len(buffer_ids) == window_size:
                flush()
    flush()

    capacity = len(windows) * window_size
    stats: dict[str, int | float] = {
        "input_sequences": len(materialized),
        "fragments": fragment_count,
        "windows": len(windows),
        "packed_windows": packed_windows,
        "skipped_empty_windows": skipped_empty_windows,
        "non_padding_tokens": non_padding_tokens,
        "padding_tokens": capacity - non_padding_tokens,
        "supervised_tokens": supervised_tokens,
        "packing_efficiency": (
            round(non_padding_tokens / capacity, 8) if capacity else 0.0
        ),
    }
    return windows, stats


def build_sample_starts(
    num_tokens: int,
    window_size: int,
    stride: int,
    tail_min_gap_ratio: float = 0.5,
) -> np.ndarray:
    """Build fixed-window starts, optionally adding a sufficiently separated tail window."""
    num_tokens = int(num_tokens)
    window_size = int(window_size)
    stride = int(stride)
    tail_min_gap_ratio = float(tail_min_gap_ratio)
    if num_tokens < window_size:
        raise ValueError("num_tokens must be >= window_size")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride > window_size:
        raise ValueError(
            f"stride ({stride}) cannot exceed window_size ({window_size}); "
            "a larger stride leaves uncovered token ranges"
        )
    if tail_min_gap_ratio < 0:
        raise ValueError("tail_min_gap_ratio must be non-negative")

    last_start = num_tokens - window_size
    starts = list(range(0, last_start + 1, stride))
    tail_gap = last_start - starts[-1]
    min_tail_gap = math.ceil(stride * tail_min_gap_ratio)
    if starts[-1] != last_start and tail_gap >= min_tail_gap:
        starts.append(last_start)
    return np.asarray(starts, dtype=np.uint64)


def build_document_sample_starts(
    doc_starts: np.ndarray,
    num_tokens: int,
    window_size: int,
    stride: int,
    tail_min_gap_ratio: float = 0.5,
) -> np.ndarray:
    """Build fixed-window starts that never cross document boundaries."""
    starts = np.asarray(doc_starts, dtype=np.uint64)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("doc_starts must be a non-empty 1D array")
    num_tokens = int(num_tokens)
    if int(starts[0]) != 0:
        raise ValueError("first document start must be 0")
    if np.any(starts[1:] <= starts[:-1]):
        raise ValueError("doc_starts must be strictly increasing")

    all_starts: list[np.ndarray] = []
    doc_ends = np.concatenate((starts[1:], np.asarray([num_tokens], dtype=np.uint64)))
    for doc_start, doc_end in zip(starts, doc_ends):
        doc_len = int(doc_end) - int(doc_start)
        if doc_len < int(window_size):
            continue
        local_starts = build_sample_starts(
            num_tokens=doc_len,
            window_size=window_size,
            stride=stride,
            tail_min_gap_ratio=tail_min_gap_ratio,
        )
        all_starts.append(local_starts + np.uint64(doc_start))

    if not all_starts:
        return np.asarray([], dtype=np.uint64)
    return np.concatenate(all_starts).astype(np.uint64, copy=False)


def document_ids_for_positions(
    doc_starts: np.ndarray,
    positions: Iterable[int] | np.ndarray,
) -> np.ndarray:
    """Return document ids for absolute token positions using sorted starts."""
    starts = np.asarray(doc_starts, dtype=np.uint64)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("doc_starts must be a non-empty 1D array")
    pos = np.asarray(list(positions) if not isinstance(positions, np.ndarray) else positions, dtype=np.uint64)
    doc_ids = np.searchsorted(starts, pos, side="right") - 1
    if np.any(doc_ids < 0):
        raise ValueError("positions include tokens before the first document start")
    return doc_ids.astype(np.int64, copy=False)


def same_document_attention_mask(doc_ids: np.ndarray) -> torch.Tensor:
    """Create a [T, T] boolean mask where tokens can only see their document."""
    ids = torch.as_tensor(doc_ids, dtype=torch.long)
    return ids[:, None].eq(ids[None, :])


def cross_document_targets(doc_ids: np.ndarray) -> torch.Tensor:
    """Return a [T-1] mask for next-token labels that cross document boundaries."""
    ids = torch.as_tensor(doc_ids, dtype=torch.long)
    return ids[:-1].ne(ids[1:])
