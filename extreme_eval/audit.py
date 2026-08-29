from __future__ import annotations

import hashlib
import json
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            records.append(payload)
    return records


def normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith(("P", "S"))
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ngrams(text: str, width: int = 4) -> set[str]:
    if len(text) <= width:
        return {text} if text else set()
    return {text[index : index + width] for index in range(len(text) - width + 1)}


def _anchors(grams: Iterable[str], count: int = 12) -> tuple[int, ...]:
    values = sorted({zlib.crc32(gram.encode("utf-8")) for gram in grams})
    return tuple(values[:count])


def near_duplicate_audit(
    train_sources: list[str],
    validation_sources: list[str],
    *,
    threshold: float = 0.85,
) -> dict[str, Any]:
    train_norm = [normalized_text(text) for text in train_sources]
    validation_norm = [normalized_text(text) for text in validation_sources]
    train_grams = [_ngrams(text) for text in train_norm]
    index: dict[int, list[int]] = defaultdict(list)
    for row_index, grams in enumerate(train_grams):
        for anchor in _anchors(grams):
            index[anchor].append(row_index)

    matches: list[dict[str, Any]] = []
    max_scores: list[float] = []
    for val_index, text in enumerate(validation_norm):
        grams = _ngrams(text)
        candidates: Counter[int] = Counter()
        for anchor in _anchors(grams):
            candidates.update(index.get(anchor, ()))
        best_index = None
        best_score = 0.0
        for train_index, _ in candidates.most_common(80):
            other = train_grams[train_index]
            union = len(grams | other)
            score = len(grams & other) / union if union else float(grams == other)
            if score > best_score:
                best_score = score
                best_index = train_index
        max_scores.append(best_score)
        if best_index is not None and best_score >= threshold:
            matches.append(
                {
                    "validation_index": val_index,
                    "train_index": best_index,
                    "jaccard_char4": round(best_score, 6),
                    "validation_source": validation_sources[val_index],
                    "train_source": train_sources[best_index],
                }
            )
    matches.sort(key=lambda item: item["jaccard_char4"], reverse=True)
    return {
        "method": "normalized character 4-gram Jaccard with deterministic anchor candidate search",
        "threshold": threshold,
        "validation_rows": len(validation_sources),
        "near_match_count": len(matches),
        "near_match_rate": round(len(matches) / len(validation_sources), 6) if validation_sources else 0.0,
        "max_similarity": round(max(max_scores), 6) if max_scores else None,
        "matches": matches[:100],
    }


def dataset_audit(train_path: Path, validation_path: Path) -> dict[str, Any]:
    train = load_jsonl(train_path)
    validation = load_jsonl(validation_path)
    train_sources = [str(row.get("source", "")) for row in train]
    train_targets = [str(row.get("target", "")) for row in train]
    val_sources = [str(row.get("source", "")) for row in validation]
    val_targets = [str(row.get("target", "")) for row in validation]

    train_source_set = set(train_sources)
    val_source_set = set(val_sources)
    train_target_set = set(train_targets)
    train_norm_set = {normalized_text(text) for text in train_sources}
    val_norm_set = {normalized_text(text) for text in val_sources}
    reserved = ("<|im_start|>", "<|im_end|>")

    return {
        "train_path": str(train_path.resolve()),
        "validation_path": str(validation_path.resolve()),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "empty_source_or_target": sum(
            not str(row.get("source", "")).strip() or not str(row.get("target", "")).strip()
            for row in train + validation
        ),
        "reserved_tag_collisions": sum(
            any(tag in str(row.get(key, "")) for tag in reserved for key in ("source", "target"))
            for row in train + validation
        ),
        "exact_train_source_duplicates": len(train_sources) - len(train_source_set),
        "exact_validation_source_duplicates": len(val_sources) - len(val_source_set),
        "exact_source_overlap": len(train_source_set & val_source_set),
        "normalized_source_overlap": len(train_norm_set & val_norm_set),
        "validation_target_seen_as_train_target": sum(target in train_target_set for target in val_targets),
        "validation_source_seen_as_train_target": sum(source in train_target_set for source in val_sources),
        "validation_target_seen_as_train_source": sum(target in train_source_set for target in val_targets),
        "train_identity_pairs": sum(source.strip() == target.strip() for source, target in zip(train_sources, train_targets)),
        "validation_identity_pairs": sum(source.strip() == target.strip() for source, target in zip(val_sources, val_targets)),
        "validation_target_longer": sum(len(target) > len(source) for source, target in zip(val_sources, val_targets)),
        "near_duplicate": near_duplicate_audit(train_sources, val_sources),
    }
