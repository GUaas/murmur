from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .augmentation_pipeline import _content_key, _sha256


ANCHOR_RATIOS: Mapping[str, float] = {
    "safety": 0.35,
    "identity": 0.20,
    "math": 0.20,
    "instruction": 0.15,
    "knowledge": 0.10,
}


@dataclass(frozen=True, slots=True)
class V5ContinuationOptions:
    base_root: Path
    expanded_root: Path
    output_root: Path
    anchor_fraction: float = 0.15
    seed: int = 20260809


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError(f"Invalid SFT row at {path}:{line_number}")
            yield row


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _stable_score(seed: int, label: str, row: dict[str, Any]) -> str:
    record_id = str(row.get("id") or _content_key(row["messages"]))
    payload = f"{seed}:{label}:{record_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _anchor_group(row: dict[str, Any]) -> str | None:
    source = str(row.get("source", "")).lower()
    category = str(row.get("category", "")).lower()

    if source in {"synthetic_safety_verified", "coig_human_value"} or category in {
        "safe",
        "credentials",
        "privacy",
        "physical_safety",
        "phishing",
        "fraud",
        "human_value_alignment",
    }:
        return "safety"
    if source == "synthetic_identity_verified" or category.startswith("identity_") or category in {
        "capability_boundary",
        "experience_boundary",
    }:
        return "identity"
    if source in {"synthetic_math_verified", "gsm8k_zh"} or category in {
        "arithmetic_reasoning",
        "add_subtract",
        "multiply_divide",
        "mixed_arithmetic",
        "percentage",
        "time_conversion",
        "逻辑推理",
        "逻辑问答",
        "推理",
    }:
        return "math"
    if source in {"synthetic_instruction_verified", "tulu_instruction_following"} or category in {
        "strict_instruction_following_english",
        "fixed_list",
        "key_value_format",
        "markdown_table",
        "json_extraction",
        "特殊格式",
    }:
        return "instruction"
    if source in {"coig_cqia", "cmrc2018", "drcd"} or category in {
        "information-seeking",
        "概念解析",
        "知识问答",
        "百科问答",
        "reading_comprehension",
        "reading_comprehension_traditional_source",
    }:
        return "knowledge"
    return None


def _allocate_quotas(target: int) -> dict[str, int]:
    minimum = 1 if target >= len(ANCHOR_RATIOS) else 0
    quotas = {name: minimum for name in ANCHOR_RATIOS}
    remaining_target = target - sum(quotas.values())
    if remaining_target <= 0:
        return quotas
    for name, ratio in ANCHOR_RATIOS.items():
        quotas[name] += int(remaining_target * ratio)
    remainder = target - sum(quotas.values())
    for name in ANCHOR_RATIOS:
        if remainder <= 0:
            break
        quotas[name] += 1
        remainder -= 1
    return quotas


def _select_anchor_records(
    base_train: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_train:
        group = _anchor_group(row)
        if group is not None:
            buckets[group].append(row)

    for group, rows in buckets.items():
        rows.sort(key=lambda row: _stable_score(seed, group, row))

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    group_counts: Counter[str] = Counter()
    quotas = _allocate_quotas(target)
    for group, quota in quotas.items():
        for row in buckets.get(group, [])[:quota]:
            key = _content_key(row["messages"])
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            group_counts[group] += 1

    if len(selected) < target:
        candidates = [
            (group, row)
            for group, rows in buckets.items()
            for row in rows
            if _content_key(row["messages"]) not in selected_keys
        ]
        candidates.sort(key=lambda item: _stable_score(seed, f"fill:{item[0]}", item[1]))
        for group, row in candidates[: target - len(selected)]:
            selected.append(row)
            selected_keys.add(_content_key(row["messages"]))
            group_counts[group] += 1

    if len(selected) < target:
        raise ValueError(f"Only {len(selected)} eligible anchor rows are available; {target} requested")
    return selected, group_counts


def _record_keys(paths: Iterable[Path]) -> set[str]:
    return {
        _content_key(row["messages"])
        for path in paths
        for row in _load_jsonl(path)
    }


def _dataset_stats(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    return {
        "records": len(materialized),
        "sources": dict(sorted(Counter(str(row.get("source", "unknown")) for row in materialized).items())),
        "categories": dict(
            sorted(Counter(str(row.get("category", "unknown")) for row in materialized).items())
        ),
        "declared_tokens": sum(int(row.get("num_tokens", 0) or 0) for row in materialized),
    }


def build_v5_continuation_dataset(options: V5ContinuationOptions) -> dict[str, Any]:
    if not 0.0 < options.anchor_fraction < 0.5:
        raise ValueError("anchor_fraction must be between 0 and 0.5")

    base_root = options.base_root.resolve()
    expanded_root = options.expanded_root.resolve()
    output_root = options.output_root.resolve()
    base_train_path = base_root / "processed" / "train.jsonl"
    base_validation_path = base_root / "processed" / "validation.jsonl"
    expanded_train_path = expanded_root / "processed" / "train.jsonl"
    expanded_validation_path = expanded_root / "processed" / "validation.jsonl"
    required = (
        base_train_path,
        base_validation_path,
        expanded_train_path,
        expanded_validation_path,
        expanded_root / "assets" / "sp_unigram_32k.model",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing v5 inputs: {missing}")

    base_keys = _record_keys((base_train_path, base_validation_path))
    base_train = list(_load_jsonl(base_train_path))
    new_train = [
        row
        for row in _load_jsonl(expanded_train_path)
        if _content_key(row["messages"]) not in base_keys
    ]
    if not new_train:
        raise ValueError("No newly added v4 training rows were found relative to the v3 base")

    anchor_target = round(len(new_train) * options.anchor_fraction / (1.0 - options.anchor_fraction))
    anchors, anchor_groups = _select_anchor_records(
        base_train,
        target=anchor_target,
        seed=options.seed,
    )
    mixed_train = [("new", row) for row in new_train] + [("anchor", row) for row in anchors]
    mixed_train.sort(key=lambda item: _stable_score(options.seed, item[0], item[1]))
    train_rows = [row for _, row in mixed_train]
    validation_rows = list(_load_jsonl(expanded_validation_path))

    train_keys = {_content_key(row["messages"]) for row in train_rows}
    validation_keys = {_content_key(row["messages"]) for row in validation_rows}
    overlap = train_keys & validation_keys
    if overlap:
        raise ValueError(f"Detected {len(overlap)} train/validation content overlaps")

    processed_dir = output_root / "processed"
    reports_dir = output_root / "reports"
    assets_dir = output_root / "assets"
    reports_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_target = assets_dir / "sp_unigram_32k.model"
    shutil.copy2(expanded_root / "assets" / "sp_unigram_32k.model", tokenizer_target)
    train_count = _write_jsonl_atomic(processed_dir / "train.jsonl", train_rows)
    validation_count = _write_jsonl_atomic(processed_dir / "validation.jsonl", validation_rows)

    actual_anchor_fraction = len(anchors) / train_count
    report: dict[str, Any] = {
        "built_for": "Murmur 203M v5 continued SFT from the previous SFT checkpoint",
        "seed": options.seed,
        "base_root": str(base_root),
        "expanded_root": str(expanded_root),
        "output_root": str(output_root),
        "policy": {
            "new_data_only_definition": "Rows present in v4 but absent from all v3 train/validation rows",
            "requested_anchor_fraction": options.anchor_fraction,
            "actual_anchor_fraction": actual_anchor_fraction,
            "anchor_ratios": dict(ANCHOR_RATIOS),
            "anchor_groups": dict(sorted(anchor_groups.items())),
        },
        "counts": {
            "new_train_records": len(new_train),
            "anchor_records": len(anchors),
            "train_records": train_count,
            "validation_records": validation_count,
            "train_validation_content_overlap": 0,
        },
        "new_data": _dataset_stats(new_train),
        "anchors": _dataset_stats(anchors),
        "files": {
            "train": {
                "path": str(processed_dir / "train.jsonl"),
                "bytes": (processed_dir / "train.jsonl").stat().st_size,
                "sha256": _sha256(processed_dir / "train.jsonl"),
            },
            "validation": {
                "path": str(processed_dir / "validation.jsonl"),
                "bytes": (processed_dir / "validation.jsonl").stat().st_size,
                "sha256": _sha256(processed_dir / "validation.jsonl"),
            },
            "tokenizer": {
                "path": str(tokenizer_target),
                "bytes": tokenizer_target.stat().st_size,
                "sha256": _sha256(tokenizer_target),
            },
        },
    }
    report_path = reports_dir / "mixture_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
