from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from muddywater.source_target import SourceTargetTemplate


@dataclass(frozen=True)
class PairRecord:
    source: str
    target: str


@dataclass(frozen=True)
class PreparationOptions:
    input_source_key: str = "data"
    input_target_key: str = "s"
    validation_ratio: float = 0.05
    seed: int = 20260812
    source_label: str = "<|im_start|>"
    target_label: str = "<|im_end|>"

    def __post_init__(self) -> None:
        if not self.input_source_key or not self.input_target_key:
            raise ValueError("input field names must not be empty")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1")
        SourceTargetTemplate(
            source_label=self.source_label,
            target_label=self.target_label,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_records(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        list_fields = [value for value in payload.values() if isinstance(value, list)]
        if len(list_fields) == 1:
            return list_fields[0]
    raise ValueError("JSON input must be an array or contain exactly one array field")


def _load_jsonl_records(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def load_pairs(path: Path, options: PreparationOptions) -> list[PairRecord]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = _load_jsonl_records(path) if path.suffix.lower() == ".jsonl" else _load_json_records(path)
    template = SourceTargetTemplate(
        source_label=options.source_label,
        target_label=options.target_label,
    )
    pairs: list[PairRecord] = []
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Row {row_index} must be a JSON object")
        if options.input_source_key not in record or options.input_target_key not in record:
            raise ValueError(
                f"Row {row_index} is missing {options.input_source_key!r} or "
                f"{options.input_target_key!r}"
            )
        try:
            source, target = template.normalize_pair(
                record[options.input_source_key],
                record[options.input_target_key],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pair at row {row_index}: {exc}") from exc
        pairs.append(PairRecord(source=source, target=target))
    if not pairs:
        raise ValueError("No valid source/target pairs were found")
    return pairs


def deduplicate_exact_pairs(pairs: Iterable[PairRecord]) -> tuple[list[PairRecord], int]:
    unique: list[PairRecord] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        key = (pair.source, pair.target)
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique, len(seen)


def _source_is_validation(source: str, *, ratio: float, seed: int) -> bool:
    payload = f"{seed}\0{source}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return bucket < int(ratio * (1 << 64))


def split_by_source(
    pairs: Sequence[PairRecord], *, validation_ratio: float, seed: int
) -> tuple[list[PairRecord], list[PairRecord]]:
    validation_sources = {
        pair.source
        for pair in pairs
        if _source_is_validation(pair.source, ratio=validation_ratio, seed=seed)
    }
    if not validation_sources:
        validation_sources.add(pairs[0].source)
    all_sources = {pair.source for pair in pairs}
    if validation_sources == all_sources:
        validation_sources.remove(pairs[0].source)

    train = [pair for pair in pairs if pair.source not in validation_sources]
    validation = [pair for pair in pairs if pair.source in validation_sources]
    if not train or not validation:
        raise ValueError("Deterministic split produced an empty train or validation set")
    return train, validation


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(ratio: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * ratio))
        return ordered[index]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 6),
    }


def build_statistics(pairs: Sequence[PairRecord]) -> dict[str, Any]:
    targets_by_source: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        targets_by_source[pair.source].add(pair.target)
    return {
        "rows": len(pairs),
        "unique_sources": len(targets_by_source),
        "sources_with_multiple_targets": sum(
            len(targets) > 1 for targets in targets_by_source.values()
        ),
        "unchanged_pairs": sum(pair.source == pair.target for pair in pairs),
        "target_longer_than_source": sum(
            len(pair.target) > len(pair.source) for pair in pairs
        ),
        "source_chars": _quantiles([float(len(pair.source)) for pair in pairs]),
        "target_chars": _quantiles([float(len(pair.target)) for pair in pairs]),
        "target_source_char_ratio": _quantiles(
            [len(pair.target) / len(pair.source) for pair in pairs]
        ),
    }


def _write_jsonl_atomic(path: Path, pairs: Sequence[PairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temp_path = Path(handle.name)
            for pair in pairs:
                handle.write(
                    json.dumps(
                        {"source": pair.source, "target": pair.target},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        assert temp_path is not None
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        assert temp_path is not None
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    options: PreparationOptions,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    loaded_pairs = load_pairs(input_path, options)
    pairs, unique_count = deduplicate_exact_pairs(loaded_pairs)
    train, validation = split_by_source(
        pairs,
        validation_ratio=options.validation_ratio,
        seed=options.seed,
    )

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    report_path = output_dir.parent / "reports" / "preparation_report.json"
    _write_jsonl_atomic(train_path, train)
    _write_jsonl_atomic(validation_path, validation)

    report: dict[str, Any] = {
        "format_version": 1,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_rows": len(loaded_pairs),
        "exact_duplicates_removed": len(loaded_pairs) - unique_count,
        "fields": {
            "input_source": options.input_source_key,
            "input_target": options.input_target_key,
            "output_source": "source",
            "output_target": "target",
        },
        "template": {
            "source_label": options.source_label,
            "target_label": options.target_label,
            "serialized": (
                f"{options.source_label}{{source}}{options.target_label}{{target}}<eos>"
            ),
            "loss": "target_and_eos_only",
        },
        "split": {
            "method": "sha256_grouped_by_source",
            "seed": options.seed,
            "requested_validation_ratio": options.validation_ratio,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "actual_validation_ratio": len(validation) / len(pairs),
            "source_overlap": len(
                {pair.source for pair in train} & {pair.source for pair in validation}
            ),
        },
        "statistics": build_statistics(pairs),
        "outputs": {
            "train": str(train_path),
            "train_sha256": sha256_file(train_path),
            "validation": str(validation_path),
            "validation_sha256": sha256_file(validation_path),
        },
    }
    _write_json_atomic(report_path, report)
    report["report"] = str(report_path)
    return report
