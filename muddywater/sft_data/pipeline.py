from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .catalog import COIG_CATEGORY_CAPS, SMOLTALK_QUOTAS, SOURCE_QUOTAS
from .download import download_sources
from .filtering import RecordFilter
from .readers import (
    read_coig,
    read_crosswoz,
    read_duconv,
    read_kdconv,
    read_oasst2,
    read_smoltalk,
    read_ultradata,
)
from .records import SFTRecord, stable_hex
from .sampling import select_records
from .synthetic import read_verified_synthetic


@dataclass(slots=True)
class BuildOptions:
    root: Path
    tokenizer_path: Path
    max_tokens: int = 2048
    validation_ratio: float = 0.005
    seed: int = 20260802
    include_ultradata: bool = True
    download: bool = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Iterable[SFTRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def _remove_stale_by_source_files(directory: Path, active_sources: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    active_names = {f"{source}.jsonl" for source in active_sources}
    removed: list[Path] = []
    for path in directory.glob("*.jsonl"):
        if path.name not in active_names:
            path.unlink()
            removed.append(path)
    return removed


def _split(records: list[SFTRecord], ratio: float) -> tuple[list[SFTRecord], list[SFTRecord]]:
    threshold = int(ratio * 1_000_000)
    train: list[SFTRecord] = []
    validation: list[SFTRecord] = []
    for record in records:
        bucket = int(stable_hex(record.group_id, 16), 16) % 1_000_000
        (validation if bucket < threshold else train).append(record)
    return train, validation


def _record_summary(records: list[SFTRecord]) -> dict[str, object]:
    token_lengths = sorted(record.num_tokens for record in records)
    assistant_tokens = sum(record.assistant_tokens for record in records)
    by_source = Counter(record.source for record in records)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_category[record.source][record.category] += 1
    percentile = lambda fraction: token_lengths[min(len(token_lengths) - 1, int(len(token_lengths) * fraction))]
    return {
        "records": len(records),
        "context_tokens": sum(token_lengths),
        "assistant_tokens": assistant_tokens,
        "token_length": {
            "min": min(token_lengths),
            "mean": round(statistics.fmean(token_lengths), 2),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": max(token_lengths),
        },
        "by_source": dict(sorted(by_source.items())),
        "by_category": {key: dict(sorted(value.items())) for key, value in sorted(by_category.items())},
    }


def _curate_coig(records: list[SFTRecord], quota: int, seed: int) -> list[SFTRecord]:
    category_counts: Counter[str] = Counter()
    # Human-reviewed examples win ties for limited model capacity. The
    # stable hash keeps the result deterministic within each quality tier.
    ordered = sorted(
        records,
        key=lambda record: (
            record.metadata.get("human_verified") is not True,
            stable_hex(f"coig:{seed}:{record.record_id}", 32),
        ),
    )
    selected: list[SFTRecord] = []
    for record in ordered:
        cap = COIG_CATEGORY_CAPS.get(record.category, quota)
        if category_counts[record.category] >= cap:
            continue
        selected.append(record)
        category_counts[record.category] += 1
        if len(selected) >= quota:
            break
    return selected


def build_dataset(options: BuildOptions) -> dict[str, object]:
    root = options.root.resolve()
    for name in ("raw", "processed", "reports", "cache"):
        (root / name).mkdir(parents=True, exist_ok=True)
    report_file = root / "reports" / "download_report.json"
    if options.download:
        download_report = download_sources(root, include_ultradata=options.include_ultradata)
    elif report_file.exists():
        download_report = json.loads(report_file.read_text(encoding="utf-8"))
    else:
        download_report = {"downloaded": {}, "skipped": {}}
    raw_dir = root / "raw"
    tokenizer_path = options.tokenizer_path.resolve()
    portable_tokenizer = root / "assets" / tokenizer_path.name
    portable_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    if not portable_tokenizer.exists() or _sha256(portable_tokenizer) != _sha256(tokenizer_path):
        shutil.copy2(tokenizer_path, portable_tokenizer)
    # SentencePiece on Windows can fail on otherwise valid paths containing
    # non-ASCII characters. The dataset-local copy is also useful provenance.
    record_filter = RecordFilter(str(portable_tokenizer), max_tokens=options.max_tokens)
    selected: list[SFTRecord] = []

    for category, quota in SMOLTALK_QUOTAS.items():
        selected.extend(
            select_records(
                read_smoltalk(raw_dir, category),
                quota=quota,
                record_filter=record_filter,
                seed=options.seed,
            )
        )

    coig_pool = select_records(
        read_coig(raw_dir),
        quota=50_000,
        record_filter=record_filter,
        seed=options.seed,
    )
    selected.extend(_curate_coig(coig_pool, SOURCE_QUOTAS["coig_cqia"], options.seed))

    readers = (
        ("oasst2_zh", read_oasst2(raw_dir)),
        ("kdconv", read_kdconv(raw_dir)),
        ("crosswoz", read_crosswoz(raw_dir)),
        ("duconv", read_duconv(raw_dir)),
    )
    for name, records in readers:
        selected.extend(
            select_records(
                records,
                quota=SOURCE_QUOTAS[name],
                record_filter=record_filter,
                seed=options.seed,
            )
        )

    selected.extend(
        select_records(
            read_verified_synthetic(),
            quota=SOURCE_QUOTAS["synthetic_verified"],
            record_filter=record_filter,
            seed=options.seed,
        )
    )

    ultradata_dir = raw_dir / "ultradata"
    if options.include_ultradata and ultradata_dir.exists() and not download_report.get("skipped", {}).get("ultradata"):
        for domain in ("chinese_general", "if", "knowledge"):
            name = f"ultradata_{domain}"
            selected.extend(
                select_records(
                    read_ultradata(raw_dir, domain),
                    quota=SOURCE_QUOTAS[name],
                    record_filter=record_filter,
                    seed=options.seed,
                )
            )

    deduplicated: list[SFTRecord] = []
    global_seen: set[str] = set()
    for record in selected:
        if record.content_key not in global_seen:
            global_seen.add(record.content_key)
            deduplicated.append(record)
    deduplicated.sort(key=lambda record: stable_hex(f"{options.seed}:{record.record_id}", 32))

    train, validation = _split(deduplicated, options.validation_ratio)
    processed_dir = root / "processed"
    _write_jsonl(processed_dir / "train.jsonl", train)
    _write_jsonl(processed_dir / "validation.jsonl", validation)
    active_sources = {record.source for record in deduplicated}
    _remove_stale_by_source_files(processed_dir / "by_source", active_sources)
    for source in sorted(active_sources):
        _write_jsonl(processed_dir / "by_source" / f"{source}.jsonl", (r for r in deduplicated if r.source == source))

    summary = _record_summary(deduplicated)
    summary.update(
        {
            "train_records": len(train),
            "validation_records": len(validation),
            "max_tokens": options.max_tokens,
            "seed": options.seed,
            "tokenizer": str(portable_tokenizer),
            "tokenizer_sha256": _sha256(portable_tokenizer),
            "train_sha256": _sha256(processed_dir / "train.jsonl"),
            "validation_sha256": _sha256(processed_dir / "validation.jsonl"),
            "download_skips": download_report.get("skipped", {}),
            "filter": {
                "accepted_before_sampling": record_filter.stats.accepted,
                "rejected": dict(record_filter.stats.rejected),
            },
        }
    )
    report_path = root / "reports" / "dataset_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
