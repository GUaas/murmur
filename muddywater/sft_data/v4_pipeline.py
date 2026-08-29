from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterator

from huggingface_hub import hf_hub_download

from .augmentation_pipeline import (
    _content_key,
    _load_jsonl,
    _sha256,
    _split,
    _token_statistics,
    _write_merged_jsonl,
    _write_sample_audit,
)
from .coverage import build_scenario_coverage
from .filtering import RecordFilter
from .records import SFTRecord, stable_hex
from .sampling import select_records
from .v4_catalog import INFINITY_REPO_ID, INFINITY_REVISION, INFINITY_SHARDS, InfinityShard
from .v4_readers import read_infinity_records


@dataclass(slots=True)
class V4Options:
    base_root: Path
    output_root: Path
    tokenizer_path: Path
    max_tokens: int = 1_024
    validation_ratio: float = 0.005
    seed: int = 20260809
    download: bool = True


def _raw_path(raw_dir: Path, shard: InfinityShard) -> Path:
    return raw_dir / shard.key / shard.subset / shard.filename


def _download_sources(raw_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for shard in INFINITY_SHARDS:
        local_dir = raw_dir / shard.key
        local_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=INFINITY_REPO_ID,
            repo_type="dataset",
            revision=INFINITY_REVISION,
            filename=f"{shard.subset}/{shard.filename}",
            local_dir=local_dir,
        )
        paths[shard.key] = str(Path(path).resolve())
    return paths


def _local_sources(raw_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for shard in INFINITY_SHARDS:
        path = _raw_path(raw_dir, shard)
        if not path.is_file():
            raise FileNotFoundError(f"Required Infinity-Instruct shard is missing: {path}")
        paths[shard.key] = str(path.resolve())
    return paths


def _select_additions(
    raw_dir: Path,
    record_filter: RecordFilter,
    seed: int,
) -> tuple[list[SFTRecord], dict[str, int]]:
    selected: list[SFTRecord] = []
    selected_counts: dict[str, int] = {}
    for shard in INFINITY_SHARDS:
        for language in shard.languages:
            key = f"{shard.key}_{language}"
            records = read_infinity_records(raw_dir, shard=shard, language=language)
            source_selection = select_records(
                records,
                quota=shard.quota(language),
                record_filter=record_filter,
                seed=seed,
            )
            selected.extend(source_selection)
            selected_counts[key] = len(source_selection)
    return selected, selected_counts


def _count_rows(path: Path) -> int:
    return sum(1 for _ in _load_jsonl(path))


def _base_content_keys(paths: tuple[Path, ...]) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        for row in _load_jsonl(path):
            seen.add(_content_key(row["messages"]))
    return seen


def _copy_support_artifacts(base_root: Path, output_root: Path) -> None:
    copies = (
        (base_root / "curated" / "murmur_identity.jsonl", output_root / "curated" / "murmur_identity.jsonl"),
        (base_root / "reports" / "identity_validation.json", output_root / "reports" / "identity_validation.json"),
        (base_root / "reports" / "identity_eval_prompts.json", output_root / "reports" / "identity_eval_prompts.json"),
    )
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_sources_manifest(base_root: Path, report_dir: Path) -> None:
    manifest = json.loads((base_root / "reports" / "sources_manifest.json").read_text(encoding="utf-8"))
    manifest["built_for"] = "Murmur 203.04M quality-focused bilingual continued SFT v4"
    manifest["sources"].append(
        {
            "repo": INFINITY_REPO_ID,
            "revision": INFINITY_REVISION,
            "license": "CC-BY-SA-4.0",
            "access": "official gated access accepted by the user; no access control bypassed",
            "files": [f"{shard.subset}/{shard.filename}" for shard in INFINITY_SHARDS],
            "purpose": "reward-filtered bilingual instruction following, general reasoning, writing, and code",
            "selection": [
                {
                    "subset": shard.subset,
                    "language_reward_thresholds": dict(shard.reward_thresholds),
                    "language_quotas": dict(shard.quotas),
                    "allowed_upstream_sources": list(shard.allowed_sources),
                    "max_tokens": 1_024,
                }
                for shard in INFINITY_SHARDS
            ],
        }
    )
    (report_dir / "sources_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_candidate_decisions(base_root: Path, report_dir: Path) -> None:
    decisions = json.loads((base_root / "reports" / "candidate_decisions.json").read_text(encoding="utf-8"))
    decisions["included"].extend(
        [
            "BAAI/Infinity-Instruct 7M_core: one pinned shard, bilingual reward thresholds, curated upstream source allowlist",
            "BAAI/Infinity-Instruct Gen: one pinned shard, bilingual reward thresholds, stable two-party instruction responses",
        ]
    )
    decisions["excluded"].extend(
        [
            {
                "candidate": "Infinity-Instruct FLAN, Orca Math, and MetaMath rows",
                "reason": "Excluded to reduce benchmark-style contamination, terse-answer imitation, and unverified math duplication for a compact model.",
            },
            {
                "candidate": "Low-reward or high-stakes Infinity-Instruct rows",
                "reason": "Excluded by language-specific reward thresholds and deterministic filters for medical, legal, financial, current-event, harmful, identity, and formatting risks.",
            },
            {
                "candidate": "Full Infinity-Instruct repository",
                "reason": "Not downloaded or mixed wholesale; quality-focused pinned shards avoid overwhelming the 203M model and keep the release under 1 GB.",
            },
        ]
    )
    (report_dir / "candidate_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_selection_report(
    path: Path,
    *,
    local_sources: dict[str, str],
    selected_counts: dict[str, int],
    additions: list[SFTRecord],
    record_filter: RecordFilter,
) -> None:
    report = {
        "repo": INFINITY_REPO_ID,
        "revision": INFINITY_REVISION,
        "raw_files": [
            {
                "key": key,
                "path": local_path,
                "bytes": Path(local_path).stat().st_size,
                "sha256": _sha256(Path(local_path)),
            }
            for key, local_path in sorted(local_sources.items())
        ],
        "policy": [
            {
                "key": shard.key,
                "subset": shard.subset,
                "reward_thresholds": dict(shard.reward_thresholds),
                "quotas": dict(shard.quotas),
                "allowed_sources": list(shard.allowed_sources),
            }
            for shard in INFINITY_SHARDS
        ],
        "selected_before_base_deduplication": dict(sorted(selected_counts.items())),
        "added_after_base_deduplication": dict(
            sorted(Counter(record.source for record in additions).items())
        ),
        "added_categories": dict(
            sorted(Counter(record.category for record in additions).items())
        ),
        "record_filter": {
            "accepted_before_sampling": record_filter.stats.accepted,
            "rejected": dict(sorted(record_filter.stats.rejected.items())),
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_v4_dataset(options: V4Options) -> dict[str, Any]:
    base_root = options.base_root.resolve()
    output_root = options.output_root.resolve()
    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed"
    report_dir = output_root / "reports"
    assets_dir = output_root / "assets"
    for directory in (raw_dir, processed_dir, report_dir, assets_dir, output_root / "curated"):
        directory.mkdir(parents=True, exist_ok=True)

    if options.download:
        _download_sources(raw_dir)
    local_sources = _local_sources(raw_dir)
    portable_tokenizer = assets_dir / options.tokenizer_path.name
    shutil.copy2(options.tokenizer_path.resolve(), portable_tokenizer)
    record_filter = RecordFilter(
        str(portable_tokenizer), max_tokens=options.max_tokens, min_assistant_tokens=2
    )
    selected, selected_counts = _select_additions(raw_dir, record_filter, options.seed)

    base_train_path = base_root / "processed" / "train.jsonl"
    base_validation_path = base_root / "processed" / "validation.jsonl"
    base_paths = (base_train_path, base_validation_path)
    base_train_records = _count_rows(base_train_path)
    base_validation_records = _count_rows(base_validation_path)
    seen = _base_content_keys(base_paths)

    additions: list[SFTRecord] = []
    for record in selected:
        if record.content_key in seen:
            continue
        seen.add(record.content_key)
        additions.append(record)
    additions.sort(key=lambda record: stable_hex(f"{options.seed}:{record.record_id}", 32))
    extra_train, extra_validation = _split(additions, options.validation_ratio)

    train_records = _write_merged_jsonl(
        processed_dir / "train.jsonl", _load_jsonl(base_train_path), extra_train
    )
    validation_records = _write_merged_jsonl(
        processed_dir / "validation.jsonl",
        _load_jsonl(base_validation_path),
        extra_validation,
    )
    _copy_support_artifacts(base_root, output_root)
    _write_sample_audit(report_dir / "sample_audit.json", additions, options.seed)

    train_keys = {
        _content_key(row["messages"]) for row in _load_jsonl(processed_dir / "train.jsonl")
    }
    validation_keys = {
        _content_key(row["messages"])
        for row in _load_jsonl(processed_dir / "validation.jsonl")
    }

    def all_rows() -> Iterator[dict[str, Any]]:
        return chain(
            _load_jsonl(processed_dir / "train.jsonl"),
            _load_jsonl(processed_dir / "validation.jsonl"),
        )

    token_statistics = _token_statistics(all_rows())
    scenario_coverage = build_scenario_coverage(all_rows())
    (report_dir / "scenario_coverage.json").write_text(
        json.dumps(scenario_coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_sources_manifest(base_root, report_dir)
    _write_candidate_decisions(base_root, report_dir)
    _write_selection_report(
        report_dir / "infinity_selection_report.json",
        local_sources=local_sources,
        selected_counts=selected_counts,
        additions=additions,
        record_filter=record_filter,
    )

    summary: dict[str, Any] = {
        "base_dataset": str(base_root),
        "base_records": base_train_records + base_validation_records,
        "base_train_records": base_train_records,
        "base_validation_records": base_validation_records,
        "added_records": len(additions),
        "total_records": train_records + validation_records,
        "train_records": train_records,
        "validation_records": validation_records,
        "added_by_source": dict(sorted(Counter(record.source for record in additions).items())),
        "added_by_category": dict(sorted(Counter(record.category for record in additions).items())),
        "max_tokens_for_additions": options.max_tokens,
        "seed": options.seed,
        "train_sha256": _sha256(processed_dir / "train.jsonl"),
        "validation_sha256": _sha256(processed_dir / "validation.jsonl"),
        "tokenizer_sha256": _sha256(portable_tokenizer),
        "train_validation_content_overlap": len(train_keys & validation_keys),
        "scenario_coverage_ratio": scenario_coverage["coverage_ratio"],
        "raw_sources": local_sources,
        "filter": {
            "accepted_before_sampling": record_filter.stats.accepted,
            "rejected": dict(sorted(record_filter.stats.rejected.items())),
        },
    }
    summary.update(token_statistics)
    (report_dir / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

