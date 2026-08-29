from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

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
from .v3_catalog import DOLLY_CATEGORY_QUOTAS, V3_HF_DATASETS, V3_QUOTAS
from .v3_readers import read_dolly_curated, read_drcd, read_tulu_instruction_following


@dataclass(slots=True)
class V3Options:
    base_root: Path
    output_root: Path
    tokenizer_path: Path
    max_tokens: int = 1024
    validation_ratio: float = 0.005
    seed: int = 20260809
    download: bool = True


def _download_sources(raw_dir: Path) -> dict[str, list[str]]:
    downloaded: dict[str, list[str]] = {}
    for spec in V3_HF_DATASETS:
        local_dir = raw_dir / spec.key
        local_dir.mkdir(parents=True, exist_ok=True)
        paths = [
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=spec.revision,
                local_dir=local_dir,
            )
            for filename in spec.files
        ]
        downloaded[spec.key] = [str(Path(path).resolve()) for path in paths]
    return downloaded


def _select_extensions(raw_dir: Path, record_filter: RecordFilter, seed: int) -> list[SFTRecord]:
    sources = [
        ("drcd", read_drcd(raw_dir)),
        ("tulu_instruction_following", read_tulu_instruction_following(raw_dir)),
        *[
            (f"dolly_{category}", read_dolly_curated(raw_dir, category))
            for category in DOLLY_CATEGORY_QUOTAS
        ],
    ]
    selected: list[SFTRecord] = []
    for source, records in sources:
        selected.extend(
            select_records(
                records,
                quota=V3_QUOTAS[source],
                record_filter=record_filter,
                seed=seed,
            )
        )
    return selected


def _copy_identity_artifacts(base_root: Path, output_root: Path) -> None:
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
    manifest["built_for"] = "Murmur 203.04M all-scenario continued SFT v3"
    manifest["sources"].extend(
        [
            {
                "repo": "ihainan/DRCD-Simplified-Chinese",
                "revision": "de11764c42349f940e89b0dbfcff16b26a45056f",
                "license": "CC-BY-SA-3.0",
                "purpose": "human-annotated Chinese extractive reading comprehension",
            },
            {
                "repo": "argilla/databricks-dolly-15k-curated-en",
                "revision": "4dcd1dedbe148307a833c931b21ca456a1fc4281",
                "license": "CC-BY-SA-3.0",
                "purpose": "human-written and collaboratively curated instruction diversity",
                "selected_categories": list(DOLLY_CATEGORY_QUOTAS),
            },
            {
                "repo": "allenai/tulu-3-sft-personas-instruction-following",
                "revision": "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
                "license": "ODC-BY-1.0",
                "purpose": "constraint-driven and verifiable strict instruction following",
                "selection": "one or two constraints, stable non-high-stakes prompts, <=1024 tokens",
            },
        ]
    )
    (report_dir / "sources_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_candidate_decisions(base_root: Path, report_dir: Path) -> None:
    decisions = json.loads((base_root / "reports" / "candidate_decisions.json").read_text(encoding="utf-8"))
    decisions["included"].extend(
        [
            "ihainan/DRCD-Simplified-Chinese: training split only",
            "argilla/databricks-dolly-15k-curated-en: grounded extraction, summarization, and closed-QA categories only",
            "allenai/tulu-3-sft-personas-instruction-following: short stable examples with one or two explicit constraints",
        ]
    )
    decisions["excluded"].extend(
        [
            {
                "candidate": "OpenAssistant/oasst2 Chinese subset",
                "reason": "Even highly reviewed samples included incorrect and outdated technical guidance during manual audit.",
            },
            {
                "candidate": "Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese",
                "reason": "No explicit dataset license was present in the official card at review time.",
            },
            {
                "candidate": "m-a-p/COIG-P",
                "reason": "Preference data card did not declare a dataset license at review time; chosen responses were not repackaged.",
            },
            {
                "candidate": "bigscience/xP3 Chinese mixture",
                "reason": "Large prompt-template multiplicity and benchmark contamination risk were a poor fit for this compact model.",
            },
        ]
    )
    (report_dir / "candidate_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_v3_dataset(options: V3Options) -> dict[str, Any]:
    base_root = options.base_root.resolve()
    output_root = options.output_root.resolve()
    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed"
    report_dir = output_root / "reports"
    assets_dir = output_root / "assets"
    for directory in (raw_dir, processed_dir, report_dir, assets_dir, output_root / "curated"):
        directory.mkdir(parents=True, exist_ok=True)

    downloaded = _download_sources(raw_dir) if options.download else {}
    portable_tokenizer = assets_dir / options.tokenizer_path.name
    shutil.copy2(options.tokenizer_path.resolve(), portable_tokenizer)
    record_filter = RecordFilter(
        str(portable_tokenizer), max_tokens=options.max_tokens, min_assistant_tokens=2
    )
    selected = _select_extensions(raw_dir, record_filter, options.seed)

    base_train_rows = list(_load_jsonl(base_root / "processed" / "train.jsonl"))
    base_validation_rows = list(_load_jsonl(base_root / "processed" / "validation.jsonl"))
    seen = {
        _content_key(row["messages"])
        for row in (*base_train_rows, *base_validation_rows)
    }
    additions: list[SFTRecord] = []
    for record in selected:
        if record.content_key in seen:
            continue
        seen.add(record.content_key)
        additions.append(record)
    additions.sort(key=lambda record: stable_hex(f"{options.seed}:{record.record_id}", 32))

    extra_train, extra_validation = _split(additions, options.validation_ratio)
    train_records = _write_merged_jsonl(
        processed_dir / "train.jsonl", base_train_rows, extra_train
    )
    validation_records = _write_merged_jsonl(
        processed_dir / "validation.jsonl", base_validation_rows, extra_validation
    )
    _copy_identity_artifacts(base_root, output_root)
    _write_sample_audit(report_dir / "sample_audit.json", additions, options.seed)

    train_keys = {_content_key(row["messages"]) for row in _load_jsonl(processed_dir / "train.jsonl")}
    validation_keys = {
        _content_key(row["messages"])
        for row in _load_jsonl(processed_dir / "validation.jsonl")
    }
    all_rows = lambda: chain(
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

    summary: dict[str, Any] = {
        "base_dataset": str(base_root),
        "base_records": len(base_train_rows) + len(base_validation_rows),
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
        "downloaded": downloaded,
        "filter": {
            "accepted_before_sampling": record_filter.stats.accepted,
            "rejected": dict(record_filter.stats.rejected),
        },
    }
    summary.update(token_statistics)
    (report_dir / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
