from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator

from huggingface_hub import hf_hub_download

from .augmentation_catalog import (
    ADDITIONAL_HF_DATASETS,
    AUGMENTATION_QUOTAS,
)
from .augmentation_readers import (
    read_cmrc2018,
    read_coig_counterfactual,
    read_coig_human_value,
    read_coig_leetcode,
    read_doit,
    read_gsm8k_zh,
)
from .augmentation_identity import identity_evaluation_cases, read_verified_identity_curriculum
from .augmentation_synthetic import (
    read_verified_instruction_curriculum,
    read_verified_math_curriculum,
    read_verified_safety_curriculum,
)
from .filtering import RecordFilter
from .coverage import build_scenario_coverage
from .records import SFTRecord, stable_hex
from .sampling import select_records


@dataclass(slots=True)
class AugmentationOptions:
    base_root: Path
    output_root: Path
    tokenizer_path: Path
    max_tokens: int = 1024
    validation_ratio: float = 0.005
    seed: int = 20260808
    download: bool = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_sources(raw_dir: Path) -> dict[str, list[str]]:
    downloaded: dict[str, list[str]] = {}
    for spec in ADDITIONAL_HF_DATASETS:
        local_dir = raw_dir / spec.key
        local_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for filename in spec.files:
            path = hf_hub_download(
                repo_id=spec.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=spec.revision,
                local_dir=local_dir,
            )
            paths.append(str(Path(path).resolve()))
        for metadata_name in ("README.md", "LICENSE", "LICENSE.md"):
            try:
                hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=metadata_name,
                    repo_type="dataset",
                    revision=spec.revision,
                    local_dir=local_dir,
                )
            except Exception:
                pass
        downloaded[spec.key] = paths
    return downloaded


def _content_key(messages: list[dict[str, str]]) -> str:
    compact = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _split(records: Iterable[SFTRecord], ratio: float) -> tuple[list[SFTRecord], list[SFTRecord]]:
    threshold = int(ratio * 1_000_000)
    train: list[SFTRecord] = []
    validation: list[SFTRecord] = []
    for record in records:
        bucket = int(stable_hex(record.group_id, 16), 16) % 1_000_000
        (validation if bucket < threshold else train).append(record)
    return train, validation


def _write_merged_jsonl(
    destination: Path,
    base_rows: Iterable[dict[str, Any]],
    additions: Iterable[SFTRecord],
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in base_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
        for record in additions:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, destination)
    return count


def _select_additions(raw_dir: Path, record_filter: RecordFilter, seed: int) -> list[SFTRecord]:
    sources = (
        ("gsm8k_zh", read_gsm8k_zh(raw_dir)),
        ("coig_human_value", read_coig_human_value(raw_dir)),
        ("coig_leetcode", read_coig_leetcode(raw_dir)),
        ("coig_counterfactual", read_coig_counterfactual(raw_dir)),
        ("cmrc2018", read_cmrc2018(raw_dir)),
        ("doit_chinese", read_doit(raw_dir, "chinese")),
        ("doit_code", read_doit(raw_dir, "code")),
        ("doit_creative_writing", read_doit(raw_dir, "creative_writing")),
        ("doit_history", read_doit(raw_dir, "history")),
        ("doit_reasoning", read_doit(raw_dir, "reasoning")),
        ("doit_role_play", read_doit(raw_dir, "role_play")),
        ("doit_understanding", read_doit(raw_dir, "understanding")),
        ("synthetic_math_verified", read_verified_math_curriculum()),
        ("synthetic_safety_verified", read_verified_safety_curriculum()),
        ("synthetic_instruction_verified", read_verified_instruction_curriculum()),
        ("synthetic_identity_verified", read_verified_identity_curriculum()),
    )
    selected: list[SFTRecord] = []
    for source, records in sources:
        selected.extend(
            select_records(
                records,
                quota=AUGMENTATION_QUOTAS[source],
                record_filter=record_filter,
                seed=seed,
            )
        )
    return selected


def _token_statistics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    token_lengths: list[int] = []
    assistant_tokens = 0
    context_tokens = 0
    for row in rows:
        length = int(row.get("num_tokens", 0))
        token_lengths.append(length)
        context_tokens += length
        assistant_tokens += int(row.get("assistant_tokens", 0))
    token_lengths.sort()

    def percentile(fraction: float) -> int:
        index = min(len(token_lengths) - 1, int(len(token_lengths) * fraction))
        return token_lengths[index]

    return {
        "context_tokens": context_tokens,
        "assistant_tokens": assistant_tokens,
        "token_length": {
            "min": token_lengths[0],
            "mean": round(statistics.fmean(token_lengths), 2),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": token_lengths[-1],
        },
    }


def _write_sample_audit(path: Path, additions: list[SFTRecord], seed: int) -> None:
    by_source: dict[str, list[SFTRecord]] = {}
    for record in additions:
        by_source.setdefault(record.source, []).append(record)
    audit: list[dict[str, Any]] = []
    for source, records in sorted(by_source.items()):
        ordered = sorted(records, key=lambda record: stable_hex(f"audit:{seed}:{record.record_id}", 32))
        for record in ordered[:5]:
            audit.append(record.to_dict())
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_candidate_decisions(path: Path) -> None:
    decisions = {
        "selection_principles": [
            "Prefer official dataset repositories with explicit licenses and pinned revisions.",
            "Retain only records that pass structural, identity, noise, and token-length filters.",
            "Reject a source when manual samples reveal systematic quality or safety problems.",
            "Keep all raw downloads outside the upload package.",
        ],
        "included": [
            "meta-math/GSM8K_zh",
            "BAAI/COIG: human value, LeetCode, counterfactual multi-round subsets",
            "ChiyuSONG/dynamics-of-instruction-tuning: selected reviewed subsets",
            "hfl/cmrc2018",
            "project-generated deterministic verified curricula",
        ],
        "excluded": [
            {
                "candidate": "openbmb/UltraData-SFT-2605",
                "reason": "Repository access is gated; no access-control bypass was attempted.",
            },
            {
                "candidate": "PaddlePaddle DuReader Robust training split",
                "reason": "Manual samples contained user-generated medical, promotional, and stale web answers despite useful extractive-QA structure.",
            },
            {
                "candidate": "DoIT biology subset",
                "reason": "Most answers were bare multiple-choice letters without explanations.",
            },
            {
                "candidate": "DoIT math subset",
                "reason": "Manual audit found incorrect examples; verified math curricula were used instead.",
            },
        ],
    }
    path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_identity_records(records: list[SFTRecord]) -> dict[str, Any]:
    model_name_categories = {"identity_name", "identity_combined", "identity_correction"}
    developer_categories = {"identity_developer", "identity_combined", "identity_correction"}
    errors: list[str] = []
    unique_contents = {record.content_key for record in records}
    if len(unique_contents) != len(records):
        errors.append("identity_content_duplicates")
    for record in records:
        answer = record.messages[-1]["content"]
        if record.category in model_name_categories and "murmur" not in answer:
            errors.append(f"missing_model_name:{record.source_id}")
        if record.category in developer_categories and "MuddyWaterAI" not in answer:
            errors.append(f"missing_developer:{record.source_id}")
    report = {
        "valid": not errors,
        "records": len(records),
        "unique_contents": len(unique_contents),
        "categories": dict(sorted(Counter(record.category for record in records).items())),
        "max_tokens": max((record.num_tokens for record in records), default=0),
        "canonical_model_name": "murmur",
        "canonical_developer": "MuddyWaterAI",
        "errors": errors,
    }
    if errors:
        raise ValueError(f"Identity curriculum validation failed: {errors[:5]}")
    return report


def build_augmented_dataset(options: AugmentationOptions) -> dict[str, Any]:
    base_root = options.base_root.resolve()
    output_root = options.output_root.resolve()
    raw_dir = output_root / "raw"
    report_dir = output_root / "reports"
    processed_dir = output_root / "processed"
    curated_dir = output_root / "curated"
    for directory in (raw_dir, report_dir, processed_dir, curated_dir, output_root / "assets"):
        directory.mkdir(parents=True, exist_ok=True)

    downloaded = _download_sources(raw_dir) if options.download else {}
    tokenizer_path = options.tokenizer_path.resolve()
    portable_tokenizer = output_root / "assets" / tokenizer_path.name
    if not portable_tokenizer.exists() or _sha256(portable_tokenizer) != _sha256(tokenizer_path):
        shutil.copy2(tokenizer_path, portable_tokenizer)

    record_filter = RecordFilter(
        str(portable_tokenizer),
        max_tokens=options.max_tokens,
        min_assistant_tokens=2,
    )
    selected = _select_additions(raw_dir, record_filter, options.seed)

    base_train_path = base_root / "processed" / "train.jsonl"
    base_validation_path = base_root / "processed" / "validation.jsonl"
    base_train_rows = list(_load_jsonl(base_train_path))
    base_validation_rows = list(_load_jsonl(base_validation_path))
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
    identity_records = [
        record for record in additions if record.source == "synthetic_identity_verified"
    ]
    identity_path = curated_dir / "murmur_identity.jsonl"
    identity_count = _write_merged_jsonl(identity_path, (), identity_records)
    identity_report = _validate_identity_records(identity_records)
    (report_dir / "identity_validation.json").write_text(
        json.dumps(identity_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "identity_eval_prompts.json").write_text(
        json.dumps(identity_evaluation_cases(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    extra_train, extra_validation = _split(additions, options.validation_ratio)

    train_records = _write_merged_jsonl(processed_dir / "train.jsonl", base_train_rows, extra_train)
    validation_records = _write_merged_jsonl(
        processed_dir / "validation.jsonl", base_validation_rows, extra_validation
    )

    _write_sample_audit(report_dir / "sample_audit.json", additions, options.seed)

    source_counts = Counter(record.source for record in additions)
    category_counts = Counter(record.category for record in additions)
    train_keys = {_content_key(row["messages"]) for row in _load_jsonl(processed_dir / "train.jsonl")}
    validation_keys = {
        _content_key(row["messages"])
        for row in _load_jsonl(processed_dir / "validation.jsonl")
    }
    overlap = len(train_keys & validation_keys)
    combined_statistics = _token_statistics(
        chain(
            _load_jsonl(processed_dir / "train.jsonl"),
            _load_jsonl(processed_dir / "validation.jsonl"),
        )
    )
    scenario_coverage = build_scenario_coverage(
        chain(
            _load_jsonl(processed_dir / "train.jsonl"),
            _load_jsonl(processed_dir / "validation.jsonl"),
        )
    )
    (report_dir / "scenario_coverage.json").write_text(
        json.dumps(scenario_coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_candidate_decisions(report_dir / "candidate_decisions.json")
    summary: dict[str, Any] = {
        "base_records": len(base_train_rows) + len(base_validation_rows),
        "added_records": len(additions),
        "total_records": train_records + validation_records,
        "train_records": train_records,
        "validation_records": validation_records,
        "added_by_source": dict(sorted(source_counts.items())),
        "added_by_category": dict(sorted(category_counts.items())),
        "max_tokens_for_additions": options.max_tokens,
        "seed": options.seed,
        "train_sha256": _sha256(processed_dir / "train.jsonl"),
        "validation_sha256": _sha256(processed_dir / "validation.jsonl"),
        "identity_records": identity_count,
        "identity_sha256": _sha256(identity_path),
        "tokenizer_sha256": _sha256(portable_tokenizer),
        "train_validation_content_overlap": overlap,
        "scenario_coverage_ratio": scenario_coverage["coverage_ratio"],
        "downloaded": downloaded,
        "filter": {
            "accepted_before_sampling": record_filter.stats.accepted,
            "rejected": dict(record_filter.stats.rejected),
        },
    }
    summary.update(combined_statistics)
    (report_dir / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "built_for": "Murmur 203.04M continued SFT",
        "sources": [
            {
                "repo": "meta-math/GSM8K_zh",
                "revision": "4a5009abc37cbb2d3fd1a745f80e5ea1405ba9aa",
                "license": "MIT",
                "purpose": "Chinese grade-school arithmetic reasoning",
            },
            {
                "repo": "BAAI/COIG",
                "revision": "9f25758ec94f82762fb9c09a5c60e908cfb83632",
                "license": "Apache-2.0 with permissive upstream components",
                "file_licenses": {
                    "leetcode_instructions.jsonl": "CC-BY-SA-4.0 upstream",
                    "counterfactural_correction_multi_round_chat.tar.gz": "Apache-2.0",
                    "human_value_alignment_instructions_part1.json": "Apache-2.0",
                },
                "files": [
                    "human_value_alignment_instructions_part1.json",
                    "leetcode_instructions.jsonl",
                    "counterfactural_correction_multi_round_chat.tar.gz",
                ],
                "purpose": "human-value alignment, Chinese code, and factual correction dialogues",
            },
            {
                "repo": "ChiyuSONG/dynamics-of-instruction-tuning",
                "revision": "4ae3b55e7fd7966aa59afb7b819558f682e4ef3c",
                "license": "MIT",
                "files": [
                    "curated/full/chinese_full.json",
                    "curated/full/code_full.json",
                    "curated/full/creative_writing_full.json",
                    "curated/full/history_full.json",
                    "curated/full/reasoning_full.json",
                    "curated/full/role_play_full.json",
                    "curated/full/understanding_full.json",
                ],
                "purpose": "human-reviewed Chinese humanities, code, instruction following, and reasoning",
            },
            {
                "repo": "hfl/cmrc2018",
                "revision": "137f2c45a24275fb68f6961c4d357f46288886aa",
                "license": "CC-BY-SA-4.0",
                "files": ["data/train-00000-of-00001.parquet"],
                "purpose": "expert-annotated Chinese reading comprehension",
            },
            {
                "source": "deterministic programmatic generators",
                "license": "project-generated",
                "purpose": "verified arithmetic, constrained output, safety, and murmur self-identity",
            },
        ],
    }
    (report_dir / "sources_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
