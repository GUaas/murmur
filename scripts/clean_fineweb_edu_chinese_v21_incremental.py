from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.parquet_corpus import (
    CorpusCleaningPolicy,
    clean_parquet_corpus,
    load_completed_parquet_paths,
    load_jsonl_dedupe_seed,
)
from muddywater.utils import atomic_write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean only newly downloaded Fineweb-Edu-Chinese-V2.1 Parquet files "
            "while deduplicating against an existing cleaned corpus."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target-root",
        default=None,
        help="Override dataset.target_root from the YAML configuration.",
    )
    parser.add_argument("--progress-interval", type=int, default=50_000)
    parser.add_argument("--seed-progress-interval", type=int, default=500_000)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def load_json_mapping(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def config_file_name(config: dict[str, Any], key: str, default: str) -> str:
    value = str(config.get(key, default))
    if Path(value).name != value:
        raise ValueError(f"dataset.{key} must be a file name, not a path")
    return value


def validate_manifest_union(
    *,
    base_manifest: dict[str, Any],
    extension_manifest: dict[str, Any],
    full_manifest: dict[str, Any],
) -> None:
    manifests = [base_manifest, extension_manifest, full_manifest]
    if any(manifest.get("status") != "completed" for manifest in manifests):
        statuses = [manifest.get("status") for manifest in manifests]
        raise ValueError(f"All download manifests must be completed; statuses={statuses}")

    datasets = [manifest.get("dataset", {}) for manifest in manifests]
    if datasets[1:] != datasets[:-1]:
        raise ValueError(f"Download manifest dataset mismatch: {datasets}")

    base_paths = set(map(str, base_manifest.get("selected_repo_paths", [])))
    extension_paths = set(map(str, extension_manifest.get("selected_repo_paths", [])))
    full_paths = set(map(str, full_manifest.get("selected_repo_paths", [])))
    overlap = base_paths & extension_paths
    if overlap:
        raise ValueError(
            f"Base and extension manifests overlap on {len(overlap)} files; "
            f"first={sorted(overlap)[:3]}"
        )
    if base_paths | extension_paths != full_paths:
        raise ValueError(
            "Full manifest is not the exact union of base and extension manifests"
        )


def validate_policy(
    policy: CorpusCleaningPolicy,
    base_report: dict[str, Any],
) -> None:
    expected = base_report.get("policy")
    actual = asdict(policy)
    if expected != actual:
        raise ValueError(
            "Incremental cleaning policy must exactly match the existing cleaned corpus: "
            f"existing={expected}, requested={actual}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_config = config.get("dataset", {})
    cleaning_config = config.get("cleaning", {})
    if not isinstance(dataset_config, dict) or not isinstance(cleaning_config, dict):
        raise ValueError("dataset and cleaning configuration sections must be mappings")

    target_root_value = args.target_root or dataset_config.get("target_root")
    if not target_root_value:
        raise ValueError("A dataset target_root is required")
    target_root = Path(str(target_root_value)).resolve()

    base_manifest_name = config_file_name(
        dataset_config,
        "base_manifest_name",
        "download_combined_manifest.json",
    )
    extension_manifest_name = config_file_name(
        dataset_config,
        "extension_manifest_name",
        "download_remainder_manifest.json",
    )
    full_manifest_name = config_file_name(
        dataset_config,
        "full_manifest_name",
        "download_full_manifest.json",
    )
    base_report_name = config_file_name(
        dataset_config,
        "base_report_name",
        "cleaning_report.json",
    )
    extension_report_name = config_file_name(
        dataset_config,
        "extension_report_name",
        "cleaning_extension_report.json",
    )
    combined_report_name = config_file_name(
        dataset_config,
        "combined_report_name",
        "cleaning_full_report.json",
    )
    base_cleaned_dir_name = config_file_name(
        dataset_config,
        "base_cleaned_dir",
        "cleaned",
    )
    extension_cleaned_dir_name = config_file_name(
        dataset_config,
        "extension_cleaned_dir",
        "cleaned_extension",
    )

    base_manifest = load_json_mapping(
        target_root / base_manifest_name,
        description="Base download manifest",
    )
    extension_manifest, extension_paths = load_completed_parquet_paths(
        target_root / extension_manifest_name
    )
    full_manifest = load_json_mapping(
        target_root / full_manifest_name,
        description="Full download manifest",
    )
    validate_manifest_union(
        base_manifest=base_manifest,
        extension_manifest=extension_manifest,
        full_manifest=full_manifest,
    )

    base_report = load_json_mapping(
        target_root / base_report_name,
        description="Base cleaning report",
    )
    if base_report.get("status") != "completed":
        raise ValueError(
            f"Base cleaning report must be completed; status={base_report.get('status')!r}"
        )

    policy = CorpusCleaningPolicy(**cleaning_config)
    policy.validate()
    validate_policy(policy, base_report)

    expected_shards = [
        str(item["file"])
        for item in base_report.get("output_shards", [])
        if isinstance(item, dict) and item.get("file")
    ]
    seed_digests, seed_stats = load_jsonl_dedupe_seed(
        target_root / base_cleaned_dir_name,
        dedupe_mode=policy.dedupe_mode,
        expected_records=int(base_report["kept_records"]),
        expected_shard_names=expected_shards,
        progress_interval=args.seed_progress_interval,
    )
    if seed_stats["duplicate_records"]:
        raise ValueError(
            "Existing cleaned corpus unexpectedly contains duplicate dedupe digests: "
            f"{seed_stats['duplicate_records']}"
        )

    extension_report = clean_parquet_corpus(
        input_paths=extension_paths,
        output_dir=target_root / extension_cleaned_dir_name,
        report_path=target_root / extension_report_name,
        source_manifest=extension_manifest,
        policy=policy,
        progress_interval=args.progress_interval,
        seed_digests=seed_digests,
        report_metadata={
            "mode": "incremental_global_exact_dedupe",
            "base_cleaned_dir": str(target_root / base_cleaned_dir_name),
            "base_report": base_report_name,
            "base_manifest": base_manifest_name,
            "extension_manifest": extension_manifest_name,
            "full_manifest": full_manifest_name,
            "seed": seed_stats,
        },
    )

    combined_report = {
        "schema_version": 1,
        "status": "completed",
        "dataset": full_manifest.get("dataset"),
        "source_selection": full_manifest.get("selection"),
        "policy": asdict(policy),
        "base": {
            "report": base_report_name,
            "output_dir": str(target_root / base_cleaned_dir_name),
            "input_files": int(base_report["input_files"]),
            "kept_records": int(base_report["kept_records"]),
            "output_bytes": int(base_report["output_bytes"]),
        },
        "extension": {
            "report": extension_report_name,
            "output_dir": str(target_root / extension_cleaned_dir_name),
            "input_files": int(extension_report["input_files"]),
            "kept_records": int(extension_report["kept_records"]),
            "output_bytes": int(extension_report["output_bytes"]),
            "duplicates_against_base": int(
                extension_report.get("drop_reasons", {}).get(
                    f"duplicate_against_seed_{policy.dedupe_mode}",
                    0,
                )
            ),
        },
        "input_files": int(base_report["input_files"])
        + int(extension_report["input_files"]),
        "kept_records": int(base_report["kept_records"])
        + int(extension_report["kept_records"]),
        "output_bytes": int(base_report["output_bytes"])
        + int(extension_report["output_bytes"]),
    }
    atomic_write_text(
        target_root / combined_report_name,
        json.dumps(combined_report, ensure_ascii=False, indent=2),
    )
    print(json.dumps(combined_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
