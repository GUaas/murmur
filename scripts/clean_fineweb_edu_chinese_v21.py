from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.parquet_corpus import (
    CorpusCleaningPolicy,
    clean_parquet_corpus,
    load_completed_parquet_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a completed Fineweb-Edu-Chinese-V2.1 Parquet selection."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target-root",
        default=None,
        help="Override dataset.target_root from the YAML configuration.",
    )
    parser.add_argument("--progress-interval", type=int, default=50_000)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


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
    manifest_name = str(dataset_config.get("manifest_name", "download_manifest.json"))
    if Path(manifest_name).name != manifest_name:
        raise ValueError("dataset.manifest_name must be a file name, not a path")
    manifest_path = target_root / manifest_name
    manifest, input_paths = load_completed_parquet_paths(manifest_path)
    policy = CorpusCleaningPolicy(**cleaning_config)
    report = clean_parquet_corpus(
        input_paths=input_paths,
        output_dir=target_root / "cleaned",
        report_path=target_root / "cleaning_report.json",
        source_manifest=manifest,
        policy=policy,
        progress_interval=args.progress_interval,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "input_files": report["input_files"],
                "raw_documents": report["raw_documents"],
                "kept_records": report["kept_records"],
                "output_bytes": report["output_bytes"],
                "output_dir": report["output_dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
