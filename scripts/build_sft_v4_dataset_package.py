from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from build_sft_dataset_package import (
    copy_package_files,
    create_archive,
    sha256_file,
    write_manifest,
)


V4_PROJECT_FILES = (
    "README_SFT_V4.md",
    "run_gpu_v4.sh",
    "configs/sft_203m_cloud_v4_continued.yaml",
    "muddywater/templates.py",
    "muddywater/sft_data/__init__.py",
    "muddywater/sft_data/augmentation_catalog.py",
    "muddywater/sft_data/augmentation_identity.py",
    "muddywater/sft_data/augmentation_pipeline.py",
    "muddywater/sft_data/augmentation_readers.py",
    "muddywater/sft_data/augmentation_synthetic.py",
    "muddywater/sft_data/catalog.py",
    "muddywater/sft_data/coverage.py",
    "muddywater/sft_data/download.py",
    "muddywater/sft_data/filtering.py",
    "muddywater/sft_data/pipeline.py",
    "muddywater/sft_data/readers.py",
    "muddywater/sft_data/records.py",
    "muddywater/sft_data/sampling.py",
    "muddywater/sft_data/synthetic.py",
    "muddywater/sft_data/v3_catalog.py",
    "muddywater/sft_data/v3_pipeline.py",
    "muddywater/sft_data/v3_readers.py",
    "muddywater/sft_data/v4_catalog.py",
    "muddywater/sft_data/v4_pipeline.py",
    "muddywater/sft_data/v4_readers.py",
    "muddywater/sft_data/validation.py",
    "scripts/prepare_resume_config.py",
    "scripts/prepare_sft_203m_v4.py",
    "scripts/validate_sft_203m.py",
    "tests/test_sft_data.py",
    "tests/test_sft_augmentation.py",
    "tests/test_sft_v4.py",
)

V4_DATA_FILES = (
    "assets/sp_unigram_32k.model",
    "curated/murmur_identity.jsonl",
    "processed/train.jsonl",
    "processed/validation.jsonl",
    "reports/augmentation_summary.json",
    "reports/candidate_decisions.json",
    "reports/identity_eval_prompts.json",
    "reports/identity_validation.json",
    "reports/infinity_selection_report.json",
    "reports/sample_audit.json",
    "reports/scenario_coverage.json",
    "reports/sources_manifest.json",
    "reports/validation_report.json",
)


def _copy_data_files(data_root: Path, package_root: Path) -> None:
    destination_root = package_root / "data" / "sft_203m_v4"
    for relative_name in V4_DATA_FILES:
        source = data_root / relative_name
        if not source.is_file():
            raise FileNotFoundError(f"Required v4 data file is missing: {source}")
        destination = destination_root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Murmur SFT v4 upload package.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    package_root = args.package_root.resolve()
    archive_path = args.archive.resolve()
    if package_root.exists() or archive_path.exists():
        raise FileExistsError("Package directory or archive already exists; choose a new target name.")
    package_root.mkdir(parents=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    copy_package_files(project_root, package_root, V4_PROJECT_FILES)
    _copy_data_files(data_root, package_root)
    manifest = write_manifest(package_root)
    create_archive(package_root, archive_path)
    archive_bytes = archive_path.stat().st_size
    if archive_bytes >= args.max_bytes:
        archive_path.unlink()
        raise ValueError(f"Archive exceeds size limit: {archive_bytes} >= {args.max_bytes}")
    print(
        json.dumps(
            {
                "package_root": str(package_root),
                "archive": str(archive_path),
                "archive_bytes": archive_bytes,
                "archive_sha256": sha256_file(archive_path),
                "files": manifest["totals"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

