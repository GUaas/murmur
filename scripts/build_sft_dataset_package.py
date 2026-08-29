from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


PACKAGE_FILES = (
    "README_SFT_V2.md",
    "run_gpu_v2.sh",
    "configs/sft_203m_cloud_v2_continued.yaml",
    "data/sft_203m_v2/assets/sp_unigram_32k.model",
    "data/sft_203m_v2/curated/murmur_identity.jsonl",
    "data/sft_203m_v2/processed/train.jsonl",
    "data/sft_203m_v2/processed/validation.jsonl",
    "data/sft_203m_v2/reports/augmentation_summary.json",
    "data/sft_203m_v2/reports/candidate_decisions.json",
    "data/sft_203m_v2/reports/identity_eval_prompts.json",
    "data/sft_203m_v2/reports/identity_validation.json",
    "data/sft_203m_v2/reports/sample_audit.json",
    "data/sft_203m_v2/reports/scenario_coverage.json",
    "data/sft_203m_v2/reports/sources_manifest.json",
    "data/sft_203m_v2/reports/validation_report.json",
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
    "muddywater/sft_data/validation.py",
    "scripts/prepare_resume_config.py",
    "scripts/prepare_sft_203m_v2.py",
    "scripts/validate_sft_203m.py",
    "tests/test_sft_augmentation.py",
    "tests/test_sft_data.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_package_files(
    project_root: Path,
    package_root: Path,
    package_files: tuple[str, ...] = PACKAGE_FILES,
) -> None:
    for relative_name in package_files:
        source = project_root / relative_name
        if not source.is_file():
            raise FileNotFoundError(f"Required package file is missing: {source}")
        destination = package_root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_manifest(package_root: Path) -> dict[str, object]:
    files = sorted(path for path in package_root.rglob("*") if path.is_file())
    entries = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest: dict[str, object] = {
        "package": package_root.name,
        "format_version": 1,
        "contains_model_weights": False,
        "contains_raw_source_downloads": False,
        "files": entries,
        "totals": {
            "files": len(entries),
            "bytes_excluding_manifest": sum(int(entry["bytes"]) for entry in entries),
        },
    }
    (package_root / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def create_archive(package_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    (Path(package_root.name) / path.relative_to(package_root)).as_posix(),
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Murmur SFT dataset-only upload package.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    package_root = args.package_root.resolve()
    archive_path = args.archive.resolve()
    if package_root.exists() or archive_path.exists():
        raise FileExistsError("Package directory or archive already exists; choose a new target name.")
    package_root.mkdir(parents=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    copy_package_files(project_root, package_root)
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
