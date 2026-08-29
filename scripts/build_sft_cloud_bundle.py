from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL_NAME = "murmur_203m_best_weights_only.pt"
TOKENIZER_NAME = "sp_unigram_32k.model"


@dataclass(frozen=True)
class CopyItem:
    source: Path
    destination: Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required file not found: {resolved}")
    return resolved


def source_items(project_root: Path, dataset_root: Path, model_path: Path) -> list[CopyItem]:
    return [
        CopyItem(model_path, Path("model") / MODEL_NAME),
        CopyItem(project_root / "tokenizer" / TOKENIZER_NAME, Path("tokenizer") / TOKENIZER_NAME),
        CopyItem(dataset_root / "processed" / "train.jsonl", Path("data/sft_203m/processed/train.jsonl")),
        CopyItem(
            dataset_root / "processed" / "validation.jsonl",
            Path("data/sft_203m/processed/validation.jsonl"),
        ),
        CopyItem(dataset_root / "README.md", Path("data/sft_203m/README.md")),
        CopyItem(
            dataset_root / "reports" / "dataset_summary.json",
            Path("data/sft_203m/reports/dataset_summary.json"),
        ),
        CopyItem(
            dataset_root / "reports" / "sources_manifest.json",
            Path("data/sft_203m/reports/sources_manifest.json"),
        ),
        CopyItem(
            dataset_root / "reports" / "validation_report.json",
            Path("data/sft_203m/reports/validation_report.json"),
        ),
        CopyItem(
            project_root / "configs" / "sft_203m_cloud_safe.yaml",
            Path("configs/sft_203m_cloud_safe.yaml"),
        ),
        CopyItem(
            project_root / "configs" / "sft_203m_cloud_high_memory.yaml",
            Path("configs/sft_203m_cloud_high_memory.yaml"),
        ),
        CopyItem(project_root / "cloud_sft" / "README_CLOUD_SFT.md", Path("README_CLOUD_SFT.md")),
        CopyItem(project_root / "cloud_sft" / "run_sft.sh", Path("run_sft.sh")),
        CopyItem(project_root / "cloud_sft" / "verify_bundle.py", Path("verify_bundle.py")),
    ]


def copy_required_files(items: Iterable[CopyItem], bundle_root: Path) -> None:
    for item in items:
        source = require_file(item.source)
        destination = bundle_root / item.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def copy_training_source(project_root: Path, bundle_root: Path) -> None:
    source_root = project_root / "source"
    destination_root = bundle_root / "source"
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    )
    shutil.copytree(source_root, destination_root, ignore=ignore)


def copy_dataset_tokenizer(bundle_root: Path) -> None:
    destination = bundle_root / "data/sft_203m/assets" / TOKENIZER_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_root / "tokenizer" / TOKENIZER_NAME, destination)


def iter_manifest_files(bundle_root: Path) -> list[Path]:
    return sorted(
        (path for path in bundle_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.relative_to(bundle_root).as_posix(),
    )


def write_manifests(bundle_root: Path) -> None:
    files = iter_manifest_files(bundle_root)
    entries = []
    checksum_lines = []
    for path in files:
        relative = path.relative_to(bundle_root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": digest})
        checksum_lines.append(f"{digest}  {relative}")

    metadata = {
        "bundle": bundle_root.name,
        "format_version": 1,
        "files": entries,
        "totals": {
            "files": len(entries),
            "bytes_excluding_manifests": sum(item["bytes"] for item in entries),
        },
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = bundle_root / "SHA256SUMS"
    manifest_files = iter_manifest_files(bundle_root)
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(bundle_root).as_posix()}"
        for path in manifest_files
    ]
    manifest_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def zip_compression(path: Path) -> int:
    if path.suffix.lower() in {".pt", ".model", ".zip", ".gz", ".png", ".jpg", ".jpeg"}:
        return zipfile.ZIP_STORED
    return zipfile.ZIP_DEFLATED


def create_archive(bundle_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", allowZip64=True, compresslevel=6) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if not path.is_file():
                continue
            arcname = (Path(bundle_root.name) / path.relative_to(bundle_root)).as_posix()
            archive.write(path, arcname, compress_type=zip_compression(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable Murmur 203M cloud SFT bundle.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    return parser.parse_args()


def validate_targets(project_root: Path, dataset_root: Path, bundle_root: Path) -> None:
    resolved_bundle = bundle_root.resolve()
    if resolved_bundle in {project_root.resolve(), dataset_root.resolve()}:
        raise ValueError("Bundle root cannot replace the project root or dataset root.")
    if bundle_root.exists():
        raise FileExistsError(f"Bundle root already exists: {bundle_root}")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve()
    model_path = require_file(args.model)
    bundle_root = args.bundle_root.resolve()
    archive_path = args.archive.resolve()

    validate_targets(project_root, dataset_root, bundle_root)
    bundle_root.mkdir(parents=True)
    copy_required_files(source_items(project_root, dataset_root, model_path), bundle_root)
    copy_training_source(project_root, bundle_root)
    copy_dataset_tokenizer(bundle_root)
    write_manifests(bundle_root)
    create_archive(bundle_root, archive_path)

    print(f"Bundle directory: {bundle_root}")
    print(f"Archive: {archive_path}")
    print(f"Archive SHA256: {sha256_file(archive_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
