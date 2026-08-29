from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "murmur-203m-text-simplification-project-longtext-v2-no-pretrained-model-20260814"
TEXT_SIMPLIFICATION_MODEL = Path(
    "outputs/sft_203m_text_simplification_pass_filtered/"
    "murmur_203m_text_simplification_best_weights_only.pt"
)
MODEL_WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


@dataclass(frozen=True)
class IncludedFile:
    path: Path
    relative: Path
    size: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the complete text-simplification project without base pretrained weights."
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT.parent)
    return parser.parse_args()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def exclusion_reason(relative: Path) -> str | None:
    parts = tuple(part.lower() for part in relative.parts)
    if relative == Path("SHA256SUMS"):
        return "superseded_bundle_metadata"
    if len(parts) >= 2 and parts[:2] == ("output", "release"):
        return "generated_release_copy"
    if len(parts) >= 2 and parts[:2] == ("output", "archive_validation"):
        return "archive_validation_copy"
    if parts and parts[0] == "tmp":
        return "temporary_artifact"
    if any(part in {"__pycache__", ".pytest_cache"} for part in parts):
        return "python_cache"
    if relative.suffix.lower() in {".pyc", ".pyo"}:
        return "python_cache"
    if relative.suffix.lower() in MODEL_WEIGHT_SUFFIXES and relative != TEXT_SIMPLIFICATION_MODEL:
        return "non_text_simplification_model_weight"
    return None


def collect_files() -> tuple[list[IncludedFile], dict[str, list[str]]]:
    included: list[IncludedFile] = []
    excluded: dict[str, list[str]] = {}
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        reason = exclusion_reason(relative)
        if reason is not None:
            excluded.setdefault(reason, []).append(relative.as_posix())
            continue
        included.append(
            IncludedFile(
                path=path,
                relative=relative,
                size=path.stat().st_size,
                sha256=file_sha256(path),
            )
        )
    return included, excluded


def validate_scope(files: list[IncludedFile]) -> None:
    relative_paths = {item.relative for item in files}
    required = {
        TEXT_SIMPLIFICATION_MODEL,
        Path("tokenizer/sp_unigram_32k.model"),
        Path("muddywater/text_simplification/chunking.py"),
        Path("muddywater/text_simplification/inference.py"),
        Path("muddywater/text_simplification/prompting.py"),
        Path("scripts/simplify_text.py"),
        Path("README_00_NEW_VERSION_ZH.md"),
        Path("output/pdf/murmur_203m_long_text_chunking_extreme_evaluation_20260814.pdf"),
    }
    missing = sorted(str(path) for path in required - relative_paths)
    if missing:
        raise FileNotFoundError(f"Required project files are missing: {missing}")

    weights = [item.relative for item in files if item.relative.suffix.lower() in MODEL_WEIGHT_SUFFIXES]
    if weights != [TEXT_SIMPLIFICATION_MODEL]:
        raise RuntimeError(f"Unexpected model weights selected for the bundle: {weights}")


def build_metadata(
    files: list[IncludedFile], excluded: dict[str, list[str]]
) -> dict[str, object]:
    model = next(item for item in files if item.relative == TEXT_SIMPLIFICATION_MODEL)
    return {
        "bundle": BUNDLE_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "complete_text_simplification_project",
        "included_files": len(files) + 2,
        "included_project_bytes": sum(item.size for item in files),
        "text_simplification_model": {
            "path": model.relative.as_posix(),
            "bytes": model.size,
            "sha256": model.sha256,
            "parameters": 203_037_056,
        },
        "base_pretrained_model_weights_included": False,
        "training_code_included": True,
        "training_and_validation_data_included": True,
        "evaluation_artifacts_included": True,
        "long_text_algorithm_included": True,
        "excluded_counts": {key: len(value) for key, value in sorted(excluded.items())},
        "excluded_paths": {key: value for key, value in sorted(excluded.items())},
    }


def checksum_text(files: list[IncludedFile], metadata_path: Path) -> str:
    lines = [f"{item.sha256}  {item.relative.as_posix()}" for item in files]
    lines.append(f"{file_sha256(metadata_path)}  BUNDLE_MANIFEST.json")
    return "\n".join(sorted(lines, key=lambda line: line.split("  ", 1)[1])) + "\n"


def add_file(archive: tarfile.TarFile, source: Path, relative: Path) -> None:
    archive.add(source, arcname=(Path(BUNDLE_NAME) / relative).as_posix(), recursive=False)


def build_archive(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{BUNDLE_NAME}.tar.gz"
    checksum_path = output_dir / f"{archive_path.name}.sha256"
    if archive_path.parent != output_dir or archive_path.name != f"{BUNDLE_NAME}.tar.gz":
        raise RuntimeError(f"Unexpected archive target: {archive_path}")

    print("[1/4] 正在扫描文件并计算 SHA-256……", flush=True)
    files, excluded = collect_files()
    validate_scope(files)
    metadata = build_metadata(files, excluded)

    with tempfile.TemporaryDirectory(prefix="murmur_bundle_") as temporary:
        temporary_dir = Path(temporary)
        metadata_path = temporary_dir / "BUNDLE_MANIFEST.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sums_path = temporary_dir / "SHA256SUMS"
        sums_path.write_text(checksum_text(files, metadata_path), encoding="utf-8")

        print(f"[2/4] 正在压缩 {len(files) + 2} 个文件……", flush=True)
        if archive_path.exists():
            archive_path.unlink()
        with tarfile.open(archive_path, mode="w:gz", compresslevel=1) as archive:
            for item in files:
                add_file(archive, item.path, item.relative)
            add_file(archive, metadata_path, Path("BUNDLE_MANIFEST.json"))
            add_file(archive, sums_path, Path("SHA256SUMS"))

    print("[3/4] 正在计算压缩包校验值……", flush=True)
    archive_digest = file_sha256(archive_path)
    checksum_path.write_text(
        f"{archive_digest}  {archive_path.name}\n",
        encoding="utf-8",
    )
    print("[4/4] 压缩完成。", flush=True)
    return {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_digest,
        "included_files": len(files) + 2,
        "included_model_weights": [TEXT_SIMPLIFICATION_MODEL.as_posix()],
        "base_pretrained_model_weights_included": False,
        "excluded_counts": dict(Counter({key: len(value) for key, value in excluded.items()})),
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(build_archive(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
