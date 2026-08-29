from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_NAME = "murmur-203m-text-simplification-longtext-v2-20260814"


@dataclass(frozen=True)
class CopySpec:
    source: str
    destination: str


RUNTIME_FILES = (
    "muddywater/__init__.py",
    "muddywater/attention.py",
    "muddywater/cache.py",
    "muddywater/config.py",
    "muddywater/generation.py",
    "muddywater/generation_runtime.py",
    "muddywater/losses.py",
    "muddywater/model.py",
    "muddywater/model_extras.py",
    "muddywater/optim.py",
    "muddywater/paths.py",
    "muddywater/templates.py",
    "muddywater/tokenizer.py",
    "muddywater/utils.py",
    "muddywater/text_simplification/chunking.py",
    "muddywater/text_simplification/inference.py",
    "muddywater/text_simplification/prompting.py",
    "scripts/simplify_text.py",
)


STATIC_FILES = (
    CopySpec("release_templates/README_ZH.md", "README.md"),
    CopySpec("release_templates/RELEASE_NOTES_ZH.md", "RELEASE_NOTES.md"),
    CopySpec("release_templates/requirements-inference.txt", "requirements-inference.txt"),
    CopySpec("release_templates/run_simplify.ps1", "run_simplify.ps1"),
    CopySpec("release_templates/run_simplify.sh", "run_simplify.sh"),
    CopySpec(
        "release_templates/text_simplification_init.py",
        "muddywater/text_simplification/__init__.py",
    ),
    CopySpec("configs/inference_text_simplification_release.yaml", "configs/inference.yaml"),
    CopySpec(
        "outputs/sft_203m_text_simplification_pass_filtered/"
        "murmur_203m_text_simplification_best_weights_only.pt",
        "model/murmur_203m_text_simplification_best_weights_only.pt",
    ),
    CopySpec("tokenizer/sp_unigram_32k.model", "tokenizer/sp_unigram_32k.model"),
    CopySpec("tools/verify_text_simplification_release.py", "tools/verify_release.py"),
    CopySpec(
        "output/pdf/murmur_203m_long_text_chunking_extreme_evaluation_20260814.pdf",
        "docs/evaluation-report.pdf",
    ),
    CopySpec(
        "output/long_text_chunking_evaluation_20260814/report.md",
        "docs/evaluation-report.md",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the inference-only release bundle.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "release",
    )
    parser.add_argument("--archive", action="store_true")
    return parser.parse_args()


def ensure_clean_target(output_dir: Path, release_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_release = release_dir.resolve()
    if resolved_release.parent != resolved_output or resolved_release.name != RELEASE_NAME:
        raise RuntimeError(f"Refusing to replace unexpected directory: {resolved_release}")
    if resolved_release.exists():
        shutil.rmtree(resolved_release)
    resolved_release.mkdir(parents=True)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_release_files(release_dir: Path) -> None:
    for relative in RUNTIME_FILES:
        copy_file(PROJECT_ROOT / relative, release_dir / relative)
    for spec in STATIC_FILES:
        copy_file(PROJECT_ROOT / spec.source, release_dir / spec.destination)
    charts_source = PROJECT_ROOT / "output/long_text_chunking_evaluation_20260814/charts"
    for chart in sorted(charts_source.glob("*.png")):
        copy_file(chart, release_dir / "docs/charts" / chart.name)


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(release_dir: Path) -> None:
    model_path = release_dir / "model/murmur_203m_text_simplification_best_weights_only.pt"
    metadata = {
        "release": RELEASE_NAME,
        "release_date": "2026-08-14",
        "purpose": "text_simplification_inference",
        "model_parameters": 203_037_056,
        "model_bytes": model_path.stat().st_size,
        "model_sha256": file_sha256(model_path),
        "default_long_text_chunk_tokens": 160,
        "contains_pretraining_assets": False,
        "contains_training_or_validation_data": False,
        "contains_optimizer_state": False,
    }
    (release_dir / "RELEASE_MANIFEST.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def audit_release_scope(release_dir: Path) -> None:
    forbidden_fragments = (
        "pretrain",
        "train.jsonl",
        "validation.jsonl",
        "optimizer",
        "train.log",
        "training_summary",
        "__pycache__",
        ".pyc",
    )
    paths = [path.relative_to(release_dir).as_posix().lower() for path in release_dir.rglob("*")]
    violations = [
        path for path in paths if any(fragment in path for fragment in forbidden_fragments)
    ]
    if violations:
        raise RuntimeError(f"Forbidden release paths found: {violations}")
    checkpoints = list(release_dir.rglob("*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected exactly one checkpoint, found {len(checkpoints)}")


def write_checksums(release_dir: Path) -> int:
    checksum_path = release_dir / "SHA256SUMS"
    files = sorted(
        path for path in release_dir.rglob("*") if path.is_file() and path != checksum_path
    )
    lines = [f"{file_sha256(path)}  {path.relative_to(release_dir).as_posix()}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def build_archive(output_dir: Path, release_dir: Path) -> tuple[Path, str]:
    archive_path = output_dir / f"{RELEASE_NAME}.tar.gz"
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, mode="w:gz", compresslevel=1) as archive:
        archive.add(release_dir, arcname=RELEASE_NAME)
    archive_digest = file_sha256(archive_path)
    (output_dir / f"{archive_path.name}.sha256").write_text(
        f"{archive_digest}  {archive_path.name}\n",
        encoding="utf-8",
    )
    return archive_path, archive_digest


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    release_dir = output_dir / RELEASE_NAME
    ensure_clean_target(output_dir, release_dir)
    copy_release_files(release_dir)
    write_metadata(release_dir)
    audit_release_scope(release_dir)
    file_count = write_checksums(release_dir)

    result: dict[str, object] = {
        "release_dir": str(release_dir),
        "files": file_count,
        "archive": None,
    }
    if args.archive:
        archive_path, archive_digest = build_archive(output_dir, release_dir)
        result["archive"] = str(archive_path)
        result["archive_sha256"] = archive_digest
        result["archive_bytes"] = archive_path.stat().st_size
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
