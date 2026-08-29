from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TEXT_SIMPLIFICATION_MODEL = Path(
    "outputs/sft_203m_text_simplification_pass_filtered/"
    "murmur_203m_text_simplification_best_weights_only.pt"
)
MODEL_WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify an extracted full project bundle.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_checksums(path: Path) -> dict[Path, str]:
    entries: dict[Path, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, separator, relative = raw_line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(f"Invalid SHA256SUMS line: {raw_line!r}")
        entries[Path(relative)] = digest.lower()
    return entries


def verify(root: Path, *, skip_hashes: bool) -> dict[str, object]:
    root = root.resolve()
    checksums = read_checksums(root / "SHA256SUMS")
    actual_files = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual_files != set(checksums):
        raise RuntimeError(
            "File inventory mismatch: "
            f"missing={sorted(map(str, set(checksums) - actual_files))}, "
            f"unexpected={sorted(map(str, actual_files - set(checksums)))}"
        )
    if not skip_hashes:
        for relative, expected in checksums.items():
            if file_sha256(root / relative) != expected:
                raise RuntimeError(f"SHA-256 mismatch: {relative}")

    weights = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_WEIGHT_SUFFIXES
    )
    if weights != [TEXT_SIMPLIFICATION_MODEL]:
        raise RuntimeError(f"Unexpected model weights in bundle: {weights}")

    metadata = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    if metadata.get("base_pretrained_model_weights_included") is not False:
        raise RuntimeError("Bundle metadata does not confirm pretrained-weight exclusion")
    return {
        "status": "passed",
        "root": str(root),
        "verified_files": len(checksums),
        "hashes_checked": not skip_hashes,
        "model_weights": [path.as_posix() for path in weights],
        "base_pretrained_model_weights_included": False,
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(verify(args.root, skip_hashes=args.skip_hashes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
