from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED_FILES = {"SHA256SUMS", "MANIFEST.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in EXCLUDED_FILES
    )


def write_manifest(root: Path, role: str) -> None:
    files = payload_files(root)
    weights = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
        }
        for path in files
        if path.suffix == ".pt"
    ]
    manifest = {
        "format": "murmur_portable_split_v1",
        "package_role": role,
        "payload_files": len(files),
        "payload_bytes": sum(path.stat().st_size for path in files),
        "weights": weights,
        "training_snapshots_included": False,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in files
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(root: Path, role: str) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    write_manifest(root, role)
    write_checksums(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest for a split portable package.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--role", required=True)
    args = parser.parse_args()
    build(args.root, args.role)


if __name__ == "__main__":
    main()
