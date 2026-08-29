from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DEFAULT_DIR_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "logs",
    "tmp",
]
DEFAULT_FILE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pt",
    "*.pth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove local training/test artifacts.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--apply", action="store_true", help="Actually delete matched artifacts.")
    parser.add_argument("--include-data", action="store_true", help="Also remove data/ under root.")
    return parser.parse_args()


def is_inside_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def collect_artifacts(root: Path, include_data: bool = False) -> list[Path]:
    patterns = list(DEFAULT_DIR_PATTERNS)
    if include_data:
        patterns.append("data")
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(path for path in root.rglob(pattern) if path.exists())
    for pattern in DEFAULT_FILE_PATTERNS:
        matches.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(matches, key=lambda path: (len(path.parts), str(path)), reverse=True)


def remove_artifact(path: Path, root: Path) -> None:
    resolved = path.resolve()
    if not is_inside_root(resolved, root):
        raise ValueError(f"Refusing to delete outside root: {resolved}")
    if resolved == root:
        raise ValueError(f"Refusing to delete root: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")
    artifacts = collect_artifacts(root, include_data=bool(args.include_data))
    for path in artifacts:
        print(path)
    if not args.apply:
        print(f"dry_run=true matched={len(artifacts)}")
        return
    for path in artifacts:
        remove_artifact(path, root)
    print(f"deleted={len(artifacts)}")


if __name__ == "__main__":
    main()
