from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_snapshot(
    model_id: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "ModelScope is required to fetch the base checkpoint. "
            "Install it with: python -m pip install 'modelscope>=1.18,<2'"
        ) from exc

    downloaded = snapshot_download(
        model_id=model_id,
        revision=revision,
        cache_dir=str(cache_dir.resolve()) if cache_dir else None,
    )
    snapshot = Path(downloaded).resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"ModelScope snapshot directory not found: {snapshot}")
    return snapshot


def find_checkpoint(snapshot: Path, checkpoint_name: str) -> Path:
    relative_candidate = snapshot / checkpoint_name
    if relative_candidate.is_file():
        return relative_candidate

    basename_matches = sorted(
        path for path in snapshot.rglob(Path(checkpoint_name).name) if path.is_file()
    )
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        choices = ", ".join(str(path.relative_to(snapshot)) for path in basename_matches)
        raise ValueError(
            f"Multiple checkpoint files named {checkpoint_name!r} were found: {choices}"
        )

    pt_files = sorted(path for path in snapshot.rglob("*.pt") if path.is_file())
    if len(pt_files) == 1:
        return pt_files[0]
    available = ", ".join(str(path.relative_to(snapshot)) for path in pt_files[:20])
    raise FileNotFoundError(
        f"Could not identify checkpoint {checkpoint_name!r} in {snapshot}. "
        f"Available .pt files: {available or 'none'}"
    )


def install_checkpoint(
    source: Path,
    target: Path,
    *,
    expected_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, str | int]:
    source = source.resolve()
    target = target.resolve()
    source_sha256 = sha256_file(source)
    if expected_sha256 and source_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected={expected_sha256} actual={source_sha256}"
        )

    if target.exists():
        target_sha256 = sha256_file(target)
        if target_sha256 == source_sha256:
            return {
                "status": "already_present",
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": target_sha256,
            }
        if not overwrite:
            raise FileExistsError(
                f"Target checkpoint already exists with different content: {target}"
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "status": "installed",
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": source_sha256,
    }
