from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from .utils import atomic_write_text, file_sha256


RUN_IDENTITY_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    identity: dict[str, Any] = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        identity["sha256"] = file_sha256(resolved)
    return identity


def _configured_resource_paths(value: Any) -> Iterable[Path]:
    """Yield existing files/directories mentioned by the data config."""

    if isinstance(value, Mapping):
        for item in value.values():
            yield from _configured_resource_paths(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _configured_resource_paths(item)
        return
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return
    try:
        path = Path(value).expanduser()
        if path.exists():
            yield path
    except (OSError, TypeError, ValueError):
        return


def _dataset_token_paths(dataset: Any) -> list[Path]:
    token_path = getattr(dataset, "token_path", None)
    if token_path is not None:
        return [Path(token_path)]
    shards = getattr(dataset, "shards", None)
    if isinstance(shards, (list, tuple)):
        paths: list[Path] = []
        for shard in shards:
            paths.extend(_dataset_token_paths(shard))
        return paths
    return []


def _sample_content_digest(dataset: Any) -> str | None:
    """Hash in-memory dataset samples without materializing another full copy."""

    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, (list, tuple)):
        return None
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            json.dumps(
                _jsonable(sample),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def tokenizer_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer_config = config.get("tokenizer", {})
    if not isinstance(tokenizer_config, Mapping):
        tokenizer_config = {}
    path_value = tokenizer_config.get("path")
    path = Path(path_value).expanduser() if path_value else None
    configured_sha = tokenizer_config.get("sha256") or tokenizer_config.get("tokenizer_sha256")
    actual_sha: str | None = None
    file_identity: dict[str, Any] | None = None
    if path is not None and path.exists() and path.is_file():
        actual_sha = file_sha256(path)
        file_identity = _file_identity(path)
    sha = str(configured_sha or actual_sha or "").strip().lower() or None
    if configured_sha and actual_sha and str(configured_sha).strip().lower() != actual_sha.lower():
        raise ValueError(
            "Configured tokenizer SHA256 does not match the tokenizer file: "
            f"configured={configured_sha}, actual={actual_sha}, path={path}"
        )
    return {
        "sha256": sha,
        "file": file_identity,
    }


def training_data_identity(config: Mapping[str, Any], dataset: Any) -> dict[str, Any]:
    """Build a stable, cheap identity for the exact training dataset.

    Token caches are identified by shard stat metadata plus cryptographic hashes
    of their small metadata/manifests.  In-memory datasets are hashed sample by
    sample.  Dataset length and sampling geometry are included in both cases.
    """

    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        data_config = {}
    dataset_length = len(dataset)
    dataset_metadata = {
        "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "length": int(dataset_length),
        "max_seq_len": getattr(dataset, "max_seq_len", None),
        "stride": getattr(dataset, "stride", None),
        "window_size": getattr(dataset, "window_size", None),
        "sample_content_sha256": _sample_content_digest(dataset),
    }

    resources: dict[str, dict[str, Any]] = {}
    configured_paths = list(_configured_resource_paths(data_config))
    token_paths = _dataset_token_paths(dataset)
    for path in [*configured_paths, *token_paths]:
        resolved = path.resolve()
        if resolved.is_dir():
            manifest_path = resolved / "manifest.json"
            if manifest_path.exists():
                resources[str(manifest_path.resolve())] = _file_identity(
                    manifest_path,
                    content_hash=True,
                )
            continue
        if not resolved.is_file():
            continue
        is_small_metadata = (
            resolved.name == "manifest.json"
            or resolved.name.endswith(".meta.json")
            or resolved.stat().st_size <= 1024 * 1024
        )
        resources[str(resolved)] = _file_identity(
            resolved,
            content_hash=is_small_metadata,
        )
        if resolved in [candidate.resolve() for candidate in token_paths]:
            meta_path = resolved.with_suffix(resolved.suffix + ".meta.json")
            if meta_path.exists():
                resources[str(meta_path.resolve())] = _file_identity(
                    meta_path,
                    content_hash=True,
                )
            doc_starts_path = resolved.with_suffix(
                resolved.suffix + ".doc_starts.npy"
            )
            if doc_starts_path.exists():
                resources[str(doc_starts_path.resolve())] = _file_identity(
                    doc_starts_path,
                    content_hash=doc_starts_path.stat().st_size <= 16 * 1024 * 1024,
                )

    payload = {
        "data_config": _jsonable(data_config),
        "dataset": _jsonable(dataset_metadata),
        "resources": [resources[key] for key in sorted(resources)],
    }
    return {
        **payload,
        "fingerprint": _stable_digest(payload),
    }


def build_training_identity(config: Mapping[str, Any], dataset: Any) -> dict[str, Any]:
    tokenizer = tokenizer_identity(config)
    data = training_data_identity(config, dataset)
    payload = {
        "version": RUN_IDENTITY_VERSION,
        "tokenizer": tokenizer,
        "data": data,
    }
    return {
        **payload,
        "fingerprint": _stable_digest(payload),
    }


def _git_value(args: list[str], cwd: str | Path | None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def git_summary(cwd: str | Path | None = None) -> dict[str, Any]:
    return {
        "commit": _git_value(["rev-parse", "HEAD"], cwd=cwd),
        "branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd),
        "is_dirty": None
        if _git_value(["rev-parse", "--is-inside-work-tree"], cwd=cwd) is None
        else bool(_git_value(["status", "--porcelain"], cwd=cwd)),
    }


def cache_manifest_summary(config: dict[str, Any]) -> dict[str, Any] | None:
    data_config = config.get("data", {})
    cache_dir = data_config.get("token_cache_dir")
    if not cache_dir:
        return None
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        return {"path": str(manifest_path), "exists": False}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(manifest_path), "exists": True, "error": str(exc)}
    return {
        "path": str(manifest_path),
        "exists": True,
        "sha256": file_sha256(manifest_path),
        "tokenizer_sha256": payload.get("tokenizer_sha256"),
        "train_tokens": payload.get("train_tokens"),
        "val_tokens": payload.get("val_tokens"),
        "train_documents": payload.get("train_documents"),
        "val_documents": payload.get("val_documents"),
        "actual_val_token_ratio": payload.get("actual_val_token_ratio"),
        "actual_val_document_ratio": payload.get("actual_val_document_ratio"),
        "train_shards": len(payload.get("train_shards", []) or []),
        "val_shards": len(payload.get("val_shards", []) or []),
    }


def runtime_summary() -> dict[str, Any]:
    cuda = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        cuda.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
            }
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": cuda,
    }


def build_initial_manifest(
    config: dict[str, Any],
    diagnostics_path: str | Path | None,
    cwd: str | Path | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": list(command or sys.argv),
        "git": git_summary(cwd=cwd),
        "runtime": runtime_summary(),
        "config": config,
        "tokenizer": config.get("tokenizer", {}),
        "cache_manifest": cache_manifest_summary(config),
        "diagnostics_path": str(diagnostics_path) if diagnostics_path else None,
        "status": "started",
    }


def write_run_manifest(
    path: str | Path,
    manifest: dict[str, Any],
) -> Path:
    output_path = Path(path)
    atomic_write_text(
        output_path,
        json.dumps(manifest, ensure_ascii=False, indent=2),
        overwrite=True,
    )
    return output_path


def write_initial_run_manifest(
    output_dir: str | Path,
    config: dict[str, Any],
    diagnostics_path: str | Path | None,
    cwd: str | Path | None = None,
    command: list[str] | None = None,
) -> Path:
    path = Path(output_dir) / "run_manifest.json"
    manifest = build_initial_manifest(
        config=config,
        diagnostics_path=diagnostics_path,
        cwd=cwd,
        command=command,
    )
    return write_run_manifest(path, manifest)


def update_run_manifest(
    path: str | Path,
    updates: dict[str, Any],
    status: str | None = None,
) -> Path:
    manifest_path = Path(path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest.update(updates)
    if status is not None:
        manifest["status"] = status
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return write_run_manifest(manifest_path, manifest)
