from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any


GLOB_CHARS = set("*?[")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _has_glob(path: Path) -> bool:
    return any(char in str(path) for char in GLOB_CHARS)


def resolve_path(path: str | Path | None, base: str | Path | None = None) -> Path | None:
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    bases = []
    if base is not None:
        bases.append(Path(base))
    bases.extend([Path.cwd(), project_root()])
    for raw_base in bases:
        base_path = raw_base if raw_base.is_dir() else raw_base.parent
        resolved = base_path / candidate
        if resolved.exists() or (_has_glob(resolved) and bool(glob(str(resolved)))):
            return resolved.resolve()
    fallback_base = Path(base) if base is not None else project_root()
    if not fallback_base.is_dir():
        fallback_base = fallback_base.parent
    return (fallback_base / candidate).resolve()


def resolve_config_path(path: str | Path, config_path: str | Path | None = None) -> Path:
    base = Path(config_path).parent if config_path is not None else project_root()
    resolved = resolve_path(path, base=base)
    assert resolved is not None
    return resolved


def resolve_output_path(path: str | Path, base: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    base_path = Path(base) if base is not None else project_root()
    if not base_path.is_dir():
        base_path = base_path.parent
    return (base_path / candidate).resolve()


def resolve_config_paths_in_data(data_config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    resolved = dict(data_config)
    for key in ("train_paths", "val_paths"):
        value = resolved.get(key)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        resolved[key] = [str(resolve_config_path(item, config_path=config_path)) for item in values]
    for key in ("token_cache_dir",):
        value = resolved.get(key)
        if value:
            resolved[key] = str(resolve_config_path(value, config_path=config_path))
    return resolved
