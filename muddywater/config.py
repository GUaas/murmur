from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .paths import resolve_path
from .utils import atomic_write_text


def _strip_internal_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_internal_keys(item)
            for key, item in value.items()
            if not str(key).startswith("__")
        }
    if isinstance(value, list):
        return [_strip_internal_keys(item) for item in value]
    return deepcopy(value)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a Python dictionary."""
    resolved = resolve_path(path)
    config_path = Path(path) if resolved is None else resolved
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def save_config(
    config: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = True,
) -> None:
    """Save a config dictionary to YAML."""
    output_path = Path(path)
    content = yaml.safe_dump(
        _strip_internal_keys(config),
        allow_unicode=True,
        sort_keys=False,
    )
    atomic_write_text(output_path, content, overwrite=overwrite)


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating either input."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_by_path(config: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Read a nested value using a dotted path such as 'training.batch_size'."""
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value
