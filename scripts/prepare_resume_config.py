#!/usr/bin/env python3
"""Create a strict full-resume config from a first-session SFT config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


def build_resume_config(config: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("Config must contain a training mapping")
    training.pop("init_from", None)
    training["resume_from"] = checkpoint
    return config


def save_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def main() -> None:
    args = parse_args()
    config = build_resume_config(load_config(args.base), args.checkpoint)
    save_config(args.output, config)
    print(f"Resume config ready: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
