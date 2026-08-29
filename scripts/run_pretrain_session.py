from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config, save_config
from muddywater.config_validation import validate_pretrain_config
from muddywater.paths import resolve_output_path, resolve_path
from muddywater.session_control import (
    SESSION_DEADLINE_ENV,
    TORCHINDUCTOR_CACHE_DIR_ENV,
    build_session_deadline,
    configure_persistent_compile_cache,
)
from muddywater.utils import find_training_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one wall-clock-bounded pretraining session with automatic resume."
    )
    parser.add_argument("--config", required=True, help="Base pretraining YAML config.")
    return parser.parse_args()


def prepare_session_config(
    config: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    session_config = deepcopy(config)
    training = dict(session_config.get("training", {}))
    latest_path = output_dir / "latest.pt"
    final_path = output_dir / "final.pt"

    if final_path.is_file():
        return session_config, "completed"
    if latest_path.is_file():
        training["resume_from"] = str(latest_path.resolve())
        session_config["training"] = training
        return session_config, "resume"

    existing = find_training_artifacts(output_dir)
    if existing:
        names = ", ".join(path.name for path in existing[:8])
        raise RuntimeError(
            f"Output directory has training artifacts but no resumable latest.pt: "
            f"{output_dir} ({names})"
        )
    training.pop("resume_from", None)
    session_config["training"] = training
    return session_config, "fresh"


def run_session(config_path: Path) -> int:
    session_started_at = time.time()
    config = load_config(config_path)
    validate_pretrain_config(config)
    training = dict(config.get("training", {}))
    output_dir = resolve_output_path(
        training.get("output_dir", "outputs/run"),
        base=ROOT,
    )
    session_config, mode = prepare_session_config(config, output_dir=output_dir)

    if mode == "completed":
        print(f"Training is already complete: {output_dir / 'final.pt'}", flush=True)
        return 0

    session_limit = session_config.get("training", {}).get("session_max_seconds")
    if session_limit is None:
        raise ValueError(
            "training.session_max_seconds is required for a bounded GPU session"
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".session.yaml",
            prefix=f".{config_path.stem}.",
            dir=config_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        save_config(session_config, temp_path)
        print(
            f"Starting {mode} session with a {float(session_limit):.0f}s wall-clock limit.",
            flush=True,
        )
        child_environment = configure_persistent_compile_cache(
            project_root=ROOT.parent,
            environ=os.environ,
        )
        print(
            "Persistent torch.compile cache: "
            f"{child_environment[TORCHINDUCTOR_CACHE_DIR_ENV]}",
            flush=True,
        )
        child_environment[SESSION_DEADLINE_ENV] = str(
            build_session_deadline(
                started_at=session_started_at,
                max_seconds=float(session_limit),
            )
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "pretrain.py"),
                "--config",
                str(temp_path),
            ],
            cwd=ROOT.parent,
            check=False,
            env=child_environment,
        )
        return int(completed.returncode)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    resolved = resolve_path(args.config)
    if resolved is None:
        raise FileNotFoundError(args.config)
    raise SystemExit(run_session(Path(resolved)))


if __name__ == "__main__":
    main()
