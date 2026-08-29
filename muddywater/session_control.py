from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SESSION_DEADLINE_ENV = "MURMUR_SESSION_DEADLINE_UNIX"
TORCHINDUCTOR_CACHE_DIR_ENV = "TORCHINDUCTOR_CACHE_DIR"
TORCHINDUCTOR_FX_GRAPH_CACHE_ENV = "TORCHINDUCTOR_FX_GRAPH_CACHE"
TORCHINDUCTOR_AUTOGRAD_CACHE_ENV = "TORCHINDUCTOR_AUTOGRAD_CACHE"
DEFAULT_COMPILE_CACHE_DIRNAME = ".torch_compile_cache"


def configure_persistent_compile_cache(
    *,
    project_root: str | os.PathLike[str],
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a child environment with a reusable on-disk torch.compile cache."""

    child_environment = dict(os.environ if environ is None else environ)
    default_cache_dir = (
        Path(project_root).expanduser().resolve() / DEFAULT_COMPILE_CACHE_DIRNAME
    )
    cache_dir = Path(
        child_environment.setdefault(
            TORCHINDUCTOR_CACHE_DIR_ENV,
            str(default_cache_dir),
        )
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    child_environment.setdefault(TORCHINDUCTOR_FX_GRAPH_CACHE_ENV, "1")
    child_environment.setdefault(TORCHINDUCTOR_AUTOGRAD_CACHE_ENV, "1")
    return child_environment


def session_deadline_from_environment(
    environ: Mapping[str, str] | None = None,
) -> float | None:
    """Read the optional end-to-end session deadline supplied by the launcher."""

    source = os.environ if environ is None else environ
    raw_value = source.get(SESSION_DEADLINE_ENV)
    if raw_value in {None, ""}:
        return None
    try:
        deadline = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{SESSION_DEADLINE_ENV} must be a finite positive Unix timestamp"
        ) from exc
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError(
            f"{SESSION_DEADLINE_ENV} must be a finite positive Unix timestamp"
        )
    return deadline


def build_session_deadline(*, started_at: float, max_seconds: float) -> float:
    """Return the absolute deadline for one launcher-bounded GPU session."""

    started_at = float(started_at)
    max_seconds = float(max_seconds)
    if not math.isfinite(started_at) or started_at <= 0:
        raise ValueError("started_at must be a finite positive Unix timestamp")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    return started_at + max_seconds


@dataclass(frozen=True)
class TrainingSessionBudget:
    """Wall-clock budget for one resumable training session."""

    started_at: float
    max_seconds: float | None = None
    launcher_deadline_at: float | None = None

    @classmethod
    def from_training_config(
        cls,
        training_config: Mapping[str, Any],
        *,
        started_at: float,
        launcher_deadline_at: float | None = None,
    ) -> "TrainingSessionBudget":
        value = training_config.get("session_max_seconds")
        max_seconds = None if value in {None, ""} else float(value)
        return cls(
            started_at=float(started_at),
            max_seconds=max_seconds,
            launcher_deadline_at=launcher_deadline_at,
        )

    @property
    def enabled(self) -> bool:
        return self.deadline_at is not None

    @property
    def deadline_at(self) -> float | None:
        trainer_deadline = (
            None
            if self.max_seconds is None
            else self.started_at + float(self.max_seconds)
        )
        if trainer_deadline is None:
            return self.launcher_deadline_at
        if self.launcher_deadline_at is None:
            return trainer_deadline
        return min(trainer_deadline, float(self.launcher_deadline_at))

    def elapsed_seconds(self, now: float) -> float:
        return max(0.0, float(now) - self.started_at)

    def remaining_seconds(self, now: float) -> float | None:
        deadline = self.deadline_at
        if deadline is None:
            return None
        return max(0.0, deadline - float(now))

    def should_pause(self, now: float) -> bool:
        deadline = self.deadline_at
        return deadline is not None and float(now) >= deadline
