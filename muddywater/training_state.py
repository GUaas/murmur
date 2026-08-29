from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

import numpy as np
import torch


SCHEDULE_STATE_VERSION = 1


@dataclass(frozen=True)
class ScheduleState:
    """Immutable schedule horizon persisted across full resumes.

    Training limits may be extended, but recomputing a cosine-style schedule
    against the larger limit would increase learning rate, Muon momentum, and
    weight decay at the resume boundary.  A full resume therefore keeps the
    horizon that was active when the optimizer state was created.
    """

    axis: str
    total_steps: int
    warmup_steps: int
    total_tokens: int | None = None
    warmup_tokens: int | None = None
    version: int = SCHEDULE_STATE_VERSION
    extension_policy: str = "preserve_original_horizon"

    def __post_init__(self) -> None:
        axis = str(self.axis).strip().lower()
        if axis not in {"steps", "tokens"}:
            raise ValueError("schedule axis must be 'steps' or 'tokens'")
        if int(self.total_steps) <= 0:
            raise ValueError("schedule total_steps must be positive")
        if int(self.warmup_steps) < 0:
            raise ValueError("schedule warmup_steps cannot be negative")
        if axis == "tokens" and (self.total_tokens is None or int(self.total_tokens) <= 0):
            raise ValueError("token-based schedules require a positive total_tokens horizon")
        if self.warmup_tokens is not None and int(self.warmup_tokens) < 0:
            raise ValueError("schedule warmup_tokens cannot be negative")
        if self.extension_policy != "preserve_original_horizon":
            raise ValueError(
                "Unsupported schedule extension policy: "
                f"{self.extension_policy!r}"
            )
        object.__setattr__(self, "axis", axis)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "axis": self.axis,
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
            "total_tokens": None if self.total_tokens is None else int(self.total_tokens),
            "warmup_tokens": (
                None if self.warmup_tokens is None else int(self.warmup_tokens)
            ),
            "extension_policy": self.extension_policy,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScheduleState":
        version = int(value.get("version", SCHEDULE_STATE_VERSION))
        if version != SCHEDULE_STATE_VERSION:
            raise ValueError(
                f"Unsupported schedule state version {version}; "
                f"expected {SCHEDULE_STATE_VERSION}"
            )
        return cls(
            version=version,
            axis=str(value.get("axis", "")),
            total_steps=int(value.get("total_steps", 0) or 0),
            warmup_steps=int(value.get("warmup_steps", 0) or 0),
            total_tokens=(
                None
                if value.get("total_tokens") is None
                else int(value["total_tokens"])
            ),
            warmup_tokens=(
                None
                if value.get("warmup_tokens") is None
                else int(value["warmup_tokens"])
            ),
            extension_policy=str(
                value.get("extension_policy", "preserve_original_horizon")
            ),
        )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
