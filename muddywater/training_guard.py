from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LossGuardConfig:
    """Configuration for the deliberately conservative early loss guard."""

    enabled: bool = False
    min_steps: int = 500
    baseline_window: int = 50
    recent_window: int = 50
    min_train_ce_drop: float | None = 0.10
    max_evals_without_improvement: int = 3
    min_val_improvement: float | None = 0.01

    @classmethod
    def from_value(cls, value: Any) -> "LossGuardConfig":
        if value is None:
            return cls()
        if isinstance(value, bool):
            return cls(enabled=value)
        if not isinstance(value, Mapping):
            raise TypeError("training.loss_guard must be a mapping or boolean")

        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise ValueError(
                "Unknown training.loss_guard option(s): " + ", ".join(unknown)
            )

        config = cls(
            enabled=bool(value.get("enabled", False)),
            min_steps=int(value.get("min_steps", 500)),
            baseline_window=int(value.get("baseline_window", 50)),
            recent_window=int(value.get("recent_window", 50)),
            min_train_ce_drop=_optional_nonnegative_float(
                value.get("min_train_ce_drop", 0.10),
                "min_train_ce_drop",
            ),
            max_evals_without_improvement=int(
                value.get("max_evals_without_improvement", 3)
            ),
            min_val_improvement=_optional_nonnegative_float(
                value.get("min_val_improvement", 0.01),
                "min_val_improvement",
            ),
        )
        if config.min_steps < 0:
            raise ValueError("training.loss_guard.min_steps cannot be negative")
        if config.baseline_window <= 0:
            raise ValueError("training.loss_guard.baseline_window must be positive")
        if config.recent_window <= 0:
            raise ValueError("training.loss_guard.recent_window must be positive")
        if config.max_evals_without_improvement <= 0:
            raise ValueError(
                "training.loss_guard.max_evals_without_improvement must be positive"
            )
        if (
            config.enabled
            and config.min_train_ce_drop is None
            and config.min_val_improvement is None
        ):
            raise ValueError(
                "training.loss_guard enabled but both train and validation checks are disabled"
            )
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_nonnegative_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"training.loss_guard.{name} must be finite and non-negative")
    return parsed


@dataclass(frozen=True)
class LossGuardDecision:
    source: str
    checked: bool = False
    passed: bool = False
    failed: bool = False
    message: str | None = None


class LossDescentGuard:
    """Track independent training and validation evidence of early descent.

    A single noisy observation never aborts training.  Each enabled signal gets
    ``max_evals_without_improvement`` eligible, spaced checks.  Once a signal
    demonstrates the configured drop it is permanently satisfied for the run.
    """

    STATE_VERSION = 1

    def __init__(self, config: LossGuardConfig) -> None:
        self.config = config
        self._baseline_train_values: list[float] = []
        self._recent_train_values: deque[float] = deque(
            maxlen=config.recent_window
        )
        self.baseline_train_ce: float | None = None
        self.latest_train_ce: float | None = None
        self.latest_train_drop: float | None = None
        self.last_train_check_step: int | None = None
        self.train_checks = 0
        self.train_failures = 0
        self.train_passed = config.min_train_ce_drop is None

        self.baseline_val_ce: float | None = None
        self.baseline_val_step: int | None = None
        self.latest_val_ce: float | None = None
        self.latest_val_improvement: float | None = None
        self.last_val_step: int | None = None
        self.val_checks = 0
        self.val_failures = 0
        self.val_passed = config.min_val_improvement is None

        self.failure_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def needs_validation_baseline(self) -> bool:
        return (
            self.enabled
            and self.config.min_val_improvement is not None
            and self.baseline_val_ce is None
        )

    def observe_train(self, step: int, ce_loss: float) -> LossGuardDecision:
        if not self.enabled or self.config.min_train_ce_drop is None:
            return LossGuardDecision(source="train")
        loss = _finite_loss(ce_loss, "training")
        step = int(step)

        if self.baseline_train_ce is None:
            self._baseline_train_values.append(loss)
            if len(self._baseline_train_values) >= self.config.baseline_window:
                self.baseline_train_ce = sum(self._baseline_train_values) / len(
                    self._baseline_train_values
                )
                self._baseline_train_values.clear()
                return LossGuardDecision(
                    source="train",
                    message=(
                        "loss guard recorded training baseline "
                        f"ce={self.baseline_train_ce:.6f} at step={step}"
                    ),
                )
            return LossGuardDecision(source="train")

        self._recent_train_values.append(loss)
        if self.train_passed or step < self.config.min_steps:
            return LossGuardDecision(source="train")
        if len(self._recent_train_values) < self.config.recent_window:
            return LossGuardDecision(source="train")
        if (
            self.last_train_check_step is not None
            and step - self.last_train_check_step < self.config.recent_window
        ):
            return LossGuardDecision(source="train")

        self.last_train_check_step = step
        self.train_checks += 1
        self.latest_train_ce = sum(self._recent_train_values) / len(
            self._recent_train_values
        )
        self.latest_train_drop = self.baseline_train_ce - self.latest_train_ce
        required = float(self.config.min_train_ce_drop)
        if self.latest_train_drop >= required:
            self.train_passed = True
            return LossGuardDecision(
                source="train",
                checked=True,
                passed=True,
                message=(
                    "loss guard training check passed: "
                    f"baseline_ce={self.baseline_train_ce:.6f}, "
                    f"recent_ce={self.latest_train_ce:.6f}, "
                    f"drop={self.latest_train_drop:.6f} >= {required:.6f}"
                ),
            )

        self.train_failures += 1
        failed = self.train_failures >= self.config.max_evals_without_improvement
        message = (
            "loss guard training check did not reach the required descent: "
            f"baseline_ce={self.baseline_train_ce:.6f}, "
            f"recent_ce={self.latest_train_ce:.6f}, "
            f"drop={self.latest_train_drop:.6f} < {required:.6f}, "
            f"failed_checks={self.train_failures}/"
            f"{self.config.max_evals_without_improvement}"
        )
        if failed:
            self.failure_reason = "training_loss_did_not_descend"
        return LossGuardDecision(
            source="train",
            checked=True,
            failed=failed,
            message=message,
        )

    def observe_validation(self, step: int, ce_loss: float) -> LossGuardDecision:
        if not self.enabled or self.config.min_val_improvement is None:
            return LossGuardDecision(source="validation")
        loss = _finite_loss(ce_loss, "validation")
        step = int(step)

        if self.baseline_val_ce is None:
            self.baseline_val_ce = loss
            self.baseline_val_step = step
            self.latest_val_ce = loss
            self.last_val_step = step
            return LossGuardDecision(
                source="validation",
                message=(
                    "loss guard recorded validation baseline "
                    f"ce={loss:.6f} at step={step}"
                ),
            )
        if self.val_passed or step < self.config.min_steps:
            self.latest_val_ce = loss
            self.last_val_step = max(step, self.last_val_step or step)
            return LossGuardDecision(source="validation")
        if self.last_val_step is not None and step <= self.last_val_step:
            return LossGuardDecision(source="validation")

        self.last_val_step = step
        self.latest_val_ce = loss
        self.val_checks += 1
        self.latest_val_improvement = self.baseline_val_ce - loss
        required = float(self.config.min_val_improvement)
        if self.latest_val_improvement >= required:
            self.val_passed = True
            return LossGuardDecision(
                source="validation",
                checked=True,
                passed=True,
                message=(
                    "loss guard validation check passed: "
                    f"baseline_ce={self.baseline_val_ce:.6f}, val_ce={loss:.6f}, "
                    f"improvement={self.latest_val_improvement:.6f} >= {required:.6f}"
                ),
            )

        self.val_failures += 1
        failed = self.val_failures >= self.config.max_evals_without_improvement
        message = (
            "loss guard validation check did not reach the required improvement: "
            f"baseline_ce={self.baseline_val_ce:.6f}, val_ce={loss:.6f}, "
            f"improvement={self.latest_val_improvement:.6f} < {required:.6f}, "
            f"failed_evals={self.val_failures}/"
            f"{self.config.max_evals_without_improvement}"
        )
        if failed:
            self.failure_reason = "validation_loss_did_not_improve"
        return LossGuardDecision(
            source="validation",
            checked=True,
            failed=failed,
            message=message,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "baseline_train_values": list(self._baseline_train_values),
            "recent_train_values": list(self._recent_train_values),
            "baseline_train_ce": self.baseline_train_ce,
            "latest_train_ce": self.latest_train_ce,
            "latest_train_drop": self.latest_train_drop,
            "last_train_check_step": self.last_train_check_step,
            "train_checks": self.train_checks,
            "train_failures": self.train_failures,
            "train_passed": self.train_passed,
            "baseline_val_ce": self.baseline_val_ce,
            "baseline_val_step": self.baseline_val_step,
            "latest_val_ce": self.latest_val_ce,
            "latest_val_improvement": self.latest_val_improvement,
            "last_val_step": self.last_val_step,
            "val_checks": self.val_checks,
            "val_failures": self.val_failures,
            "val_passed": self.val_passed,
            "failure_reason": self.failure_reason,
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        version = int(state.get("version", self.STATE_VERSION))
        if version != self.STATE_VERSION:
            raise ValueError(
                f"Unsupported loss guard state version {version}; "
                f"expected {self.STATE_VERSION}"
            )
        self._baseline_train_values = [
            float(value) for value in state.get("baseline_train_values", [])
        ][-self.config.baseline_window :]
        self._recent_train_values = deque(
            (float(value) for value in state.get("recent_train_values", [])),
            maxlen=self.config.recent_window,
        )
        for name in (
            "baseline_train_ce",
            "latest_train_ce",
            "latest_train_drop",
            "baseline_val_ce",
            "latest_val_ce",
            "latest_val_improvement",
        ):
            value = state.get(name)
            setattr(self, name, None if value is None else float(value))
        for name in (
            "last_train_check_step",
            "train_checks",
            "train_failures",
            "baseline_val_step",
            "last_val_step",
            "val_checks",
            "val_failures",
        ):
            value = state.get(name)
            setattr(self, name, None if value is None else int(value))
        self.train_passed = bool(state.get("train_passed", self.train_passed))
        self.val_passed = bool(state.get("val_passed", self.val_passed))
        failure_reason = state.get("failure_reason")
        self.failure_reason = None if failure_reason is None else str(failure_reason)

    def to_dict(self) -> dict[str, Any]:
        state = self.state_dict()
        state.pop("baseline_train_values", None)
        state.pop("recent_train_values", None)
        return {
            "config": self.config.to_dict(),
            "state": state,
        }


def _finite_loss(value: float, source: str) -> float:
    loss = float(value)
    if not math.isfinite(loss):
        raise FloatingPointError(f"Loss guard received non-finite {source} CE loss: {loss}")
    return loss
