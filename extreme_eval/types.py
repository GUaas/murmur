from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    source: str
    target: str
    must_keep: tuple[str, ...] = field(default_factory=tuple)
    expect_unchanged: bool = False
    forbidden_exact: tuple[str, ...] = field(default_factory=tuple)
    perturbation_group: str | None = None
    perturbation: str | None = None


@dataclass(frozen=True)
class GenerationMeasurement:
    text: str
    finish_reason: str
    generated_tokens: int
    latency_ms: float
    prompt_tokens: int

