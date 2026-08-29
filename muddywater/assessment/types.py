from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChoiceExample:
    """One multiple-choice language-model evaluation example."""

    task: str
    example_id: str
    category: str
    context: str
    options: tuple[str, ...]
    answer_index: int
    primary_metric: str = "accuracy"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("ChoiceExample requires at least two options")
        if not 0 <= int(self.answer_index) < len(self.options):
            raise ValueError("answer_index is outside the option range")
        if self.primary_metric not in {"accuracy", "accuracy_norm"}:
            raise ValueError("primary_metric must be accuracy or accuracy_norm")


@dataclass(frozen=True)
class TextProbe:
    """A text sample used for tokenizer and language-model diagnostics."""

    probe_id: str
    category: str
    text: str
    language: str


@dataclass(frozen=True)
class GenerationProbe:
    """A prompt plus optional lightweight, deterministic checks."""

    probe_id: str
    category: str
    prompt: str
    accepted_substrings: tuple[str, ...] = ()
    check_refusal: bool = False
    check_python_syntax: bool = False
    notes: str = ""
