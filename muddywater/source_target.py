from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_TARGET_SEPARATOR = "<|im_start|>"


def _clean_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


@dataclass(frozen=True)
class SourceTargetTemplate:
    """Compact single-task format with target-only supervision."""

    source_key: str = "source"
    target_key: str = "target"
    target_separator: str = DEFAULT_TARGET_SEPARATOR
    source_label: str = ""
    target_label: str | None = None

    def __post_init__(self) -> None:
        if not self.source_key.strip() or not self.target_key.strip():
            raise ValueError("source_key and target_key must not be empty")
        if not self.effective_target_label:
            raise ValueError("target_label/target_separator must not be empty")
        if self.source_label and self.source_label == self.effective_target_label:
            raise ValueError("source_label and target_label must be different")

    @property
    def effective_target_label(self) -> str:
        """Return the configured target label, preserving separator compatibility."""

        return self.target_label if self.target_label is not None else self.target_separator

    @property
    def reserved_labels(self) -> tuple[str, ...]:
        """Labels that may never occur inside source or target content."""

        labels = (self.source_label, self.effective_target_label)
        return tuple(label for label in labels if label)

    def normalize_pair(self, source: Any, target: Any) -> tuple[str, str]:
        normalized_source = _clean_text(source, field_name=self.source_key)
        normalized_target = _clean_text(target, field_name=self.target_key)
        for field_name, value in (
            (self.source_key, normalized_source),
            (self.target_key, normalized_target),
        ):
            for label in self.reserved_labels:
                if label in value:
                    raise ValueError(
                        f"{field_name} contains the reserved label {label!r}"
                    )
        return normalized_source, normalized_target

    def from_record(self, record: Mapping[str, Any]) -> tuple[str, str]:
        if self.source_key not in record:
            raise KeyError(f"Missing source key: {self.source_key}")
        if self.target_key not in record:
            raise KeyError(f"Missing target key: {self.target_key}")
        return self.normalize_pair(record[self.source_key], record[self.target_key])

    def format_pair(self, source: Any, target: Any) -> tuple[str, str]:
        normalized_source, normalized_target = self.normalize_pair(source, target)
        prompt = self.generation_prompt(normalized_source)
        text = prompt + normalized_target
        label_mask = "0" * len(prompt) + "1" * len(normalized_target)
        return text, label_mask

    def format_record(self, record: Mapping[str, Any]) -> tuple[str, str]:
        source, target = self.from_record(record)
        return self.format_pair(source, target)

    def generation_prompt(self, source: Any) -> str:
        normalized_source = _clean_text(source, field_name=self.source_key)
        for label in self.reserved_labels:
            if label in normalized_source:
                raise ValueError(
                    f"{self.source_key} contains the reserved label {label!r}"
                )
        return self.source_label + normalized_source + self.effective_target_label
