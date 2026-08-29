from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .identity_repair import evaluate_candidate, extract_numbers, semantic_surface, topic_flags


VALID_DECISIONS = frozenset({"keep", "simplified"})


@dataclass(frozen=True)
class ValidationIssue:
    key: str
    severity: str
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedOutput:
    key: str
    decision: str
    target: str
    warnings: tuple[str, ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: JSON value must be an object")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _numbers_match(source: str, target: str) -> bool:
    return Counter(extract_numbers(source)) == Counter(extract_numbers(target))


def validate_output_rows(
    input_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> tuple[list[ValidatedOutput], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    validated: list[ValidatedOutput] = []
    if len(input_rows) != len(output_rows):
        issues.append(
            ValidationIssue(
                key="__shard__",
                severity="error",
                code="row_count_mismatch",
                detail=f"input={len(input_rows)}, output={len(output_rows)}",
            )
        )

    for index, input_row in enumerate(input_rows):
        expected_key = str(input_row.get("key", ""))
        source = str(input_row.get("source", ""))
        if index >= len(output_rows):
            issues.append(
                ValidationIssue(expected_key, "error", "missing_output", "output row is missing")
            )
            continue
        output_row = output_rows[index]
        actual_key = str(output_row.get("key", ""))
        decision = str(output_row.get("decision", ""))
        target_value = output_row.get("target")
        target = target_value if isinstance(target_value, str) else ""

        row_has_error = False
        if actual_key != expected_key:
            issues.append(
                ValidationIssue(
                    expected_key,
                    "error",
                    "key_or_order_mismatch",
                    f"expected={expected_key!r}, actual={actual_key!r}",
                )
            )
            row_has_error = True
        if decision not in VALID_DECISIONS:
            issues.append(
                ValidationIssue(expected_key, "error", "invalid_decision", repr(decision))
            )
            row_has_error = True
        if not target.strip():
            issues.append(ValidationIssue(expected_key, "error", "empty_target", "target is empty"))
            row_has_error = True
        if decision == "keep" and target != source:
            issues.append(
                ValidationIssue(expected_key, "error", "keep_changed_text", "keep must copy source exactly")
            )
            row_has_error = True
        if decision == "simplified" and target == source:
            issues.append(
                ValidationIssue(expected_key, "error", "simplified_is_identity", "target equals source")
            )
            row_has_error = True
        if decision == "simplified" and not _numbers_match(source, target):
            issues.append(
                ValidationIssue(
                    expected_key,
                    "error",
                    "number_mismatch",
                    f"source={extract_numbers(source)!r}, target={extract_numbers(target)!r}",
                )
            )
            row_has_error = True

        warnings: tuple[str, ...] = ()
        if decision == "simplified" and not row_has_error:
            assessment = evaluate_candidate(source, target)
            warnings = tuple(
                reason
                for reason in assessment.reasons
                if reason not in {"model_kept_identity", "number_loss"}
            )
            for warning in warnings:
                issues.append(
                    ValidationIssue(expected_key, "warning", warning, "automatic quality heuristic")
                )
        if not row_has_error:
            validated.append(ValidatedOutput(expected_key, decision, target, warnings))

    for output_row in output_rows[len(input_rows) :]:
        issues.append(
            ValidationIssue(
                str(output_row.get("key", "__extra__")),
                "error",
                "extra_output",
                "output has no matching input row",
            )
        )
    return validated, issues


def load_and_validate_shard(
    input_path: Path,
    output_path: Path,
) -> tuple[list[ValidatedOutput], list[ValidationIssue]]:
    return validate_output_rows(read_jsonl(input_path), read_jsonl(output_path))


def summarize_changes(
    sources: Mapping[str, str],
    outputs: Iterable[ValidatedOutput],
) -> dict[str, Any]:
    output_rows = list(outputs)
    simplified = [row for row in output_rows if row.decision == "simplified"]
    kept = [row for row in output_rows if row.decision == "keep"]
    source_chars = sum(len(sources[row.key]) for row in output_rows)
    target_chars = sum(len(row.target) for row in output_rows)
    political = [row for row in output_rows if topic_flags(sources[row.key])]
    political_simplified = sum(row.decision == "simplified" for row in political)
    return {
        "processed": len(output_rows),
        "simplified": len(simplified),
        "kept": len(kept),
        "simplified_rate": len(simplified) / len(output_rows) if output_rows else 0.0,
        "source_characters": source_chars,
        "target_characters": target_chars,
        "character_reduction_rate": (
            (source_chars - target_chars) / source_chars if source_chars else 0.0
        ),
        "political_rows": len(political),
        "political_simplified": political_simplified,
        "political_simplified_rate": (
            political_simplified / len(political) if political else 0.0
        ),
        "warning_rows": sum(bool(row.warnings) for row in output_rows),
    }


def revert_surface_only_changes(
    input_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if len(input_rows) != len(output_rows):
        raise ValueError(f"input rows={len(input_rows)}, output rows={len(output_rows)}")
    cleaned: list[dict[str, Any]] = []
    reverted: list[str] = []
    for input_row, output_row in zip(input_rows, output_rows, strict=True):
        key = str(input_row["key"])
        if str(output_row.get("key", "")) != key:
            raise ValueError(f"{key}: key/order mismatch")
        source = str(input_row["source"])
        target = str(output_row.get("target", ""))
        row = dict(output_row)
        if (
            row.get("decision") == "simplified"
            and source != target
            and semantic_surface(source) == semantic_surface(target)
        ):
            row["decision"] = "keep"
            row["target"] = source
            reverted.append(key)
        cleaned.append(row)
    return cleaned, reverted
