from __future__ import annotations

from muddywater.text_simplification.sol_distillation import (
    revert_surface_only_changes,
    validate_output_rows,
)


def test_valid_keep_and_simplified_rows() -> None:
    inputs = [
        {"key": "train:1", "source": "他今天来到学校。"},
        {"key": "train:2", "source": "该项目于2025年完成建设工作。"},
    ]
    outputs = [
        {"key": "train:1", "decision": "keep", "target": "他今天来到学校。"},
        {"key": "train:2", "decision": "simplified", "target": "该项目于2025年建成。"},
    ]

    validated, issues = validate_output_rows(inputs, outputs)

    assert len(validated) == 2
    assert not [issue for issue in issues if issue.severity == "error"]


def test_number_loss_is_a_hard_error() -> None:
    inputs = [{"key": "train:3", "source": "2025年共完成3项任务。"}]
    outputs = [{"key": "train:3", "decision": "simplified", "target": "当年完成任务。"}]

    validated, issues = validate_output_rows(inputs, outputs)

    assert not validated
    assert "number_mismatch" in {issue.code for issue in issues}


def test_order_mismatch_is_a_hard_error() -> None:
    inputs = [
        {"key": "train:4", "source": "甲。"},
        {"key": "train:5", "source": "乙。"},
    ]
    outputs = [
        {"key": "train:5", "decision": "keep", "target": "甲。"},
        {"key": "train:4", "decision": "keep", "target": "乙。"},
    ]

    validated, issues = validate_output_rows(inputs, outputs)

    assert not validated
    assert sum(issue.code == "key_or_order_mismatch" for issue in issues) == 2


def test_row_count_mismatch_is_a_hard_error() -> None:
    inputs = [{"key": "validation:1", "source": "一句话。"}]

    validated, issues = validate_output_rows(inputs, [])

    assert not validated
    codes = {issue.code for issue in issues}
    assert {"row_count_mismatch", "missing_output"}.issubset(codes)


def test_surface_only_change_is_reverted_to_keep() -> None:
    inputs = [{"key": "train:6", "source": "改革开放40年来,经济持续发展。"}]
    outputs = [
        {
            "key": "train:6",
            "decision": "simplified",
            "target": "改革开放40年来，经济持续发展。",
        }
    ]

    cleaned, reverted = revert_surface_only_changes(inputs, outputs)

    assert reverted == ["train:6"]
    assert cleaned == [{"key": "train:6", "decision": "keep", "target": inputs[0]["source"]}]
