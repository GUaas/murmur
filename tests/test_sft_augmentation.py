from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from muddywater.sft_data.augmentation_readers import (
    _clean_serialized_code_list,
    _clean_gsm8k_answer,
    _decode_counterfactual_response,
    _is_curated_chinese_instruction,
    read_doit,
    read_gsm8k_zh,
)
from muddywater.sft_data.coverage import build_scenario_coverage
from muddywater.sft_data.v3_readers import (
    _best_oasst_path,
    _curated_field,
    _meets_obvious_constraints,
)
from muddywater.sft_data.augmentation_identity import (
    identity_evaluation_cases,
    read_verified_identity_curriculum,
)
from muddywater.sft_data.augmentation_synthetic import (
    read_verified_instruction_curriculum,
    read_verified_math_curriculum,
    read_verified_safety_curriculum,
)


def test_gsm8k_answer_cleanup_preserves_reasoning_and_final_answer() -> None:
    cleaned = _clean_gsm8k_answer("先算 <<12-5=7>>7。\n#### 7")
    assert "12-5 = 7" in cleaned
    assert "答案：7" in cleaned
    assert "<<" not in cleaned


def test_gsm8k_reader_uses_only_training_rows() -> None:
    with TemporaryDirectory() as temporary:
        raw_dir = Path(temporary)
        source_dir = raw_dir / "gsm8k_zh"
        source_dir.mkdir()
        rows = [
            {"split": "train", "question_zh": "一加一是多少？", "answer_zh": "#### 2"},
            {"split": "test", "question_zh": "二加二是多少？", "answer_zh": "#### 4"},
        ]
        (source_dir / "GSM8K_zh.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
        records = list(read_gsm8k_zh(raw_dir))
        assert len(records) == 1
        assert records[0].messages[-1]["content"] == "答案：2"


def test_verified_augmentation_generators_are_deterministic_and_sized() -> None:
    math_records = list(read_verified_math_curriculum())
    safety_records = list(read_verified_safety_curriculum())
    instruction_records = list(read_verified_instruction_curriculum())
    assert len(math_records) == 8_000
    assert len(safety_records) == 2_000
    assert len(instruction_records) == 2_000
    assert math_records[0].messages[-1]["content"].isdigit()
    assert "不要" in safety_records[0].messages[-1]["content"]
    assert json.loads(instruction_records[0].messages[-1]["content"])["age"] == 8


def test_identity_curriculum_is_balanced_unique_and_canonical() -> None:
    records = list(read_verified_identity_curriculum())
    assert len(records) == 2_500
    assert len({record.content_key for record in records}) == 2_500
    counts: dict[str, int] = {}
    for record in records:
        counts[record.category] = counts.get(record.category, 0) + 1
    assert counts == {
        "identity_name": 320,
        "identity_developer": 320,
        "identity_combined": 480,
        "identity_correction": 480,
        "capability_boundary": 400,
        "experience_boundary": 300,
        "identity_unknown_internal": 200,
    }
    combined = [record for record in records if record.category == "identity_combined"]
    assert all("murmur" in record.messages[-1]["content"] for record in combined)
    assert all("MuddyWaterAI" in record.messages[-1]["content"] for record in combined)
    assert not any("我是ChatGPT" in record.messages[-1]["content"] for record in records)
    eval_cases = identity_evaluation_cases()
    assert len(eval_cases) == 12
    assert all(case.get("must_include") or case.get("semantic_expectation") for case in eval_cases)


def test_translated_instruction_filter_rejects_under_specified_or_english_tasks() -> None:
    assert not _is_curated_chinese_instruction(
        "你会得到一组选项，请选择。该校的_____一直下降。", "b) 入学"
    )
    assert not _is_curated_chinese_instruction(
        "在这个任务中，请补全英文句子：I have swimming.", "I do not know how to swim."
    )
    assert _is_curated_chinese_instruction(
        "把下面这句话改写得更简洁：虽然下雨，但是我们仍按计划出发。",
        "虽然下雨，我们仍按计划出发。",
    )


def test_doit_reader_preserves_multiturn_conversations() -> None:
    with TemporaryDirectory() as temporary:
        raw_dir = Path(temporary)
        source_dir = raw_dir / "doit" / "curated" / "full"
        source_dir.mkdir(parents=True)
        row = {
            "messages": [
                {"role": "user", "content": "第一问"},
                {"role": "assistant", "content": "第一答"},
                {"role": "user", "content": "第二问"},
                {"role": "assistant", "content": "第二答"},
            ],
            "idx": 7,
            "question_format": 1,
        }
        path = source_dir / "understanding_full.json"
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        records = list(read_doit(raw_dir, "understanding"))
        assert len(records) == 1
        assert len(records[0].messages) == 4
        assert records[0].source == "doit_understanding"


def test_code_and_counterfactual_helpers_normalize_source_formats() -> None:
    assert _clean_serialized_code_list("参考：['print(1)', 'print(2)']") == (
        "参考：\nprint(1)\n\nprint(2)"
    )
    assert _decode_counterfactual_response('```json\n{"Q":"问题", "A":"回答"}\n```') == {
        "Q": "问题",
        "A": "回答",
    }


def test_scenario_coverage_supports_overlapping_scenarios() -> None:
    report = build_scenario_coverage(
        [
            {"source": "coig_leetcode", "category": "code_generation"},
            {"source": "synthetic_identity_verified", "category": "identity_combined"},
            {"source": "cmrc2018", "category": "reading_comprehension"},
        ]
    )
    counts = {row["key"]: row["records"] for row in report["scenarios"]}
    assert report["coverage_ratio"] == 1.0
    assert counts["programming"] == 1
    assert counts["identity"] == 1
    assert counts["reading_comprehension"] == 1


def test_oasst_path_uses_reviewed_top_ranked_chinese_reply() -> None:
    labels = {
        "quality": {"value": 0.8, "count": 3},
        "helpfulness": {"value": 0.8, "count": 3},
    }
    root = {
        "message_id": "root",
        "role": "prompter",
        "lang": "zh",
        "text": "请解释这个概念。",
        "review_result": True,
        "deleted": False,
        "replies": [
            {
                "message_id": "answer",
                "role": "assistant",
                "lang": "zh",
                "text": "这是通过审核的回答。",
                "review_count": 3,
                "review_result": True,
                "deleted": False,
                "rank": 0,
                "synthetic": False,
                "labels": labels,
                "replies": [],
            },
            {
                "message_id": "lower-rank",
                "role": "assistant",
                "lang": "zh",
                "text": "较低排名回答。",
                "review_count": 3,
                "review_result": True,
                "deleted": False,
                "rank": 1,
                "labels": labels,
                "replies": [],
            },
        ],
    }
    path = _best_oasst_path(root)
    assert [message["message_id"] for message in path] == ["root", "answer"]


def test_dolly_curated_field_prefers_latest_nonempty_edit() -> None:
    row = {
        "original-response": "original",
        "new-response": {"value": ["first edit", "", "final edit"]},
    }
    assert _curated_field(row, "response") == "final edit"


def test_obvious_instruction_constraints_are_verified() -> None:
    assert _meets_obvious_constraints(
        'Write exactly 2 sentences and do not mention "blue".',
        "The sky looks clear. The sea looks calm.",
    )
    assert not _meets_obvious_constraints(
        'Write exactly 2 sentences and do not mention "blue".',
        "The blue sky looks clear. The sea looks calm.",
    )
    assert not _meets_obvious_constraints(
        "Write exactly 3 sentences.",
        "One sentence. Two sentences.",
    )
