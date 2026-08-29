from __future__ import annotations

from muddywater.sft_data.v4_catalog import INFINITY_GEN
from muddywater.sft_data.v4_readers import (
    convert_infinity_row,
    infinity_category,
    normalize_infinity_language,
)


def _row(*, reward: float = 30.0, answer: str = "A concise, useful answer.") -> dict[str, object]:
    return {
        "id": "example-1",
        "source": "Subjective",
        "langdetect": "en",
        "reward": reward,
        "label": {
            "ability_en": ["programming ability"],
            "cate_ability_en": ["programming and software development"],
        },
        "conversations": [
            {"from": "human", "value": "Explain how a Python iterator works."},
            {"from": "gpt", "value": answer},
        ],
    }


def test_infinity_language_normalization_is_bilingual() -> None:
    assert normalize_infinity_language("zh-cn") == "zh"
    assert normalize_infinity_language("zh_TW") == "zh"
    assert normalize_infinity_language("en") == "en"


def test_infinity_category_maps_programming_labels() -> None:
    assert infinity_category(
        {"cate_ability_en": ["programming and software development"]}
    ) == "code_and_programming"


def test_infinity_row_conversion_preserves_roles_and_provenance() -> None:
    record = convert_infinity_row(_row(), shard=INFINITY_GEN, language="en")
    assert record is not None
    assert [message["role"] for message in record.messages] == ["user", "assistant"]
    assert record.source == "infinity_gen_en"
    assert record.metadata["license"] == "CC-BY-SA-4.0"


def test_infinity_row_conversion_rejects_low_reward_and_identity_pollution() -> None:
    assert convert_infinity_row(_row(reward=1.0), shard=INFINITY_GEN, language="en") is None
    polluted = _row(answer="As an AI language model created by OpenAI, I cannot answer.")
    assert convert_infinity_row(polluted, shard=INFINITY_GEN, language="en") is None


def test_infinity_row_conversion_rejects_high_stakes_advice() -> None:
    row = _row(answer="Here is the dosage and a prescription you should take.")
    assert convert_infinity_row(row, shard=INFINITY_GEN, language="en") is None


def test_infinity_row_conversion_rejects_dangerous_child_activity() -> None:
    row = _row(answer="Let the children compete by throwing a Swiss army knife at targets.")
    assert convert_infinity_row(row, shard=INFINITY_GEN, language="en") is None
