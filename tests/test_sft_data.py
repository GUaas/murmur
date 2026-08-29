from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from muddywater.sft_data.filtering import SMOLTALK_TEMPLATE_ARTIFACT_PATTERN
from muddywater.sft_data.filtering import RecordFilter
from muddywater.sft_data.pipeline import _remove_stale_by_source_files
from muddywater.sft_data.readers import _window_dialogue
from muddywater.sft_data.records import SFTRecord, normalize_messages
from muddywater.sft_data.sampling import select_records
from muddywater.sft_data.synthetic import read_verified_synthetic


def test_normalize_messages_maps_source_roles() -> None:
    messages = normalize_messages(
        [
            {"role": "usr", "content": "  你好  "},
            {"role": "sys", "content": "您好"},
        ]
    )
    assert messages == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好"},
    ]


def test_window_dialogue_preserves_assistant_ending() -> None:
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
        for index in range(20)
    ]
    windows = list(_window_dialogue(messages, max_messages=8, stride=6))
    assert windows
    assert all(window[0]["role"] == "user" for _, window in windows)
    assert all(window[-1]["role"] == "assistant" for _, window in windows)


def test_record_id_is_stable() -> None:
    record = SFTRecord(
        messages=[{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}],
        source="test",
        category="test",
        group_id="group",
        source_id="1",
    )
    assert record.record_id == record.record_id


def test_verified_synthetic_records_are_well_formed() -> None:
    records = list(read_verified_synthetic())
    assert len(records) >= 2_000
    assert all(record.messages[0]["role"] == "user" for record in records)
    assert all(record.messages[-1]["role"] == "assistant" for record in records)


def test_smoltalk_template_artifacts_are_detected() -> None:
    polluted = (
        "过年.GetUserProblemByUidparentId _measurement uprofile "
        "_fieldName1 nickname _value1 user"
    )
    assert SMOLTALK_TEMPLATE_ARTIFACT_PATTERN.search(polluted)
    assert SMOLTALK_TEMPLATE_ARTIFACT_PATTERN.search("这和主题无关，请忽略不相关的内容")


def test_zero_quota_skips_source_without_filtering() -> None:
    class FilterMustNotRun:
        def apply(self, record: SFTRecord) -> SFTRecord:
            raise AssertionError("zero quota should not consume the source")

    record = SFTRecord(
        messages=[{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}],
        source="test",
        category="test",
        group_id="test",
    )
    assert select_records([record], quota=0, record_filter=FilterMustNotRun(), seed=1) == []


def test_stale_by_source_files_are_removed() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        active = directory / "active.jsonl"
        stale = directory / "stale.jsonl"
        unrelated = directory / "notes.txt"
        active.write_text("active", encoding="utf-8")
        stale.write_text("stale", encoding="utf-8")
        unrelated.write_text("keep", encoding="utf-8")
        removed = _remove_stale_by_source_files(directory, {"active"})
        assert removed == [stale]
        assert active.exists()
        assert unrelated.exists()


def test_record_filter_can_load_tokenizer_from_non_ascii_path() -> None:
    tokenizer = Path(__file__).resolve().parents[1] / "tokenizer" / "sp_unigram_32k.model"
    record_filter = RecordFilter(str(tokenizer), min_assistant_tokens=1)
    assert record_filter.min_assistant_tokens == 1
