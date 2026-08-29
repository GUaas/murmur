from __future__ import annotations

import json
from pathlib import Path

from muddywater.sft_data.v5_continuation import (
    V5ContinuationOptions,
    build_v5_continuation_dataset,
)


def _row(index: int, source: str, category: str) -> dict[str, object]:
    return {
        "id": f"row-{index}",
        "messages": [
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ],
        "source": source,
        "category": category,
        "num_tokens": 8,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_builds_new_data_plus_stratified_anchor_replay(tmp_path: Path) -> None:
    base = tmp_path / "v3"
    expanded = tmp_path / "v4"
    output = tmp_path / "v5"
    base_train = []
    sources = [
        ("synthetic_safety_verified", "safe"),
        ("synthetic_identity_verified", "identity_name"),
        ("synthetic_math_verified", "add_subtract"),
        ("synthetic_instruction_verified", "fixed_list"),
        ("coig_cqia", "information-seeking"),
    ]
    for index in range(50):
        source, category = sources[index % len(sources)]
        base_train.append(_row(index, source, category))
    base_validation = [_row(90, "cmrc2018", "reading_comprehension")]
    additions = [_row(100 + index, "infinity", "instruction") for index in range(20)]

    _write(base / "processed" / "train.jsonl", base_train)
    _write(base / "processed" / "validation.jsonl", base_validation)
    _write(expanded / "processed" / "train.jsonl", base_train + additions)
    _write(expanded / "processed" / "validation.jsonl", base_validation)
    tokenizer = expanded / "assets" / "sp_unigram_32k.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.write_bytes(b"tokenizer")

    report = build_v5_continuation_dataset(
        V5ContinuationOptions(
            base_root=base,
            expanded_root=expanded,
            output_root=output,
            anchor_fraction=0.20,
            seed=7,
        )
    )

    assert report["counts"]["new_train_records"] == 20
    assert report["counts"]["anchor_records"] == 5
    assert report["counts"]["train_records"] == 25
    assert report["counts"]["train_validation_content_overlap"] == 0
    assert set(report["policy"]["anchor_groups"]) == {
        "identity",
        "instruction",
        "knowledge",
        "math",
        "safety",
    }

    repeated = build_v5_continuation_dataset(
        V5ContinuationOptions(
            base_root=base,
            expanded_root=expanded,
            output_root=tmp_path / "v5-repeat",
            anchor_fraction=0.20,
            seed=7,
        )
    )
    assert report["files"]["train"]["sha256"] == repeated["files"]["train"]["sha256"]
