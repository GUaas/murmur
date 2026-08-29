from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import sentencepiece as spm

from muddywater.templates import format_messages


def _content_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_and_check(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"{path.name}:{line_number}: invalid messages")
                continue
            roles = [message.get("role") for message in messages]
            non_system = [role for role in roles if role != "system"]
            if not non_system or non_system[0] != "user" or non_system[-1] != "assistant":
                errors.append(f"{path.name}:{line_number}: invalid boundary roles")
            if any(left == right for left, right in zip(non_system, non_system[1:])):
                errors.append(f"{path.name}:{line_number}: non-alternating roles")
            if int(row.get("num_tokens", 0)) > 2048:
                errors.append(f"{path.name}:{line_number}: stored token length exceeds 2048")
            rows.append(row)
    return rows, errors


def validate_dataset(
    processed_dir: Path,
    tokenizer_path: Path,
    *,
    sample_size: int = 2_000,
    seed: int = 20260802,
) -> dict[str, Any]:
    train, train_errors = _load_and_check(processed_dir / "train.jsonl")
    validation, validation_errors = _load_and_check(processed_dir / "validation.jsonl")
    all_rows = train + validation
    errors = train_errors + validation_errors

    ids = [str(row.get("id", "")) for row in all_rows]
    content_hashes = [_content_hash(row["messages"]) for row in all_rows]
    duplicate_ids = len(ids) - len(set(ids))
    duplicate_contents = len(content_hashes) - len(set(content_hashes))
    train_groups = {str(row.get("group_id")) for row in train}
    validation_groups = {str(row.get("group_id")) for row in validation}
    group_overlap = sorted(train_groups & validation_groups)

    try:
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    except OSError:
        tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    rng = random.Random(seed)
    sample = rng.sample(all_rows, min(sample_size, len(all_rows)))
    token_mismatches = 0
    actual_max = 0
    for row in sample:
        text = format_messages(row["messages"], chat_template="chatml")
        actual = len(tokenizer.encode(text, out_type=int))
        actual_max = max(actual_max, actual)
        if actual != int(row.get("num_tokens", -1)):
            token_mismatches += 1
        if actual > 2048:
            errors.append(f"sampled record {row.get('id')} exceeds 2048 tokens")

    source_counts = Counter(str(row.get("source")) for row in all_rows)
    valid = not errors and duplicate_ids == 0 and duplicate_contents == 0 and not group_overlap and token_mismatches == 0
    return {
        "valid": valid,
        "train_records": len(train),
        "validation_records": len(validation),
        "total_records": len(all_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_contents": duplicate_contents,
        "train_validation_group_overlap": len(group_overlap),
        "sampled_token_checks": len(sample),
        "sampled_token_mismatches": token_mismatches,
        "sampled_actual_max_tokens": actual_max,
        "errors": errors[:100],
    }
