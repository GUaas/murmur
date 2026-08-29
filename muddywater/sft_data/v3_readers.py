from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from .records import SFTRecord, normalize_text


OASST_RISK_LABELS = (
    "spam",
    "fails_task",
    "lang_mismatch",
    "pii",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
    "toxicity",
    "violence",
)
HIGH_STAKES_ADVICE = re.compile(
    r"诊断|治疗|药物|处方|剂量|服用|手术|怀孕|孕期|症状|股票|投资|诉讼|法律意见"
)
TULU_UNSTABLE_OR_HIGH_STAKES = re.compile(
    r"\b(?:recent|current|latest|today|financial advice|medical advice|diagnos|treatment|dosage|lawsuit)\b",
    re.IGNORECASE,
)
EXCLUSION_MARKER = re.compile(r"excluding|do not (?:mention|use)|without using", re.IGNORECASE)
EXACT_SENTENCES = re.compile(r"exactly\s+(\d+)\s+sentences?", re.IGNORECASE)
EXACT_BULLETS = re.compile(r"exactly\s+(\d+)\s+bullet", re.IGNORECASE)
EXACT_ENDING = re.compile(
    r"end (?:your (?:response|answer)|the (?:response|answer)) with (?:the exact (?:sentence|phrase)\s*:?)?\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)


def _pair_record(
    *,
    instruction: Any,
    context: Any,
    answer: Any,
    source: str,
    category: str,
    source_id: str,
    license_name: str,
) -> SFTRecord | None:
    prompt = normalize_text(instruction)
    context_text = normalize_text(context)
    answer_text = normalize_text(answer)
    if context_text:
        prompt = f"{prompt}\n\n{context_text}" if prompt else context_text
    if not prompt or not answer_text:
        return None
    return SFTRecord(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_text},
        ],
        source=source,
        category=category,
        group_id=f"{source}:{source_id}",
        source_id=source_id,
        metadata={"license": license_name},
    )


def _oasst_label(message: dict[str, Any], name: str) -> tuple[float, int]:
    label = (message.get("labels") or {}).get(name) or {}
    return float(label.get("value", 0.0) or 0.0), int(label.get("count", 0) or 0)


def _oasst_message_is_approved(message: dict[str, Any], expected_role: str) -> bool:
    if message.get("role") != expected_role or message.get("lang") != "zh":
        return False
    if message.get("deleted") or message.get("review_result") is False:
        return False
    if not normalize_text(message.get("text")):
        return False
    if expected_role != "assistant":
        return True
    if message.get("rank") not in (0, None) or int(message.get("review_count") or 0) < 2:
        return False
    for name in OASST_RISK_LABELS:
        value, count = _oasst_label(message, name)
        if count and value > 0.1:
            return False
    for name in ("quality", "helpfulness"):
        value, count = _oasst_label(message, name)
        if count >= 2 and value < 0.5:
            return False
    return True


def _oasst_message_score(message: dict[str, Any]) -> tuple[float, ...]:
    quality, _ = _oasst_label(message, "quality")
    helpfulness, _ = _oasst_label(message, "helpfulness")
    return (
        float(message.get("rank") == 0),
        quality + helpfulness,
        float(message.get("review_count") or 0),
        float(not message.get("synthetic")),
    )


def _best_oasst_path(root: dict[str, Any]) -> list[dict[str, Any]]:
    if not _oasst_message_is_approved(root, "prompter"):
        return []
    path = [root]
    current = root
    expected_role = "assistant"
    while True:
        candidates = [
            reply
            for reply in current.get("replies", [])
            if _oasst_message_is_approved(reply, expected_role)
        ]
        if not candidates:
            break
        current = max(
            candidates,
            key=lambda message: (*_oasst_message_score(message), str(message.get("message_id"))),
        )
        path.append(current)
        expected_role = "prompter" if expected_role == "assistant" else "assistant"
    if path[-1].get("role") != "assistant":
        path.pop()
    return path if len(path) >= 2 else []


def read_oasst2_zh(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "oasst2" / "2023-11-05_oasst2_ready.trees.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            tree = json.loads(line)
            tree_id = str(tree.get("message_tree_id", ""))
            message_path = _best_oasst_path(tree.get("prompt", {}))
            if not message_path:
                continue
            messages = [
                {
                    "role": "user" if message["role"] == "prompter" else "assistant",
                    "content": normalize_text(message.get("text")),
                }
                for message in message_path
            ]
            if HIGH_STAKES_ADVICE.search("\n".join(message["content"] for message in messages)):
                continue
            yield SFTRecord(
                messages=messages,
                source="oasst2_zh",
                category="human_feedback_dialogue",
                group_id=f"oasst2_zh:{tree_id}",
                source_id=tree_id,
                metadata={"license": "Apache-2.0", "human_feedback": True},
            )


def _reading_prompt(context: str, question: str) -> str:
    return (
        "请根据给定材料回答问题，只依据材料，不补充材料外的信息。\n"
        f"材料：{context}\n问题：{question}"
    )


def read_drcd(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "drcd" / "DRCD_train_simplified_chinese.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for article in payload.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = normalize_text(paragraph.get("context"))
            for question_row in paragraph.get("qas", []):
                answers = question_row.get("answers", [])
                answer = normalize_text(answers[0].get("text") if answers else "")
                question = normalize_text(question_row.get("question"))
                if not context or not question or not answer or answer not in context:
                    continue
                source_id = str(question_row.get("id", ""))
                yield SFTRecord(
                    messages=[
                        {"role": "user", "content": _reading_prompt(context, question)},
                        {"role": "assistant", "content": answer},
                    ],
                    source="drcd",
                    category="reading_comprehension_traditional_source",
                    group_id=f"drcd:{paragraph.get('id', source_id)}",
                    source_id=source_id,
                    metadata={"license": "CC-BY-SA-3.0", "human_annotated": True},
                )


def _curated_field(row: dict[str, Any], field_name: str) -> str:
    curated = row.get(f"new-{field_name}") or {}
    values = curated.get("value") or []
    for value in reversed(values):
        normalized = normalize_text(value)
        if normalized:
            return normalized
    return normalize_text(row.get(f"original-{field_name}"))


def _meets_obvious_constraints(prompt: str, answer: str) -> bool:
    for clause in re.split(r"[.\n]", prompt):
        if not EXCLUSION_MARKER.search(clause):
            continue
        for excluded in re.findall(r'[\"\u201c\u201d]([^\"\u201c\u201d]+)[\"\u201c\u201d]', clause):
            if re.search(rf"\b{re.escape(excluded)}\b", answer, re.IGNORECASE):
                return False
    if re.search(r"(?:all|entire output).*lowercase|all lowercase", prompt, re.IGNORECASE):
        if re.search(r"[A-Z]", answer):
            return False
    sentence_match = EXACT_SENTENCES.search(prompt)
    if sentence_match:
        sentence_count = len(re.findall(r"[.!?](?=\s|$|[\"'])", answer))
        if sentence_count != int(sentence_match.group(1)):
            return False
    bullet_match = EXACT_BULLETS.search(prompt)
    if bullet_match:
        bullet_count = len(re.findall(r"(?m)^\s*[-*•]\s+", answer))
        if bullet_count != int(bullet_match.group(1)):
            return False
    ending_match = EXACT_ENDING.search(prompt)
    if ending_match and not answer.rstrip().endswith(ending_match.group(1)):
        return False
    return True


def read_dolly_curated(raw_dir: Path, category: str) -> Iterator[SFTRecord]:
    path = raw_dir / "dolly_curated" / "data" / "train-00000-of-00001-15a05aeec7726f9d.parquet"
    for row in pq.read_table(path).to_pylist():
        if row.get("category") != category:
            continue
        record = _pair_record(
            instruction=_curated_field(row, "instruction"),
            context=_curated_field(row, "context"),
            answer=_curated_field(row, "response"),
            source=f"dolly_{category}",
            category=f"english_{category}",
            source_id=str(row.get("id", "")),
            license_name="CC-BY-SA-3.0",
        )
        if record is not None:
            yield record


def read_tulu_instruction_following(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "tulu_if" / "data" / "train-00000-of-00001.parquet"
    for row in pq.read_table(path).to_pylist():
        constraints = row.get("constraints") or []
        messages = row.get("messages") or []
        if not 1 <= len(constraints) <= 2 or len(messages) != 2:
            continue
        normalized_messages = [
            {
                "role": str(message.get("role", "")),
                "content": normalize_text(message.get("content")),
            }
            for message in messages
        ]
        if [message["role"] for message in normalized_messages] != ["user", "assistant"]:
            continue
        combined = "\n".join(message["content"] for message in normalized_messages)
        if TULU_UNSTABLE_OR_HIGH_STAKES.search(combined):
            continue
        if not _meets_obvious_constraints(
            normalized_messages[0]["content"], normalized_messages[1]["content"]
        ):
            continue
        source_id = str(row.get("id", ""))
        yield SFTRecord(
            messages=normalized_messages,
            source="tulu_instruction_following",
            category="strict_instruction_following_english",
            group_id=f"tulu_instruction_following:{source_id}",
            source_id=source_id,
            metadata={
                "license": "ODC-BY-1.0",
                "constraints": list(map(str, constraints)),
            },
        )
