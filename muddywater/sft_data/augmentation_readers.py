from __future__ import annotations

import ast
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from .records import SFTRecord, normalize_messages, normalize_text


CALCULATOR_ANNOTATION_WITH_RESULT = re.compile(r"<<([^=<>]+)=([^<>]+)>>\s*\2")
CALCULATOR_ANNOTATION = re.compile(r"<<([^<>]+)>>")
FINAL_ANSWER = re.compile(r"####\s*([^\n]+)")
HAN_CHARACTER = re.compile(r"[\u3400-\u9fff]")
LATIN_CHARACTER = re.compile(r"[A-Za-z]")
CHOICE_ANSWER = re.compile(r"^\s*[A-Da-d]\s*[)）.、:：]")
CHOICE_OPTION = re.compile(r"(?:^|\n)\s*[A-Da-d]\s*[)）.、:：]")
TRANSLATED_TASK_META = re.compile(
    r"^(?:在这个任务中|在这项任务中|在本任务中|你会得到|你将得到|给定一组)"
)


def _pair_record(
    *,
    instruction: Any,
    input_text: Any,
    output: Any,
    source: str,
    category: str,
    source_id: str,
    metadata: dict[str, Any] | None = None,
) -> SFTRecord | None:
    prompt = normalize_text(instruction)
    context = normalize_text(input_text)
    answer = normalize_text(output)
    if context:
        prompt = f"{prompt}\n\n{context}" if prompt else context
    if not prompt or not answer:
        return None
    return SFTRecord(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        source=source,
        category=category,
        group_id=f"{source}:{source_id}",
        source_id=source_id,
        metadata=metadata or {},
    )


def _clean_gsm8k_answer(value: Any) -> str:
    answer = normalize_text(value)
    answer = CALCULATOR_ANNOTATION_WITH_RESULT.sub(r"\1 = \2", answer)
    answer = CALCULATOR_ANNOTATION.sub(lambda match: match.group(1).replace("=", " = "), answer)
    answer = FINAL_ANSWER.sub(r"答案：\1", answer)
    return answer


def _is_curated_chinese_instruction(prompt: str, answer: str) -> bool:
    combined = f"{prompt}\n{answer}"
    han_count = len(HAN_CHARACTER.findall(combined))
    latin_count = len(LATIN_CHARACTER.findall(combined))
    if han_count < 12 or latin_count > max(24, int(han_count * 0.25)):
        return False
    if TRANSLATED_TASK_META.search(prompt):
        return False
    if CHOICE_ANSWER.search(answer) and len(CHOICE_OPTION.findall(prompt)) < 2:
        return False
    if len(prompt) > 1_200 or len(answer) > 1_600:
        return False
    return True


def read_gsm8k_zh(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "gsm8k_zh" / "GSM8K_zh.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    for index, row in enumerate(rows, start=1):
        if str(row.get("split", "train")).lower() != "train":
            continue
        record = _pair_record(
            instruction=row.get("question_zh"),
            input_text="",
            output=_clean_gsm8k_answer(row.get("answer_zh")),
            source="gsm8k_zh",
            category="arithmetic_reasoning",
            source_id=str(index),
            metadata={"license": "MIT", "translated_from": "openai/gsm8k"},
        )
        if record is not None:
            yield record


def read_coig_translated(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "coig_alignment" / "translated_instructions.jsonl"
    with path.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = normalize_text(row.get("trans_instruction", row.get("instruction")))
            input_text = normalize_text(row.get("trans_input", row.get("input")))
            output = normalize_text(row.get("trans_output", row.get("output")))
            prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
            if not _is_curated_chinese_instruction(prompt, output):
                continue
            record = _pair_record(
                instruction=instruction,
                input_text=input_text,
                output=output,
                source="coig_translated_verified",
                category="general_instruction",
                source_id=str(index),
                metadata={"license": "Apache-2.0/MIT", "human_reviewed": True},
            )
            if record is not None:
                yield record


def read_coig_human_value(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "coig_alignment" / "human_value_alignment_instructions_part1.json"
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    for index, row in enumerate(rows, start=1):
        record = _pair_record(
            instruction=row.get("instruction"),
            input_text=row.get("input"),
            output=row.get("output"),
            source="coig_human_value",
            category="human_value_alignment",
            source_id=str(index),
            metadata={"license": "Apache-2.0"},
        )
        if record is not None:
            yield record


def _clean_serialized_code_list(value: Any) -> str:
    text = normalize_text(value)
    list_start = text.find("[")
    if list_start < 0:
        return text
    try:
        parsed = ast.literal_eval(text[list_start:])
    except (SyntaxError, ValueError):
        return text
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, str) for item in parsed):
        return text
    prefix = text[:list_start].rstrip()
    body = "\n\n".join(normalize_text(item) for item in parsed)
    return f"{prefix}\n{body}" if prefix else body


def read_coig_leetcode(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "coig_alignment" / "leetcode_instructions.jsonl"
    with path.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_type = normalize_text(row.get("task_type", "code")) or "code"
            record = _pair_record(
                instruction=row.get("instruction"),
                input_text=_clean_serialized_code_list(row.get("input")),
                output=_clean_serialized_code_list(row.get("output")),
                source="coig_leetcode",
                category=f"code_{task_type}",
                source_id=str(index),
                metadata={
                    "license": "Apache-2.0/CC-BY-SA-4.0",
                    "program_lang": row.get("program_lang"),
                },
            )
            if record is not None:
                yield record


def _decode_counterfactual_response(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    text = normalize_text(value)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def read_coig_counterfactual(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "coig_alignment" / "counterfactural_correction_multi_round_chat.tar.gz"
    with tarfile.open(path, "r:gz") as archive:
        # Iterate in archive order. Random seeking inside a gzip-compressed tar
        # repeatedly decompresses earlier bytes and is prohibitively slow here.
        for member in archive:
            if not member.isfile():
                continue
            binary = archive.extractfile(member)
            if binary is None:
                continue
            row = json.load(binary)
            messages: list[dict[str, str]] = []
            for round_index in range(5):
                round_data = row.get(f"round_{round_index}", {})
                response = _decode_counterfactual_response(round_data.get("response"))
                if response is None:
                    continue
                question = normalize_text(response.get("Q"))
                answer = normalize_text(response.get("A"))
                if question and answer:
                    messages.extend(
                        [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    )
            if len(messages) < 4:
                continue
            source_id = Path(member.name).stem
            yield SFTRecord(
                messages=messages,
                source="coig_counterfactual",
                category="factual_correction_dialogue",
                group_id=f"coig_counterfactual:{source_id}",
                source_id=source_id,
                metadata={"license": "Apache-2.0", "entity": row.get("entity")},
            )


def _reading_prompt(context: str, question: str) -> str:
    return (
        "请根据给定材料回答问题，只依据材料，不补充材料外的信息。\n"
        f"材料：{context}\n问题：{question}"
    )


def read_cmrc2018(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "cmrc2018" / "data" / "train-00000-of-00001.parquet"
    for row in pq.read_table(path).to_pylist():
        answers = row.get("answers", {}).get("text", [])
        answer = normalize_text(answers[0] if answers else "")
        context = normalize_text(row.get("context"))
        question = normalize_text(row.get("question"))
        if not context or not question or not answer or answer not in context:
            continue
        yield SFTRecord(
            messages=[
                {"role": "user", "content": _reading_prompt(context, question)},
                {"role": "assistant", "content": answer},
            ],
            source="cmrc2018",
            category="reading_comprehension",
            group_id=f"cmrc2018:{row['id']}",
            source_id=str(row["id"]),
            metadata={"license": "CC-BY-SA-4.0", "human_annotated": True},
        )


DUREADER_NOISE = re.compile(
    r"\\x0a|望采纳|请采纳|谢谢阅读|希望能够帮助|以下结果由|译典通|\|.{0,200}\|"
)


def read_dureader_robust(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "dureader_robust" / "dureader_robust-data.tar.gz"
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember("dureader_robust-data/train.json")
        binary = archive.extractfile(member)
        if binary is None:
            return
        data = json.load(binary)
    for article_index, article in enumerate(data.get("data", []), start=1):
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", []), start=1):
            context = normalize_text(paragraph.get("context"))
            if len(context) < 80 or DUREADER_NOISE.search(context):
                continue
            for qa in paragraph.get("qas", []):
                question = normalize_text(qa.get("question"))
                answers = qa.get("answers", [])
                answer = normalize_text(answers[0].get("text") if answers else "")
                if not 4 <= len(question) <= 160 or not answer or len(answer) > 160:
                    continue
                if answer not in context:
                    continue
                source_id = str(qa.get("id", f"{article_index}:{paragraph_index}"))
                yield SFTRecord(
                    messages=[
                        {"role": "user", "content": _reading_prompt(context, question)},
                        {"role": "assistant", "content": answer},
                    ],
                    source="dureader_robust",
                    category="reading_comprehension_robust",
                    group_id=f"dureader_robust:{source_id}",
                    source_id=source_id,
                    metadata={"license": "Apache-2.0", "human_annotated": True},
                )


def read_doit(raw_dir: Path, ability: str) -> Iterator[SFTRecord]:
    """Read one human-reviewed DoIT ability split as conversation records."""
    path = raw_dir / "doit" / "curated" / "full" / f"{ability}_full.json"
    source = f"doit_{ability}"
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = normalize_messages(row.get("messages", []))
            if len(messages) < 2:
                continue
            source_id = str(row.get("idx", line_number))
            yield SFTRecord(
                messages=messages,
                source=source,
                category=ability,
                group_id=f"{source}:{source_id}",
                source_id=source_id,
                metadata={
                    "license": "MIT",
                    "human_reviewed": True,
                    "question_format": row.get("question_format"),
                },
            )
