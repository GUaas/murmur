from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from .records import SFTRecord, normalize_text
from .v4_catalog import InfinityShard


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
}

IDENTITY_CONTAMINATION = re.compile(
    r"\b(?:ChatGPT|OpenAI|Claude|Gemini|Llama)\b|\bas an AI\b|\bas a language model\b"
    r"|我是(?:ChatGPT|Claude|Gemini)|作为(?:一个)?(?:AI|人工智能|语言模型)",
    re.IGNORECASE,
)
HIGH_STAKES_OR_UNSTABLE = re.compile(
    r"\b(?:diagnos(?:e|is)|dosage|prescription|medical advice|legal advice|investment advice|"
    r"stock recommendation|latest|today|current president|current law|breaking news)\b"
    r"|诊断|处方|剂量|服药|医疗建议|法律建议|投资建议|股票推荐|最新消息|今天的新闻|现任总统|现行法律",
    re.IGNORECASE,
)
HARMFUL_INSTRUCTION = re.compile(
    r"\b(?:ransomware|keylogger|credential theft|steal passwords|phishing kit|make a bomb|"
    r"build a bomb|poison someone|evade law enforcement|swiss army knife|throwing knife|"
    r"knife throwing|criminal suspect|profile a suspect)\b"
    r"|勒索软件|键盘记录器|窃取密码|钓鱼网站|制作炸弹|制造炸弹|投毒|逃避执法"
    r"|瑞士军刀|飞刀|投掷.{0,10}(?:刀|军刀)|犯罪嫌疑人|刑事侦查|犯罪画像",
    re.IGNORECASE,
)
DATASET_ARTIFACT = re.compile(
    r"<\|im_(?:start|end)\|>|<tool_(?:call|response)>|<<[^<>\n]{1,120}>>|^\s*####\s*",
    re.MULTILINE,
)
BENCHMARK_PROMPT = re.compile(
    r"\b(?:MMLU|GSM8K|HumanEval|MBPP|AlpacaEval|MT-Bench|Arena-Hard)\b",
    re.IGNORECASE,
)


def normalize_infinity_language(value: Any) -> str:
    language = normalize_text(value).lower().replace("_", "-")
    if language.startswith("zh"):
        return "zh"
    if language.startswith("en"):
        return "en"
    return language


def infinity_category(label: dict[str, Any] | None) -> str:
    label = label or {}
    terms = " ".join(
        str(value).lower()
        for key in ("cate_ability_en", "ability_en")
        for value in (label.get(key) or [])
    )
    categories = (
        (("programming", "software development", "code"), "code_and_programming"),
        (("mathematical", "calculation", "arithmetic"), "mathematical_reasoning"),
        (("logic", "reasoning"), "logical_reasoning"),
        (("creativity", "creative writing"), "creative_writing"),
        (("humanities", "history", "philosophy", "sociology"), "humanities_knowledge"),
        (("data science", "data analysis", "analytics"), "data_analysis"),
        (("information processing", "knowledge", "fact checking"), "knowledge_qa"),
        (("problem solving", "support", "planning"), "planning_advice"),
        (("writing", "language", "translation"), "writing_revision"),
    )
    for needles, category in categories:
        if any(needle in terms for needle in needles):
            return category
    return "general_instruction"


def _normalize_conversations(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return []
        role = ROLE_MAP.get(normalize_text(item.get("from", item.get("role"))).lower())
        content = normalize_text(item.get("value", item.get("content")))
        if not role or not content:
            return []
        messages.append({"role": role, "content": content})
    return messages


def _alternates_and_ends_with_assistant(messages: list[dict[str, str]]) -> bool:
    non_system = [message for message in messages if message["role"] != "system"]
    if len(non_system) < 2 or non_system[0]["role"] != "user":
        return False
    if non_system[-1]["role"] != "assistant":
        return False
    return all(left["role"] != right["role"] for left, right in zip(non_system, non_system[1:]))


def convert_infinity_row(
    row: dict[str, Any],
    *,
    shard: InfinityShard,
    language: str,
) -> SFTRecord | None:
    normalized_language = normalize_infinity_language(row.get("langdetect"))
    if normalized_language != language:
        return None
    source_name = normalize_text(row.get("source"))
    if shard.allowed_sources and source_name not in shard.allowed_sources:
        return None
    try:
        reward = float(row.get("reward"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(reward) or reward < shard.reward_threshold(language):
        return None
    messages = _normalize_conversations(row.get("conversations"))
    if not _alternates_and_ends_with_assistant(messages):
        return None
    combined = "\n".join(message["content"] for message in messages)
    assistant_text = "\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )
    if (
        IDENTITY_CONTAMINATION.search(assistant_text)
        or HIGH_STAKES_OR_UNSTABLE.search(combined)
        or HARMFUL_INSTRUCTION.search(combined)
        or DATASET_ARTIFACT.search(combined)
        or BENCHMARK_PROMPT.search(combined)
    ):
        return None
    source_id = f"{shard.subset}:{normalize_text(row.get('id'))}"
    return SFTRecord(
        messages=messages,
        source=f"{shard.key}_{language}",
        category=infinity_category(row.get("label")),
        group_id=f"{shard.key}:{source_id}",
        source_id=source_id,
        metadata={
            "license": "CC-BY-SA-4.0",
            "reward": reward,
            "upstream_source": source_name,
            "language": language,
        },
    )


def read_infinity_records(
    raw_dir: Path,
    *,
    shard: InfinityShard,
    language: str,
    batch_size: int = 2_048,
) -> Iterator[SFTRecord]:
    path = raw_dir / shard.key / shard.subset / shard.filename
    parquet = pq.ParquetFile(path)
    columns = ["id", "conversations", "label", "langdetect", "source", "reward"]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            record = convert_infinity_row(row, shard=shard, language=language)
            if record is not None:
                yield record
