from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import sentencepiece as spm

from muddywater.templates import format_messages

from .records import SFTRecord, normalize_messages


VISIBLE_THINK_PATTERN = re.compile(r"<\/?think>|<\/?analysis>", re.IGNORECASE)
ASSISTANT_IDENTITY_PATTERN = re.compile(
    r"(?:作为|身为).{0,12}(?:AI|人工智能|语言模型)"
    r"|(?:我是|我只是|我虽然是).{0,8}(?:AI|人工智能|语言模型)"
    r"|我(?:并)?没有(?:真实的)?(?:感情|情感)"
    r"|开源语言模型",
    re.IGNORECASE,
)
FOREIGN_CHAT_MARKER_PATTERN = re.compile(
    r"<tool_(?:call|response)>|<\|im_(?:start|end)\|>",
    re.IGNORECASE,
)
EMBEDDED_ASSISTANT_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:assistant|助手|助理)\s*(?::|：|\n)",
    re.IGNORECASE,
)
UNEXPECTED_SCRIPT_PATTERN = re.compile(
    r"[\u0400-\u04ff\u0600-\u06ff\u0e00-\u0e7f"
    r"\u3040-\u30ff\uac00-\ud7af\ufb50-\ufdff]"
)
REPEATED_LATIN_WORD_PATTERN = re.compile(r"\b([A-Za-z]{3,})\b(?:\s+\1){3,}", re.IGNORECASE)
EVERYDAY_GOAL_LEAK_PATTERN = re.compile(r"(?:^|[\s，。！？])goal(?:[!\s，。！？]|$)", re.IGNORECASE)
SMOLTALK_TEMPLATE_ARTIFACT_PATTERN = re.compile(
    r"_measurement|_(?:fieldName|value)\d*|GetUserProblemByUid|parentId|divisbble"
    r"|(?<=[\u3400-\u9fff])\s+intersections\s+(?=[\u3400-\u9fff])"
    r"|(?:^|\n)\s*(?:disgusting|horrible|terrible)\s+merger\b"
    r"|(?:不相关|无关)(?:的)?(?:主题|内容).{0,24}(?:忽略|无视)"
    r"|(?:忽略|无视).{0,24}(?:不相关|无关)(?:的)?(?:主题|内容)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class FilterStats:
    accepted: int = 0
    rejected: Counter[str] = field(default_factory=Counter)


class RecordFilter:
    def __init__(
        self,
        tokenizer_path: str,
        max_tokens: int = 2048,
        min_assistant_tokens: int = 2,
    ) -> None:
        try:
            self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        except OSError:
            # SentencePiece's Windows file loader may reject otherwise valid
            # paths containing non-ASCII user or workspace names. Loading the
            # same model bytes avoids changing or copying the user's path.
            self.tokenizer = spm.SentencePieceProcessor(
                model_proto=Path(tokenizer_path).read_bytes()
            )
        self.max_tokens = int(max_tokens)
        self.min_assistant_tokens = int(min_assistant_tokens)
        self.stats = FilterStats()

    def apply(self, record: SFTRecord) -> SFTRecord | None:
        messages = normalize_messages(record.messages)
        reason = self._structural_rejection(messages)
        if reason:
            self.stats.rejected[reason] += 1
            return None
        raw_content = "\n".join(msg["content"] for msg in messages)
        transcript = format_messages(messages, chat_template="chatml")
        if VISIBLE_THINK_PATTERN.search(raw_content):
            self.stats.rejected["visible_think_tags"] += 1
            return None
        if FOREIGN_CHAT_MARKER_PATTERN.search(raw_content):
            self.stats.rejected["foreign_chat_marker"] += 1
            return None
        user_text = "\n".join(msg["content"] for msg in messages if msg["role"] == "user")
        if record.source == "smoltalk_chinese" and EMBEDDED_ASSISTANT_PATTERN.search(user_text):
            self.stats.rejected["embedded_assistant_text"] += 1
            return None
        if (
            record.source == "smoltalk_chinese"
            and record.category != "translate"
            and UNEXPECTED_SCRIPT_PATTERN.search(raw_content)
        ):
            self.stats.rejected["unexpected_script_mixing"] += 1
            return None
        if REPEATED_LATIN_WORD_PATTERN.search(raw_content):
            self.stats.rejected["repeated_latin_noise"] += 1
            return None
        if (
            record.source == "smoltalk_chinese"
            and record.category == "everyday"
            and EVERYDAY_GOAL_LEAK_PATTERN.search(raw_content)
        ):
            self.stats.rejected["everyday_goal_leak"] += 1
            return None
        if record.source == "smoltalk_chinese" and SMOLTALK_TEMPLATE_ARTIFACT_PATTERN.search(raw_content):
            self.stats.rejected["smoltalk_template_artifact"] += 1
            return None
        if "\ufffd" in raw_content:
            self.stats.rejected["replacement_character"] += 1
            return None
        if any(ord(char) < 32 and char not in "\n\t" for char in raw_content):
            self.stats.rejected["control_character"] += 1
            return None
        assistant_text = "\n".join(msg["content"] for msg in messages if msg["role"] == "assistant")
        if (
            record.source != "synthetic_identity_verified"
            and ASSISTANT_IDENTITY_PATTERN.search(assistant_text)
        ):
            self.stats.rejected["assistant_model_identity"] += 1
            return None
        token_ids = self.tokenizer.encode(transcript, out_type=int)
        if len(token_ids) > self.max_tokens:
            self.stats.rejected["over_max_tokens"] += 1
            return None
        assistant_tokens = len(self.tokenizer.encode(assistant_text, out_type=int))
        if assistant_tokens < self.min_assistant_tokens:
            self.stats.rejected["assistant_too_short"] += 1
            return None
        record.messages = messages
        record.num_tokens = len(token_ids)
        record.assistant_tokens = assistant_tokens
        self.stats.accepted += 1
        return record

    @staticmethod
    def _structural_rejection(messages: list[dict[str, str]]) -> str | None:
        if len(messages) < 2:
            return "too_few_messages"
        if messages[-1]["role"] != "assistant":
            return "does_not_end_assistant"
        non_system = [message for message in messages if message["role"] != "system"]
        if not non_system or non_system[0]["role"] != "user":
            return "does_not_start_user"
        for previous, current in zip(non_system, non_system[1:]):
            if previous["role"] == current["role"]:
                return "roles_not_alternating"
        total_chars = sum(len(message["content"]) for message in messages)
        if total_chars < 12:
            return "too_short"
        if any(len(message["content"]) > 12_000 for message in messages):
            return "single_message_too_long"
        return None
