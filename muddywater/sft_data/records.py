from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


ROLE_ALIASES = {
    "human": "user",
    "prompter": "user",
    "usr": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "sys": "assistant",
    "system": "system",
}

HAN_RANGE = "\\u3400-\\u9fff"
HAN_INTERNAL_SPACE = re.compile(rf"(?<=[{HAN_RANGE}]) +(?=[{HAN_RANGE}])")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    text = HAN_INTERNAL_SPACE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for message in messages:
        role = ROLE_ALIASES.get(str(message.get("role", "")).strip().lower())
        content = normalize_text(message.get("content", message.get("message", "")))
        if role and content:
            output.append({"role": role, "content": content})
    return output


def stable_hex(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


@dataclass(slots=True)
class SFTRecord:
    messages: list[dict[str, str]]
    source: str
    category: str
    group_id: str
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    num_tokens: int = 0
    assistant_tokens: int = 0

    @property
    def content_key(self) -> str:
        compact = json.dumps(self.messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return stable_hex(compact, 64)

    @property
    def record_id(self) -> str:
        basis = f"{self.source}\n{self.source_id}\n{self.content_key}"
        return stable_hex(basis, 24)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "messages": self.messages,
            "source": self.source,
            "category": self.category,
            "group_id": self.group_id,
            "num_tokens": self.num_tokens,
            "assistant_tokens": self.assistant_tokens,
        }
