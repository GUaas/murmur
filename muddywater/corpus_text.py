from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any


ZERO_WIDTH_TRANSLATION = dict.fromkeys(
    map(
        ord,
        [
            "\ufeff",
            "\u200b",
            "\u200c",
            "\u200d",
            "\u200e",
            "\u200f",
            "\u2060",
        ],
    ),
    None,
)
SPACE_RE = re.compile(r"[ \t\u00a0\u3000]+")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
DEDUP_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>\n]{1,120}>")
CHUNK_BOUNDARY_CHARS = "\n。！？!?；;，,"


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.min_value: int | None = None
        self.max_value: int | None = None

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "mean": round(self.total / self.count, 4) if self.count else None,
        }


def clean_text(text: str, normalize_unicode: bool = True) -> str:
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    text = text.translate(ZERO_WIDTH_TRANSLATION)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    chars: list[str] = []
    for char in text:
        if char in {"\n", "\t"}:
            chars.append(char)
            continue
        if unicodedata.category(char).startswith("C"):
            chars.append(" ")
        else:
            chars.append(char)

    normalized_lines: list[str] = []
    for line in "".join(chars).split("\n"):
        line = SPACE_RE.sub(" ", line).strip()
        if line:
            normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()


def _choose_chunk_cut(text: str, max_len: int, min_len: int) -> int:
    limit = min(len(text), max_len)
    for index in range(limit, min_len - 1, -1):
        if text[index - 1] in CHUNK_BOUNDARY_CHARS:
            return index
    return limit


def split_long_text(text: str, max_len: int, min_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text] if len(text) >= min_len else []

    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_len:
        cut = _choose_chunk_cut(remaining, max_len=max_len, min_len=min_len)
        chunk = remaining[:cut].strip()
        if len(chunk) >= min_len:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()

    if len(remaining) >= min_len:
        chunks.append(remaining)
    elif remaining and chunks and len(chunks[-1]) + 1 + len(remaining) <= max_len:
        chunks[-1] = f"{chunks[-1]}\n{remaining}".strip()
    return chunks


def cjk_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    return len(CJK_RE.findall(text)) / len(visible)


def text_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def normalized_dedupe_text(text: str) -> str:
    return DEDUP_NORMALIZE_RE.sub("", text).lower()


def dedupe_digest(text: str, mode: str) -> bytes | None:
    if mode == "none":
        return None
    if mode == "exact":
        return text_digest(text)
    if mode == "normalized":
        return text_digest(normalized_dedupe_text(text))
    raise ValueError("dedupe mode must be one of: normalized, exact, none")


def _char_ngrams(text: str, n: int = 5) -> list[str]:
    text = normalized_dedupe_text(text)
    if not text:
        return []
    if len(text) <= n:
        return [text]
    return [text[index : index + n] for index in range(0, len(text) - n + 1)]


def simhash64(text: str) -> int:
    weights = Counter(_char_ngrams(text))
    vector = [0] * 64
    for feature, weight in weights.items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        for bit in range(64):
            vector[bit] += weight if (value >> bit) & 1 else -weight
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return int(left ^ right).bit_count()


class SimHashDeduper:
    def __init__(self, threshold: int = 3) -> None:
        self.threshold = max(0, int(threshold))
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    def _bucket_keys(self, value: int) -> list[tuple[int, int]]:
        return [(band, (value >> (band * 8)) & 0xFF) for band in range(8)]

    def is_duplicate(self, text: str) -> bool:
        value = simhash64(text)
        candidates: set[int] = set()
        for key in self._bucket_keys(value):
            candidates.update(self.buckets.get(key, []))
        if any(hamming_distance(value, candidate) <= self.threshold for candidate in candidates):
            return True
        for key in self._bucket_keys(value):
            self.buckets[key].append(value)
        return False


def quality_reject_reason(
    text: str,
    max_repeated_line_ratio: float,
    max_url_ratio: float,
    max_html_ratio: float,
    max_punctuation_ratio: float,
    max_digit_ratio: float,
) -> str | None:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return "empty_text_after_clean"
    visible_count = len(visible)
    text_len = max(1, len(text))

    url_ratio = sum(len(match.group(0)) for match in URL_RE.finditer(text)) / text_len
    if max_url_ratio >= 0 and url_ratio > max_url_ratio:
        return "high_url_ratio"

    html_ratio = sum(len(match.group(0)) for match in HTML_TAG_RE.finditer(text)) / text_len
    if max_html_ratio >= 0 and html_ratio > max_html_ratio:
        return "high_html_ratio"

    punctuation_ratio = (
        sum(1 for char in visible if unicodedata.category(char).startswith("P"))
        / visible_count
    )
    if max_punctuation_ratio >= 0 and punctuation_ratio > max_punctuation_ratio:
        return "high_punctuation_ratio"

    digit_ratio = sum(1 for char in visible if char.isdigit()) / visible_count
    if max_digit_ratio >= 0 and digit_ratio > max_digit_ratio:
        return "high_digit_ratio"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4 and max_repeated_line_ratio >= 0:
        counts = Counter(lines)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        if repeated / len(lines) > max_repeated_line_ratio:
            return "high_repeated_line_ratio"

    return None
