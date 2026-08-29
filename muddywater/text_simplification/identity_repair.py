from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Iterable, Mapping


POLITICAL_KEYWORD_GROUPS: Mapping[str, tuple[str, ...]] = {
    "party_state": (
        "共产党",
        "中共中央",
        "政治局",
        "党委",
        "党组织",
        "毛泽东",
        "习近平",
        "邓小平",
        "社会主义",
        "共产主义",
        "马克思",
    ),
    "government_law": (
        "国务院",
        "全国人大",
        "政府",
        "人民法院",
        "国家主席",
        "总理",
        "法律",
        "条例",
        "法院",
        "司法",
    ),
    "foreign_sensitive": (
        "台湾",
        "香港",
        "澳门",
        "美国",
        "苏联",
        "外交",
        "殖民主义",
        "帝国主义",
    ),
    "military": ("解放军", "军队", "全军", "国防", "军事"),
}

NUMBER_PATTERN = re.compile(r"\d+(?:[.:：/\-—]\d+)*%?")
SPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_PATTERN = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SENTENCE_ENDINGS = frozenset("。！？.!?；;）)]】》\"'”’")


@dataclass(frozen=True)
class PairRecord:
    split: str
    line_number: int
    source: str
    target: str

    @property
    def key(self) -> str:
        return f"{self.split}:{self.line_number}"

    @property
    def is_identity(self) -> bool:
        return self.source == self.target


@dataclass(frozen=True)
class QualityConfig:
    min_output_chars: int = 2
    min_length_ratio: float = 0.35
    max_length_ratio: float = 1.05
    min_similarity: float = 0.40
    incomplete_check_min_chars: int = 40


@dataclass(frozen=True)
class QualityDecision:
    accepted: bool
    candidate: str
    reasons: tuple[str, ...]
    source_chars: int
    output_chars: int
    length_ratio: float
    similarity: float
    source_numbers: tuple[str, ...]
    output_numbers: tuple[str, ...]
    truncated: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def compact_text(text: str) -> str:
    return SPACE_PATTERN.sub("", normalize_text(text))


def semantic_surface(text: str) -> str:
    return PUNCTUATION_PATTERN.sub("", compact_text(text)).lower()


def extract_numbers(text: str) -> tuple[str, ...]:
    return tuple(NUMBER_PATTERN.findall(normalize_text(text)))


def topic_flags(text: str) -> tuple[str, ...]:
    source = str(text)
    return tuple(
        group
        for group, keywords in POLITICAL_KEYWORD_GROUPS.items()
        if any(keyword in source for keyword in keywords)
    )


def _numbers_preserved(source_numbers: Iterable[str], output_numbers: Iterable[str]) -> bool:
    source_counter = Counter(source_numbers)
    output_counter = Counter(output_numbers)
    return all(output_counter[number] >= count for number, count in source_counter.items())


def _looks_incomplete(source: str, candidate: str, minimum_source_chars: int) -> bool:
    stripped_source = source.rstrip()
    stripped_candidate = candidate.rstrip()
    if len(stripped_source) < minimum_source_chars or not stripped_source:
        return False
    if stripped_source[-1] not in SENTENCE_ENDINGS:
        return False
    return not stripped_candidate or stripped_candidate[-1] not in SENTENCE_ENDINGS


def evaluate_candidate(
    source: str,
    candidate: str,
    *,
    truncated: bool = False,
    config: QualityConfig | None = None,
) -> QualityDecision:
    quality = config or QualityConfig()
    normalized_source = normalize_text(source)
    normalized_candidate = normalize_text(candidate)
    source_surface = semantic_surface(normalized_source)
    candidate_surface = semantic_surface(normalized_candidate)
    source_chars = len(compact_text(normalized_source))
    output_chars = len(compact_text(normalized_candidate))
    length_ratio = output_chars / source_chars if source_chars else 0.0
    similarity = (
        SequenceMatcher(None, source_surface, candidate_surface, autojunk=False).ratio()
        if source_surface and candidate_surface
        else 0.0
    )
    source_numbers = extract_numbers(normalized_source)
    output_numbers = extract_numbers(normalized_candidate)
    reasons: list[str] = []

    if not normalized_candidate:
        reasons.append("empty_output")
    if normalized_candidate == normalized_source:
        reasons.append("model_kept_identity")
    if source_surface == candidate_surface and normalized_candidate != normalized_source:
        reasons.append("punctuation_or_spacing_only")
    if output_chars < quality.min_output_chars:
        reasons.append("too_short")
    if source_chars and length_ratio < quality.min_length_ratio:
        reasons.append("over_compressed")
    if source_chars and length_ratio > quality.max_length_ratio:
        reasons.append("longer_than_allowed")
    if similarity < quality.min_similarity:
        reasons.append("low_surface_similarity")
    if not _numbers_preserved(source_numbers, output_numbers):
        reasons.append("number_loss")
    if truncated:
        reasons.append("generation_length_limit")
    if CONTROL_PATTERN.search(normalized_candidate):
        reasons.append("control_character")
    if _looks_incomplete(
        normalized_source,
        normalized_candidate,
        quality.incomplete_check_min_chars,
    ):
        reasons.append("incomplete_ending")

    return QualityDecision(
        accepted=not reasons,
        candidate=normalized_candidate,
        reasons=tuple(reasons),
        source_chars=source_chars,
        output_chars=output_chars,
        length_ratio=length_ratio,
        similarity=similarity,
        source_numbers=source_numbers,
        output_numbers=output_numbers,
        truncated=bool(truncated),
    )


def analyze_identity_rates(records: Iterable[PairRecord]) -> dict[str, object]:
    rows = list(records)
    identity_total = sum(record.is_identity for record in rows)
    groups: dict[str, object] = {}
    for group in POLITICAL_KEYWORD_GROUPS:
        matched = [record for record in rows if group in topic_flags(record.source)]
        unmatched = [record for record in rows if group not in topic_flags(record.source)]
        matched_identity = sum(record.is_identity for record in matched)
        unmatched_identity = sum(record.is_identity for record in unmatched)
        matched_rate = matched_identity / len(matched) if matched else 0.0
        unmatched_rate = unmatched_identity / len(unmatched) if unmatched else 0.0
        groups[group] = {
            "matched": len(matched),
            "matched_identity": matched_identity,
            "matched_identity_rate": matched_rate,
            "unmatched": len(unmatched),
            "unmatched_identity": unmatched_identity,
            "unmatched_identity_rate": unmatched_rate,
            "risk_ratio": matched_rate / unmatched_rate if unmatched_rate else None,
        }
    any_political = [record for record in rows if topic_flags(record.source)]
    non_political = [record for record in rows if not topic_flags(record.source)]
    any_identity = sum(record.is_identity for record in any_political)
    non_identity = sum(record.is_identity for record in non_political)
    return {
        "rows": len(rows),
        "identity_rows": identity_total,
        "identity_rate": identity_total / len(rows) if rows else 0.0,
        "any_political": {
            "matched": len(any_political),
            "matched_identity": any_identity,
            "matched_identity_rate": any_identity / len(any_political) if any_political else 0.0,
            "unmatched": len(non_political),
            "unmatched_identity": non_identity,
            "unmatched_identity_rate": non_identity / len(non_political) if non_political else 0.0,
            "risk_ratio": (
                (any_identity / len(any_political)) / (non_identity / len(non_political))
                if any_political and non_political and non_identity
                else None
            ),
        },
        "groups": groups,
    }
