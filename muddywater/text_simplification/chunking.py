from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


TokenCounter = Callable[[str], int]

_HARD_ENDINGS = frozenset("。！？!?；;")
_CLOSING_MARKS = frozenset('"\'”’」』）》】]}〉')
_SOFT_ENDINGS = frozenset("，,、：:—- ")
_NEWLINES = frozenset("\r\n")
_COMMON_ASCII_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "vs.",
    "no.",
)


@dataclass(frozen=True)
class SentenceUnit:
    """A sentence-like unit and the exact whitespace that followed it."""

    text: str
    separator_after: str = ""


@dataclass(frozen=True)
class InferenceChunk:
    """A token-bounded source fragment that can be inferred independently."""

    source: str
    separator_after: str
    prompt_tokens: int
    sentence_count: int


@dataclass(frozen=True)
class ChunkPlan:
    """Lossless document structure plus independently inferable chunks."""

    leading_whitespace: str
    chunks: tuple[InferenceChunk, ...]


def _is_ascii_period_boundary(text: str, index: int) -> bool:
    if text[index] != ".":
        return False
    previous = text[index - 1] if index > 0 else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return False
    prefix = text[max(0, index - 5) : index + 1].lower()
    if any(prefix.endswith(abbreviation) for abbreviation in _COMMON_ASCII_ABBREVIATIONS):
        return False
    return not following or following.isspace() or following in _CLOSING_MARKS


def _is_ellipsis_boundary(text: str, index: int) -> bool:
    if text[index] != "…":
        return False
    previous_same = index > 0 and text[index - 1] == "…"
    following_same = index + 1 < len(text) and text[index + 1] == "…"
    following = text[index + 1] if index + 1 < len(text) else ""
    return previous_same or following_same or not following or following.isspace()


def _is_sentence_boundary(text: str, index: int) -> bool:
    char = text[index]
    return (
        char in _HARD_ENDINGS
        or _is_ascii_period_boundary(text, index)
        or _is_ellipsis_boundary(text, index)
    )


def _consume_line_break(text: str, index: int) -> int:
    if text[index : index + 2] == "\r\n":
        return index + 2
    return index + 1


def split_sentences(text: str) -> tuple[str, tuple[SentenceUnit, ...]]:
    """Split Chinese text without losing punctuation, spaces, or paragraph breaks.

    Chinese/English sentence endings and line breaks are treated as boundaries.
    Decimal points such as ``3.14`` and dots inside URLs are not split.
    """

    source = str(text)
    length = len(source)
    cursor = 0
    while cursor < length and source[cursor].isspace():
        cursor += 1
    leading_whitespace = source[:cursor]
    units: list[SentenceUnit] = []

    while cursor < length:
        start = cursor
        boundary_end = length
        while cursor < length:
            if source[cursor] in _NEWLINES:
                boundary_end = cursor
                break
            if _is_sentence_boundary(source, cursor):
                cursor += 1
                while cursor < length and (
                    _is_sentence_boundary(source, cursor)
                    or source[cursor] in _CLOSING_MARKS
                ):
                    cursor += 1
                boundary_end = cursor
                break
            cursor += 1

        raw_unit = source[start:boundary_end]
        core = raw_unit.rstrip()
        separator = raw_unit[len(core) :]
        cursor = boundary_end
        if cursor < length and source[cursor] in _NEWLINES:
            cursor = _consume_line_break(source, cursor)
            separator += source[boundary_end:cursor]
        while cursor < length and source[cursor].isspace():
            whitespace_start = cursor
            if source[cursor] in _NEWLINES:
                cursor = _consume_line_break(source, cursor)
            else:
                cursor += 1
            separator += source[whitespace_start:cursor]

        if core:
            units.append(SentenceUnit(text=core, separator_after=separator))
        elif units:
            previous = units[-1]
            units[-1] = SentenceUnit(
                text=previous.text,
                separator_after=previous.separator_after + raw_unit + separator,
            )
        else:
            leading_whitespace += raw_unit + separator

    return leading_whitespace, tuple(units)


def reconstruct_source(leading_whitespace: str, units: Iterable[SentenceUnit]) -> str:
    """Rebuild the exact input, primarily useful for validation and tests."""

    return leading_whitespace + "".join(unit.text + unit.separator_after for unit in units)


def _soft_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in _SOFT_ENDINGS:
            fragments.append(text[start : index + 1])
            start = index + 1
    if start < len(text):
        fragments.append(text[start:])
    return [fragment for fragment in fragments if fragment]


def _largest_fitting_prefix(text: str, token_count: TokenCounter, max_tokens: int) -> int:
    # Start near the expected fit and expand only when needed.  Binary-searching
    # the full remaining string on every split makes a punctuation-free document
    # quadratic in practice because tokenization repeatedly scans huge suffixes.
    low = 1
    high = min(len(text), max(8, max_tokens))
    best = 0
    while high < len(text) and token_count(text[:high]) <= max_tokens:
        best = high
        low = high + 1
        high = min(len(text), high * 2)
    while low <= high:
        middle = (low + high) // 2
        if token_count(text[:middle]) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best <= 0:
        raise ValueError(
            "max_prompt_tokens is too small to hold even one source character and the task prompt"
        )

    search_floor = max(1, int(best * 0.6))
    for index in range(best - 1, search_floor - 1, -1):
        if text[index] in _SOFT_ENDINGS:
            return index + 1
    return best


def _split_oversized_source(
    text: str,
    token_count: TokenCounter,
    max_tokens: int,
) -> list[str]:
    if token_count(text) <= max_tokens:
        return [text]

    pieces: list[str] = []
    pending = ""
    for fragment in _soft_fragments(text):
        candidate = pending + fragment
        if pending and token_count(candidate) > max_tokens:
            pieces.append(pending)
            pending = ""
        if token_count(fragment) <= max_tokens:
            pending += fragment
            continue

        if pending:
            pieces.append(pending)
            pending = ""
        remainder = fragment
        while remainder:
            cut_at = _largest_fitting_prefix(remainder, token_count, max_tokens)
            if cut_at == len(remainder):
                pending = remainder
                break
            pieces.append(remainder[:cut_at])
            remainder = remainder[cut_at:]

    if pending:
        pieces.append(pending)
    return pieces


def _expand_oversized_units(
    units: Iterable[SentenceUnit],
    token_count: TokenCounter,
    max_tokens: int,
) -> list[tuple[SentenceUnit, int]]:
    expanded: list[tuple[SentenceUnit, int]] = []
    for unit in units:
        pieces = _split_oversized_source(unit.text, token_count, max_tokens)
        for index, piece in enumerate(pieces):
            separator = unit.separator_after if index == len(pieces) - 1 else ""
            expanded.append(
                (SentenceUnit(text=piece, separator_after=separator), int(index == 0))
            )
    return expanded


def plan_inference_chunks(
    text: str,
    token_count: TokenCounter,
    max_prompt_tokens: int = 160,
) -> ChunkPlan:
    """Create paragraph-aware chunks whose rendered prompts fit the token budget."""

    if isinstance(max_prompt_tokens, bool) or int(max_prompt_tokens) < 8:
        raise ValueError("max_prompt_tokens must be an integer of at least 8")
    max_prompt_tokens = int(max_prompt_tokens)
    leading, units = split_sentences(text)
    expanded = _expand_oversized_units(units, token_count, max_prompt_tokens)
    chunks: list[InferenceChunk] = []
    active_source = ""
    active_separator = ""
    active_sentence_count = 0

    def flush() -> None:
        nonlocal active_source, active_separator, active_sentence_count
        if not active_source:
            return
        chunks.append(
            InferenceChunk(
                source=active_source,
                separator_after=active_separator,
                prompt_tokens=token_count(active_source),
                sentence_count=max(1, active_sentence_count),
            )
        )
        active_source = ""
        active_separator = ""
        active_sentence_count = 0

    for unit, sentence_increment in expanded:
        if not active_source:
            active_source = unit.text
            active_separator = unit.separator_after
            active_sentence_count = sentence_increment
            continue

        paragraph_boundary = "\n" in active_separator or "\r" in active_separator
        candidate = active_source + active_separator + unit.text
        if not paragraph_boundary and token_count(candidate) <= max_prompt_tokens:
            active_source = candidate
            active_separator = unit.separator_after
            active_sentence_count += sentence_increment
        else:
            flush()
            active_source = unit.text
            active_separator = unit.separator_after
            active_sentence_count = sentence_increment
    flush()
    return ChunkPlan(leading_whitespace=leading, chunks=tuple(chunks))
