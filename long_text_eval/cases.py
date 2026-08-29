from __future__ import annotations

from collections import defaultdict
from itertools import cycle
from typing import Iterable, Sequence

from extreme_eval.cases import build_stress_cases
from extreme_eval.types import EvalCase

from .types import LongDocument


_EXCLUDED_CATEGORIES = {"identity", "injection", "long_context", "perturbation"}


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _length_tier(sentence_count: int) -> str:
    if sentence_count <= 8:
        return "short_long"
    if sentence_count <= 16:
        return "medium_long"
    if sentence_count <= 32:
        return "long"
    return "extreme_long"


def _join_layout(texts: Sequence[str], layout: str) -> str:
    if layout == "continuous":
        return "".join(texts)
    if layout == "line_per_sentence":
        return "\n".join(texts)
    if layout == "paragraphs":
        paragraphs = ["".join(texts[index : index + 4]) for index in range(0, len(texts), 4)]
        return "\n\n".join(paragraphs)
    if layout == "mixed":
        groups = ["".join(texts[index : index + 3]) for index in range(0, len(texts), 3)]
        return "\n".join(groups)
    raise ValueError(f"Unknown layout: {layout}")


def _eligible_cases() -> list[EvalCase]:
    return [
        case
        for case in build_stress_cases()
        if case.category not in _EXCLUDED_CATEGORIES and "__" not in case.case_id
    ]


def _select_cases(
    pools: dict[str, list[EvalCase]],
    categories: Sequence[str],
    count: int,
    offset: int,
) -> list[EvalCase]:
    available = [case for category in categories for case in pools.get(category, [])]
    if not available:
        raise ValueError(f"No cases for categories: {categories}")
    rotated = available[offset % len(available) :] + available[: offset % len(available)]
    picker = cycle(rotated)
    return [next(picker) for _ in range(count)]


def _composite_documents() -> list[LongDocument]:
    eligible = _eligible_cases()
    pools: dict[str, list[EvalCase]] = defaultdict(list)
    by_id = {case.case_id: case for case in eligible}
    for case in eligible:
        pools[case.category].append(case)

    specs = [
        ("composite_01", "formal_news", ("formal", "news"), 6, "continuous", "num_03"),
        ("composite_02", "technical_mixed", ("technical", "mixed"), 8, "line_per_sentence", "mixed_02"),
        ("composite_03", "numbers", ("numbers",), 10, "paragraphs", "num_06"),
        ("composite_04", "spoken", ("spoken",), 12, "mixed", "num_09"),
        ("composite_05", "legal_negation", ("legal", "negation"), 16, "paragraphs", "negation_02"),
        ("composite_06", "formal_academic", ("formal", "academic"), 20, "continuous", "num_10"),
        ("composite_07", "technical_numbers", ("technical", "numbers"), 24, "line_per_sentence", "num_04"),
        ("composite_08", "news_entity_mixed", ("news", "entity", "mixed"), 28, "paragraphs", "entity_01"),
        ("composite_09", "cross_domain", tuple(pools), 32, "mixed", "num_07"),
        ("composite_10", "cross_domain", tuple(pools), 40, "continuous", "negation_03"),
        ("composite_11", "cross_domain", tuple(pools), 48, "line_per_sentence", "num_08"),
        ("composite_12", "cross_domain", tuple(pools), 60, "paragraphs", "num_01"),
    ]

    documents: list[LongDocument] = []
    for spec_index, (doc_id, category, categories, count, layout, anchor_id) in enumerate(specs):
        anchor = by_id[anchor_id]
        selected = _select_cases(pools, categories, count - 1, offset=spec_index * 3)
        selected.append(anchor)
        source = _join_layout([case.source for case in selected], layout)
        target = _join_layout([case.target for case in selected], layout)
        documents.append(
            LongDocument(
                document_id=doc_id,
                category=category,
                source=source,
                target=target,
                sentence_count=len(selected),
                layout=layout,
                length_tier=_length_tier(len(selected)),
                must_keep=_unique(item for case in selected for item in case.must_keep),
                tail_keep=_unique(anchor.must_keep),
                provenance=(
                    "Deterministic composite assembled from fixed authored stress cases; "
                    "no packaged train or validation rows."
                ),
            )
        )
    return documents


def _legacy_long_documents() -> list[LongDocument]:
    cases = [case for case in build_stress_cases() if case.category == "long_context"]
    documents = []
    for case in cases:
        repeats = int(case.case_id.rsplit("_", 1)[-1])
        documents.append(
            LongDocument(
                document_id=case.case_id,
                category="repetition_regression",
                source=case.source,
                target=case.target,
                sentence_count=(4, 8, 16, 28)[repeats - 1] + 1,
                layout="continuous",
                length_tier=("short_long", "medium_long", "long", "extreme_long")[repeats - 1],
                must_keep=case.must_keep,
                tail_keep=case.must_keep,
                provenance="Fixed independent long-context regression from the original extreme evaluation.",
            )
        )
    return documents


def build_long_documents() -> list[LongDocument]:
    """Return fixed long-document cases without reading packaged train/validation data."""

    return _legacy_long_documents() + _composite_documents()
