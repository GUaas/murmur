from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any, Sequence

from .paths import EvaluationPaths


PROTOCOL_VERSION = "2026-08-15.v1"
SEED = 20260815
DEFAULT_VALIDATION_SIZE = 500


@dataclass(frozen=True)
class ValidationItem:
    dataset_index: int
    source: str
    target: str


def prepare_imports(paths: EvaluationPaths) -> None:
    for import_path in (paths.v2_dir, paths.v2_source):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))


def load_stress_cases(paths: EvaluationPaths) -> list[Any]:
    prepare_imports(paths)
    from extreme_eval.cases import build_stress_cases

    return build_stress_cases()


def load_validation_sample(
    paths: EvaluationPaths,
    sample_size: int = DEFAULT_VALIDATION_SIZE,
) -> list[ValidationItem]:
    prepare_imports(paths)
    from muddywater.text_simplification.evaluation import load_pairs

    pairs = load_pairs(paths.validation_file)
    if sample_size <= 0 or sample_size > len(pairs):
        sample_size = len(pairs)
    indices = sorted(random.Random(SEED).sample(range(len(pairs)), sample_size))
    return [
        ValidationItem(
            dataset_index=index,
            source=pairs[index].source,
            target=pairs[index].target,
        )
        for index in indices
    ]


def protocol_manifest(
    stress_cases: Sequence[Any],
    validation_items: Sequence[ValidationItem],
) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    for case in stress_cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1
    return {
        "version": PROTOCOL_VERSION,
        "seed": SEED,
        "stress_count": len(stress_cases),
        "stress_category_counts": dict(sorted(category_counts.items())),
        "stress_provenance": (
            "Human-authored fixed cases plus deterministic perturbations; "
            "not drawn from either packaged training split."
        ),
        "validation_count": len(validation_items),
        "validation_indices": [item.dataset_index for item in validation_items],
        "validation_warning": (
            "This split was used by version 2 during checkpoint selection. "
            "It is reported as an auxiliary diagnostic and excluded from the overall score."
        ),
        "inference_policy": (
            "Each release uses its shipped production defaults, including its native long-text "
            "chunking and decoding policy. All measurements run sequentially on CPU with eight threads."
        ),
    }
