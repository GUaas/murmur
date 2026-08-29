from __future__ import annotations

import re
import unicodedata
from typing import Any


# Case-scoped semantic equivalents for constraints whose literal surface form
# may change under valid simplification. These aliases apply equally to both
# releases and never relax numeric/entity constraints.
CASE_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "negation_01": {
        "不能": ("不能", "未证明", "无法证明", "不能证明", "不能说明"),
    },
    "negation_03": {
        "没有证据": ("没有证据", "无证据"),
    },
    "long_01": {"不会取消": ("不会取消", "不取消")},
    "long_02": {"不会取消": ("不会取消", "不取消")},
    "long_03": {"不会取消": ("不会取消", "不取消")},
    "long_04": {"不会取消": ("不会取消", "不取消")},
}


def normalize_constraint_text(text: str) -> str:
    """Normalize compatibility forms and remove formatting-only separators."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", normalized)


def constraint_item_kept(case_id: str, required: str, prediction: str) -> tuple[bool, str]:
    normalized_prediction = normalize_constraint_text(prediction)
    candidates = CASE_ALIASES.get(case_id, {}).get(required, (required,))
    for candidate in candidates:
        if normalize_constraint_text(candidate) in normalized_prediction:
            mode = "exact" if candidate == required and required in prediction else "normalized_or_alias"
            return True, mode
    return False, "missing"


def semantic_constraint_audit(record: dict[str, Any]) -> dict[str, Any]:
    required_items = list(record.get("constraints", {}).get("must_keep_missing", []))
    strict_total = int(record.get("constraints", {}).get("must_keep_total", 0))
    if strict_total:
        # The raw scorer stores only missing items. Recover present requirements
        # from the stress record's strict pass result by auditing the known
        # missing items and treating the remainder as exact hits.
        strict_missing = set(required_items)
        semantic_missing = []
        recovered = []
        for required in required_items:
            kept, mode = constraint_item_kept(
                str(record.get("case_id", "")),
                required,
                str(record.get("prediction", "")),
            )
            if kept:
                recovered.append({"required": required, "mode": mode})
            else:
                semantic_missing.append(required)
        semantic_pass = not semantic_missing
    else:
        strict_missing = set()
        semantic_missing = []
        recovered = []
        semantic_pass = True
    return {
        "must_keep_total": strict_total,
        "strict_missing": sorted(strict_missing),
        "semantic_missing": semantic_missing,
        "recovered_by_normalization_or_alias": recovered,
        "strict_pass": bool(record.get("constraints", {}).get("must_keep_pass", True)),
        "semantic_pass": semantic_pass,
    }
