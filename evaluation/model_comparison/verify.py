from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .adapters import sha256_file
from .paths import EvaluationPaths
from .scoring import score_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify comparison artifacts and pairing invariants.")
    parser.add_argument("--output-dir", default="model_comparison_results")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_finite_records(records: list[dict[str, Any]], label: str) -> None:
    metric_keys = ("sari", "rouge_l_f1", "chrf", "compression_ratio", "latency_ms")
    for index, record in enumerate(records):
        for key in metric_keys:
            value = float(record[key])
            if not math.isfinite(value):
                raise AssertionError(f"{label}[{index}].{key} is not finite")


def verify(paths: EvaluationPaths) -> dict[str, Any]:
    v1 = _load(paths.raw_dir / "v1_results.json")
    v2 = _load(paths.raw_dir / "v2_results.json")
    if v1["protocol"] != v2["protocol"]:
        raise AssertionError("The two model runs do not share an identical protocol manifest")

    v1_stress_ids = [row["case_id"] for row in v1["stress"]["records"]]
    v2_stress_ids = [row["case_id"] for row in v2["stress"]["records"]]
    if v1_stress_ids != v2_stress_ids or len(set(v1_stress_ids)) != 135:
        raise AssertionError("Stress case pairing/count is invalid")

    v1_validation_ids = [row["dataset_index"] for row in v1["validation"]["records"]]
    v2_validation_ids = [row["dataset_index"] for row in v2["validation"]["records"]]
    if v1_validation_ids != v2_validation_ids or len(set(v1_validation_ids)) != 500:
        raise AssertionError("Validation case pairing/count is invalid")

    for result in (v1, v2):
        _assert_finite_records(result["stress"]["records"], f"{result['model_id']}.stress")
        _assert_finite_records(result["validation"]["records"], f"{result['model_id']}.validation")
        if any(row["empty"] for row in result["stress"]["records"]):
            raise AssertionError(f"Unexpected empty stress output from {result['model_id']}")
        weight_path = Path(result["runtime"]["weight_path"])
        if sha256_file(weight_path) != result["runtime"]["weight_sha256"]:
            raise AssertionError(f"Weight checksum changed for {result['model_id']}")

    scores = score_models([v1, v2])
    for model in scores["models"].values():
        if not 0.0 <= float(model["overall"]) <= 100.0:
            raise AssertionError("Overall score outside [0, 100]")
        for dimension in model["dimensions"].values():
            if not 0.0 <= float(dimension["score"]) <= 100.0:
                raise AssertionError("Dimension score outside [0, 100]")

    required_artifacts = (
        paths.output_dir / "comparison_report.md",
        paths.tables_dir / "final_scorecard.csv",
        paths.tables_dir / "category_metrics.csv",
        paths.tables_dir / "constraint_semantic_review.csv",
        paths.raw_dir / "comparison_summary.json",
    )
    missing = [str(path) for path in required_artifacts if not path.is_file()]
    if missing:
        raise AssertionError("Missing final artifacts: " + ", ".join(missing))

    return {
        "status": "passed",
        "protocol_version": v1["protocol"]["version"],
        "paired_stress_cases": len(v1_stress_ids),
        "paired_validation_cases": len(v1_validation_ids),
        "empty_stress_outputs": {"v1": 0, "v2": 0},
        "determinism": {
            "v1": v1["determinism"]["exact_determinism_rate"],
            "v2": v2["determinism"]["exact_determinism_rate"],
        },
        "weight_sha256_verified": True,
        "score_bounds_verified": True,
    }


def main() -> None:
    paths = EvaluationPaths.discover(parse_args().output_dir)
    payload = verify(paths)
    output_path = paths.raw_dir / "verification.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
