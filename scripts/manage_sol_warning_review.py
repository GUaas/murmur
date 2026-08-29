from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from muddywater.text_simplification.identity_repair import extract_numbers
from muddywater.text_simplification.sol_distillation import read_jsonl, write_jsonl


REVIEW_DECISIONS = frozenset({"accept", "revert", "revise"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or apply GPT-5.6-sol warning reviews.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--review-file", type=Path, required=True)
    prepare.add_argument("--review-dir", type=Path, required=True)
    prepare.add_argument("--shards", type=int, default=3)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--work-dir", type=Path, required=True)
    apply.add_argument("--review-dir", type=Path, required=True)
    return parser.parse_args()


def prepare_review(review_file: Path, review_dir: Path, shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("shards must be positive")
    rows = read_jsonl(review_file)
    review_dir.mkdir(parents=True, exist_ok=True)
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, row in enumerate(rows):
        partitions[index % shard_count].append(
            {
                "key": row["key"],
                "source": row["source"],
                "candidate_target": row["target"],
                "warnings": row["warnings"],
            }
        )
    for index, partition in enumerate(partitions):
        write_jsonl(review_dir / f"review_input_{index:03d}.jsonl", partition)
    manifest = {
        "teacher_model": "gpt-5.6-sol",
        "rows": len(rows),
        "shards": [
            {
                "input": f"review_input_{index:03d}.jsonl",
                "output": f"review_output_{index:03d}.jsonl",
                "rows": len(partition),
            }
            for index, partition in enumerate(partitions)
        ],
    }
    (review_dir / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def validate_reviews(review_dir: Path) -> dict[str, dict[str, str]]:
    manifest = json.loads((review_dir / "review_manifest.json").read_text(encoding="utf-8-sig"))
    accepted: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for shard in manifest["shards"]:
        inputs = read_jsonl(review_dir / shard["input"])
        outputs = read_jsonl(review_dir / shard["output"])
        if len(inputs) != len(outputs):
            errors.append(f"{shard['output']}: expected {len(inputs)} rows, got {len(outputs)}")
            continue
        for input_row, output_row in zip(inputs, outputs, strict=True):
            key = str(input_row["key"])
            decision = str(output_row.get("decision", ""))
            target = output_row.get("target")
            if str(output_row.get("key", "")) != key:
                errors.append(f"{key}: key/order mismatch")
                continue
            if decision not in REVIEW_DECISIONS or not isinstance(target, str) or not target.strip():
                errors.append(f"{key}: invalid decision or target")
                continue
            source = str(input_row["source"])
            candidate = str(input_row["candidate_target"])
            if decision == "accept" and target != candidate:
                errors.append(f"{key}: accept target differs from candidate")
                continue
            if decision == "revert" and target != source:
                errors.append(f"{key}: revert target differs from source")
                continue
            if decision == "revise":
                if target == source:
                    errors.append(f"{key}: revised target equals source")
                    continue
                if Counter(extract_numbers(source)) != Counter(extract_numbers(target)):
                    errors.append(f"{key}: revised target changes numbers")
                    continue
            accepted[key] = {"decision": decision, "target": target}
    if errors:
        raise ValueError("warning review validation failed:\n" + "\n".join(errors))
    if len(accepted) != int(manifest["rows"]):
        raise ValueError(f"review rows={len(accepted)}, expected={manifest['rows']}")
    return accepted


def apply_reviews(work_dir: Path, review_dir: Path) -> None:
    reviews = validate_reviews(review_dir)
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    changed = Counter()
    found: set[str] = set()
    for shard in manifest["shards"]:
        output_path = work_dir / shard["output"]
        rows = read_jsonl(output_path)
        touched = False
        for row in rows:
            key = str(row["key"])
            review = reviews.get(key)
            if review is None:
                continue
            found.add(key)
            changed[review["decision"]] += 1
            if review["decision"] == "revert":
                row["decision"] = "keep"
            else:
                row["decision"] = "simplified"
            row["target"] = review["target"]
            touched = True
        if touched:
            write_jsonl(output_path, rows)
    missing = set(reviews) - found
    if missing:
        raise KeyError(f"review keys not found in output shards: {sorted(missing)}")
    print(json.dumps({"applied": len(found), "decisions": changed}, ensure_ascii=False, default=dict))


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_review(args.review_file.resolve(), args.review_dir.resolve(), args.shards)
    else:
        apply_reviews(args.work_dir.resolve(), args.review_dir.resolve())


if __name__ == "__main__":
    main()
