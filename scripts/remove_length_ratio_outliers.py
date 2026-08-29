from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from muddywater.text_simplification.sol_distillation import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove target/source length-ratio outliers.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--previous-removed", type=Path)
    parser.add_argument("--removed-output", type=Path)
    return parser.parse_args()


def select_candidates(path: Path, threshold: float) -> list[dict[str, Any]]:
    selected = [row for row in read_jsonl(path) if float(row.get("ratio", 0.0)) > threshold]
    if not selected:
        raise ValueError(f"no candidates exceed ratio threshold {threshold}")
    return selected


def validate_candidates(
    split_rows: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
) -> dict[str, set[int]]:
    removals: dict[str, set[int]] = {"train": set(), "validation": set()}
    for candidate in candidates:
        split, raw_line_number = str(candidate["key"]).split(":", 1)
        line_number = int(raw_line_number)
        if split not in removals:
            raise ValueError(f"unsupported split: {split}")
        if line_number < 1 or line_number > len(split_rows[split]):
            raise IndexError(f"{candidate['key']}: line number is outside the dataset")
        current = split_rows[split][line_number - 1]
        if str(current.get("source", "")) != str(candidate.get("source", "")):
            raise ValueError(f"{candidate['key']}: source no longer matches; refusing destructive edit")
        if str(current.get("target", "")) != str(candidate.get("target", "")):
            raise ValueError(f"{candidate['key']}: target no longer matches; refusing destructive edit")
        removals[split].add(line_number)
    return removals


def refresh_distillation_report(
    dataset_dir: Path,
    removed_by_split: dict[str, int],
    cumulative_removed_by_split: dict[str, int],
    threshold: float,
) -> None:
    report_path = dataset_dir / "distillation_report.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    for split, removed in removed_by_split.items():
        report["splits"][split]["rows"] -= removed
    report["length_ratio_filter"] = {
        "removed_rows": sum(cumulative_removed_by_split.values()),
        "removed_by_split": cumulative_removed_by_split,
        "rule": "target_characters / source_characters > 1.2",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = dataset_dir / "distillation_report.md"
    if markdown_path.is_file():
        markdown = markdown_path.read_text(encoding="utf-8")
        for split, removed in removed_by_split.items():
            if not removed:
                continue
            old_rows = report["splits"][split]["rows"] + removed
            new_rows = report["splits"][split]["rows"]
            markdown = markdown.replace(f"{old_rows:,} 条", f"{new_rows:,} 条", 1)
        note = (
            "\n## 长度异常过滤\n\n"
            f"- 本次删除目标/原文字符长度比超过 {threshold:.2f} 的 {sum(removed_by_split.values()):,} 条样本。\n"
            f"- 累计删除长度比超过 1.20 的 {sum(cumulative_removed_by_split.values()):,} 条样本。\n"
            "- 这些样本不属于 GPT-5.6-sol 本轮新增改写。\n"
        )
        markdown_path.write_text(markdown.rstrip() + "\n" + note, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    candidates = select_candidates(args.candidates.resolve(), args.threshold)
    previous_removed = (
        read_jsonl(args.previous_removed.resolve()) if args.previous_removed else []
    )
    split_rows = {
        split: read_jsonl(dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    removals = validate_candidates(split_rows, candidates)
    removed_rows = []
    previous_lines: dict[str, set[int]] = {"train": set(), "validation": set()}
    for row in previous_removed:
        previous_lines[str(row["split"])].add(int(row["line_number"]))
    original_line_maps = {
        split: [
            line_number
            for line_number in range(1, len(rows) + len(previous_lines[split]) + 1)
            if line_number not in previous_lines[split]
        ]
        for split, rows in split_rows.items()
    }
    before = {split: len(rows) for split, rows in split_rows.items()}
    for split, rows in split_rows.items():
        kept = []
        for line_number, row in enumerate(rows, start=1):
            if line_number in removals[split]:
                original_line_number = original_line_maps[split][line_number - 1]
                removed_rows.append(
                    {
                        "key": f"{split}:{original_line_number}",
                        "current_key": f"{split}:{line_number}",
                        "split": split,
                        "line_number": original_line_number,
                        "source": row.get("source", ""),
                        "target": row.get("target", ""),
                        "ratio": next(
                            float(item["ratio"])
                            for item in candidates
                            if item["key"] == f"{split}:{line_number}"
                        ),
                    }
                )
            else:
                kept.append(row)
        write_jsonl(dataset_dir / f"{split}.jsonl", kept)
    cumulative_removed = [*previous_removed, *removed_rows]
    cumulative_removed.sort(key=lambda row: (row["split"], int(row["line_number"])))
    removed_output = (
        args.removed_output.resolve()
        if args.removed_output
        else dataset_dir / "removed_length_ratio_outliers.jsonl"
    )
    write_jsonl(removed_output, cumulative_removed)
    removed_by_split = {split: len(lines) for split, lines in removals.items()}
    cumulative_removed_by_split = {
        split: sum(str(row["split"]) == split for row in cumulative_removed)
        for split in split_rows
    }
    refresh_distillation_report(
        dataset_dir,
        removed_by_split,
        cumulative_removed_by_split,
        args.threshold,
    )
    after = {split: len(read_jsonl(dataset_dir / f"{split}.jsonl")) for split in split_rows}
    result = {
        "threshold": args.threshold,
        "removed": len(removed_rows),
        "cumulative_removed": len(cumulative_removed),
        "removed_by_split": removed_by_split,
        "before": before,
        "after": after,
        "removed_keys": [row["key"] for row in removed_rows],
        "removed_output": str(removed_output),
    }
    (dataset_dir / "length_ratio_filter_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
