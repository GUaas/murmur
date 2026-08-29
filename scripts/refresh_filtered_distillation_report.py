from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from muddywater.text_simplification.identity_repair import compact_text, topic_flags
from muddywater.text_simplification.sol_distillation import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh reports after destructive row filtering.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--removed-rows", type=Path, required=True)
    return parser.parse_args()


def removed_lines(path: Path) -> dict[str, set[int]]:
    result = {"train": set(), "validation": set()}
    for row in read_jsonl(path):
        result[str(row["split"])].add(int(row["line_number"]))
    return result


def filter_reviewed_flags(dataset_dir: Path, removed_keys: set[str]) -> int:
    path = dataset_dir / "reviewed_heuristic_flags.jsonl"
    if not path.is_file():
        return 0
    kept = [row for row in read_jsonl(path) if str(row.get("key", "")) not in removed_keys]
    write_jsonl(path, kept)
    return len(kept)


def collect_identity_stats(
    dataset_dir: Path,
    baseline_dir: Path,
    removed: dict[str, set[int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    totals = {
        "processed": 0,
        "simplified": 0,
        "kept": 0,
        "source_characters": 0,
        "target_characters": 0,
        "political_rows": 0,
        "political_simplified": 0,
    }
    splits: dict[str, Any] = {}
    for split in ("train", "validation"):
        baseline = [
            row
            for line_number, row in enumerate(read_jsonl(baseline_dir / f"{split}.jsonl"), start=1)
            if line_number not in removed[split]
        ]
        current = read_jsonl(dataset_dir / f"{split}.jsonl")
        if len(baseline) != len(current):
            raise ValueError(f"{split}: baseline={len(baseline)}, current={len(current)}")
        changed = identity_remaining = reviewed = 0
        for before, after in zip(baseline, current, strict=True):
            source = str(before.get("source", ""))
            if source != str(before.get("target", "")):
                continue
            reviewed += 1
            target = str(after.get("target", ""))
            is_changed = target != source
            changed += is_changed
            identity_remaining += not is_changed
            totals["source_characters"] += len(compact_text(source))
            totals["target_characters"] += len(compact_text(target))
            political = bool(topic_flags(source))
            totals["political_rows"] += political
            totals["political_simplified"] += political and is_changed
        splits[split] = {
            "rows": len(current),
            "reviewed_identity_rows": reviewed,
            "changed": changed,
            "identity_remaining": identity_remaining,
        }
        totals["processed"] += reviewed
        totals["simplified"] += changed
        totals["kept"] += identity_remaining
    totals["simplified_rate"] = totals["simplified"] / totals["processed"]
    totals["character_reduction_rate"] = (
        (totals["source_characters"] - totals["target_characters"])
        / totals["source_characters"]
    )
    totals["political_simplified_rate"] = (
        totals["political_simplified"] / totals["political_rows"]
    )
    return totals, splits


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    splits = report["splits"]
    review = report.get("warning_review") or {"rows": 0, "decisions": {}}
    decisions = review["decisions"]
    removed = report["length_ratio_filter"]["removed_rows"]
    return "\n".join(
        [
            "# GPT-5.6-sol 蒸馏与长度过滤报告",
            "",
            f"- 当前数据：{splits['train']['rows'] + splits['validation']['rows']:,} 条。",
            f"- 审核恒等样本：{summary['processed']:,} 条。",
            f"- 重新简化：{summary['simplified']:,} 条（{summary['simplified_rate']:.2%}）。",
            f"- 合理保留：{summary['kept']:,} 条。",
            f"- 恒等样本字符压缩率：{summary['character_reduction_rate']:.2%}。",
            f"- 政治关键词样本重新简化率：{summary['political_simplified_rate']:.2%}。",
            f"- 已删除目标/原文长度比超过 1.20 的样本：{removed:,} 条。",
            f"- 二审：{review['rows']:,} 条（接受 {decisions.get('accept', 0):,}、回退 {decisions.get('revert', 0):,}、重写 {decisions.get('revise', 0):,}）。",
            "",
            "## 数据划分",
            "",
            f"- 训练集：{splits['train']['rows']:,} 条，重新简化 {splits['train']['changed']:,} 条。",
            f"- 验证集：{splits['validation']['rows']:,} 条，重新简化 {splits['validation']['changed']:,} 条。",
            "",
            "## 当前分布图",
            "",
            "![长度关系](dataset_distribution_assets/length_relation.png)",
            "",
            "![长度分布](dataset_distribution_assets/source_target_length_distribution.png)",
            "",
            "![分区关系](dataset_distribution_assets/relation_by_source_length.png)",
            "",
            "![超长拆分](dataset_distribution_assets/longer_than_source_breakdown.png)",
            "",
            "![长度密度](dataset_distribution_assets/source_target_length_hexbin.png)",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    removed_path = args.removed_rows.resolve()
    removed = removed_lines(removed_path)
    removed_keys = {f"{split}:{line_number}" for split, lines in removed.items() for line_number in lines}
    warning_rows = filter_reviewed_flags(dataset_dir, removed_keys)
    previous_report_path = dataset_dir / "distillation_report.json"
    previous = (
        json.loads(previous_report_path.read_text(encoding="utf-8-sig"))
        if previous_report_path.is_file()
        else {}
    )
    summary, splits = collect_identity_stats(
        dataset_dir,
        args.baseline_dir.resolve(),
        removed,
    )
    summary["warning_rows"] = warning_rows
    report = {
        "teacher_model": "gpt-5.6-sol",
        "summary": summary,
        "splits": splits,
        "warnings": warning_rows,
        "warning_review": previous.get("warning_review"),
        "length_ratio_filter": {
            "rule": "target_characters / source_characters > 1.2",
            "removed_rows": len(removed_keys),
            "removed_by_split": {split: len(lines) for split, lines in removed.items()},
        },
    }
    previous_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "distillation_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
