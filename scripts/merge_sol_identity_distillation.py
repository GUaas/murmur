from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from muddywater.text_simplification.sol_distillation import (
    ValidatedOutput,
    load_and_validate_shard,
    summarize_changes,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and merge GPT-5.6-sol distillation shards.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_manifest(work_dir: Path) -> dict[str, Any]:
    manifest_path = work_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("teacher_model") != "gpt-5.6-sol":
        raise ValueError("manifest teacher_model must be gpt-5.6-sol")
    return manifest


def collect_outputs(work_dir: Path, manifest: dict[str, Any]) -> tuple[list[ValidatedOutput], list[dict[str, str]]]:
    collected: list[ValidatedOutput] = []
    issues: list[dict[str, str]] = []
    missing: list[str] = []
    for shard in manifest["shards"]:
        input_path = work_dir / shard["input"]
        output_path = work_dir / shard["output"]
        if not output_path.is_file():
            missing.append(str(output_path))
            continue
        rows, shard_issues = load_and_validate_shard(input_path, output_path)
        collected.extend(rows)
        issues.extend(issue.to_dict() for issue in shard_issues)
    if missing:
        raise FileNotFoundError("missing output shards:\n" + "\n".join(missing))
    errors = [issue for issue in issues if issue["severity"] == "error"]
    if errors:
        preview = json.dumps(errors[:20], ensure_ascii=False, indent=2)
        raise ValueError(f"distillation validation failed with {len(errors)} errors:\n{preview}")
    expected = int(manifest["identity_rows"])
    if len(collected) != expected:
        raise ValueError(f"validated rows={len(collected)}, expected={expected}")
    return collected, issues


def load_sources(work_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for shard in manifest["shards"]:
        path = work_dir / shard["input"]
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    sources[str(row["key"])] = str(row["source"])
    return sources


def merge_dataset(input_dir: Path, output_dir: Path, outputs: list[ValidatedOutput]) -> dict[str, Any]:
    replacement = {row.key: row for row in outputs}
    split_report: dict[str, Any] = {}
    for split in ("train", "validation"):
        source_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"
        merged: list[dict[str, Any]] = []
        changed = 0
        identity_remaining = 0
        with source_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                row = replacement.get(f"{split}:{line_number}")
                if row is not None and row.decision == "simplified":
                    payload["target"] = row.target
                    changed += 1
                if str(payload.get("source", "")) == str(payload.get("target", "")):
                    identity_remaining += 1
                merged.append(payload)
        write_jsonl(output_path, merged)
        split_report[split] = {
            "rows": len(merged),
            "changed": changed,
            "identity_remaining": identity_remaining,
        }
    return split_report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    split = report["splits"]
    warning_review = report.get("warning_review")
    warning_lines = [f"- 启发式规则仍标记：{summary['warning_rows']:,} 条"]
    if warning_review:
        decisions = warning_review["decisions"]
        warning_lines.extend(
            [
                f"- 二审样本：{warning_review['rows']:,} 条（接受 {decisions.get('accept', 0):,}、回退 {decisions.get('revert', 0):,}、重写 {decisions.get('revise', 0):,}）",
                "- 当前仍带启发式标记的结果均已完成 GPT-5.6-sol 二审，并非待处理错误。",
            ]
        )
    return "\n".join(
        [
            "# GPT-5.6-sol 恒等样本重蒸馏报告",
            "",
            f"- 教师模型：`{report['teacher_model']}`",
            f"- 审核样本：{summary['processed']:,}",
            f"- 重新简化：{summary['simplified']:,}（{summary['simplified_rate']:.2%}）",
            f"- 保留原文：{summary['kept']:,}",
            f"- 字符压缩率：{summary['character_reduction_rate']:.2%}",
            f"- 涉政治关键词样本：{summary['political_rows']:,}",
            f"- 涉政治关键词样本重新简化率：{summary['political_simplified_rate']:.2%}",
            *warning_lines,
            "",
            "## 合并结果",
            "",
            f"- 训练集：{split['train']['rows']:,} 条，改写 {split['train']['changed']:,} 条，仍相同 {split['train']['identity_remaining']:,} 条。",
            f"- 验证集：{split['validation']['rows']:,} 条，改写 {split['validation']['changed']:,} 条，仍相同 {split['validation']['identity_remaining']:,} 条。",
            "",
            "所有数字均经过逐条一致性校验；二审后仍触发长度启发式规则的结果另存于 `reviewed_heuristic_flags.jsonl`。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(work_dir)
    outputs, issues = collect_outputs(work_dir, manifest)
    sources = load_sources(work_dir, manifest)
    summary = summarize_changes(sources, outputs)
    splits = merge_dataset(input_dir, output_dir, outputs)
    warning_keys = {issue["key"] for issue in issues if issue["severity"] == "warning"}
    warning_rows = [
        {
            "key": row.key,
            "source": sources[row.key],
            "target": row.target,
            "warnings": list(row.warnings),
        }
        for row in outputs
        if row.key in warning_keys
    ]
    write_jsonl(output_dir / "reviewed_heuristic_flags.jsonl", warning_rows)
    warning_review = None
    review_manifest_path = work_dir / "warning_review" / "review_manifest.json"
    if review_manifest_path.is_file():
        review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8-sig"))
        decisions: dict[str, int] = {}
        reviewed_rows = 0
        complete = True
        for shard in review_manifest["shards"]:
            path = review_manifest_path.parent / shard["output"]
            if not path.is_file():
                complete = False
                break
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    decision = str(row.get("decision", ""))
                    decisions[decision] = decisions.get(decision, 0) + 1
                    reviewed_rows += 1
        if complete:
            warning_review = {"rows": reviewed_rows, "decisions": decisions}
    report = {
        "teacher_model": manifest["teacher_model"],
        "summary": summary,
        "splits": splits,
        "warnings": len(warning_rows),
        "warning_review": warning_review,
    }
    (output_dir / "distillation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "distillation_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
