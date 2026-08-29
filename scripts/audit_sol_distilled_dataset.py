from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from muddywater.text_simplification.identity_repair import (
    compact_text,
    extract_numbers,
    semantic_surface,
    topic_flags,
)
from muddywater.text_simplification.sol_distillation import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extreme integrity audit for distilled JSONL data.")
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--distilled-dir", type=Path, required=True)
    parser.add_argument("--removed-rows", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def audit_split(
    original_path: Path,
    distilled_path: Path,
    split: str,
    removed_lines: set[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    original = read_jsonl(original_path)
    if removed_lines:
        original = [
            row
            for line_number, row in enumerate(original, start=1)
            if line_number not in removed_lines
        ]
    distilled = read_jsonl(distilled_path)
    errors: Counter[str] = Counter()
    changed_samples: list[dict[str, str]] = []
    reductions: list[float] = []
    changed_lengths: Counter[str] = Counter()
    identity_before = identity_after = changed = 0
    political_identity = political_changed = 0
    nonpolitical_identity = nonpolitical_changed = 0
    protected_nonidentity = protected_nonidentity_unchanged = 0

    if len(original) != len(distilled):
        errors["row_count_mismatch"] += abs(len(original) - len(distilled)) or 1
    for index, (before, after) in enumerate(zip(original, distilled), start=1):
        key = f"{split}:{index}"
        before_source = str(before.get("source", ""))
        before_target = str(before.get("target", ""))
        after_source = str(after.get("source", ""))
        after_target = str(after.get("target", ""))
        was_identity = before_source == before_target
        is_identity = after_source == after_target
        identity_before += was_identity
        identity_after += is_identity
        if after_source != before_source:
            errors["source_changed"] += 1
        if {k: v for k, v in after.items() if k != "target"} != {
            k: v for k, v in before.items() if k != "target"
        }:
            errors["non_target_field_changed"] += 1
        if not was_identity:
            protected_nonidentity += 1
            if after == before:
                protected_nonidentity_unchanged += 1
            else:
                errors["preexisting_pair_changed"] += 1
            continue
        political = bool(topic_flags(before_source))
        if political:
            political_identity += 1
        else:
            nonpolitical_identity += 1
        if after_target == before_target:
            continue
        changed += 1
        if political:
            political_changed += 1
        else:
            nonpolitical_changed += 1
        if not after_target.strip():
            errors["empty_target"] += 1
        if Counter(extract_numbers(before_source)) != Counter(extract_numbers(after_target)):
            errors["number_mismatch"] += 1
        if semantic_surface(before_source) == semantic_surface(after_target):
            errors["punctuation_or_spacing_only"] += 1
        source_chars = len(compact_text(before_source))
        target_chars = len(compact_text(after_target))
        if target_chars < source_chars:
            changed_lengths["shorter"] += 1
        elif target_chars > source_chars:
            changed_lengths["longer"] += 1
        else:
            changed_lengths["same_length"] += 1
        if source_chars:
            reductions.append((source_chars - target_chars) / source_chars)
        if len(changed_samples) < 100:
            changed_samples.append({"key": key, "source": before_source, "target": after_target})

    report = {
        "rows": len(distilled),
        "identity_before": identity_before,
        "identity_after": identity_after,
        "identity_rate_before": identity_before / len(original) if original else 0.0,
        "identity_rate_after": identity_after / len(distilled) if distilled else 0.0,
        "changed_identity_rows": changed,
        "protected_nonidentity_rows": protected_nonidentity,
        "protected_nonidentity_rows_unchanged": protected_nonidentity_unchanged,
        "changed_length_direction": dict(changed_lengths),
        "changed_reduction_mean": statistics.fmean(reductions) if reductions else 0.0,
        "changed_reduction_median": statistics.median(reductions) if reductions else 0.0,
        "changed_reduction_p10": percentile(reductions, 0.10),
        "changed_reduction_p90": percentile(reductions, 0.90),
        "political_identity_rows": political_identity,
        "political_changed": political_changed,
        "political_changed_rate": political_changed / political_identity if political_identity else 0.0,
        "nonpolitical_identity_rows": nonpolitical_identity,
        "nonpolitical_changed": nonpolitical_changed,
        "nonpolitical_changed_rate": (
            nonpolitical_changed / nonpolitical_identity if nonpolitical_identity else 0.0
        ),
        "errors": dict(errors),
        "sha256": sha256(distilled_path),
        "bytes": distilled_path.stat().st_size,
    }
    return report, changed_samples


def render_markdown(report: dict[str, Any]) -> str:
    total = report["total"]
    train = report["splits"]["train"]
    validation = report["splits"]["validation"]
    status = "通过" if report["passed"] else "失败"
    return "\n".join(
        [
            "# GPT-5.6-sol 蒸馏数据极致审计",
            "",
            f"- 结论：**{status}**",
            f"- 总行数：{total['rows']:,}",
            f"- 重新简化：{total['changed_identity_rows']:,}",
            f"- source=target：{total['identity_before']:,} → {total['identity_after']:,}（{total['identity_rate_before']:.2%} → {total['identity_rate_after']:.2%}）",
            f"- 原有非恒等训练对保护：{total['protected_nonidentity_rows_unchanged']:,}/{total['protected_nonidentity_rows']:,}",
            f"- 数字变化错误：{total['errors'].get('number_mismatch', 0)}",
            f"- 仅标点/空格变化错误：{total['errors'].get('punctuation_or_spacing_only', 0)}",
            f"- 训练集：{train['rows']:,} 行，改写 {train['changed_identity_rows']:,} 行",
            f"- 验证集：{validation['rows']:,} 行，改写 {validation['changed_identity_rows']:,} 行",
            f"- 改写样本字符压缩率：均值 {total['changed_reduction_mean']:.2%}，中位数 {total['changed_reduction_median']:.2%}",
            f"- 政治关键词样本改写率：{total['political_changed_rate']:.2%}",
            f"- 非政治关键词样本改写率：{total['nonpolitical_changed_rate']:.2%}",
            f"- 审计性能：{report['performance']['rows_per_second']:,.0f} 行/秒，耗时 {report['performance']['elapsed_seconds']:.3f} 秒",
            "",
            "## 文件校验值",
            "",
            f"- `train.jsonl`：`{train['sha256']}`",
            f"- `validation.jsonl`：`{validation['sha256']}`",
            "",
            "## 判定说明",
            "",
            "完整性审计覆盖行数、字段、源文本、非目标字段、原有非恒等样本保护、空输出、数字一致性和纯标点改写。语义质量由 GPT-5.6-sol 首轮逐条判断，并对启发式警告样本进行了第二轮独立复核。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    removed_lines: dict[str, set[int]] = {"train": set(), "validation": set()}
    if args.removed_rows:
        for row in read_jsonl(args.removed_rows.resolve()):
            removed_lines[str(row["split"])].add(int(row["line_number"]))
    split_reports: dict[str, Any] = {}
    samples: list[dict[str, str]] = []
    for split in ("train", "validation"):
        report, split_samples = audit_split(
            args.original_dir.resolve() / f"{split}.jsonl",
            args.distilled_dir.resolve() / f"{split}.jsonl",
            split,
            removed_lines[split],
        )
        split_reports[split] = report
        samples.extend(split_samples)
    total_rows = sum(item["rows"] for item in split_reports.values())
    total_errors = Counter()
    for item in split_reports.values():
        total_errors.update(item["errors"])
    total_identity_before = sum(item["identity_before"] for item in split_reports.values())
    total_identity_after = sum(item["identity_after"] for item in split_reports.values())
    total_changed = sum(item["changed_identity_rows"] for item in split_reports.values())
    protected = sum(item["protected_nonidentity_rows"] for item in split_reports.values())
    protected_unchanged = sum(
        item["protected_nonidentity_rows_unchanged"] for item in split_reports.values()
    )
    political_rows = sum(item["political_identity_rows"] for item in split_reports.values())
    political_changed = sum(item["political_changed"] for item in split_reports.values())
    nonpolitical_rows = sum(item["nonpolitical_identity_rows"] for item in split_reports.values())
    nonpolitical_changed = sum(item["nonpolitical_changed"] for item in split_reports.values())
    weighted_reduction = sum(
        item["changed_reduction_mean"] * item["changed_identity_rows"]
        for item in split_reports.values()
    ) / total_changed if total_changed else 0.0
    reduction_median = statistics.median(
        [item["changed_reduction_median"] for item in split_reports.values()]
    )
    elapsed = time.perf_counter() - started
    total = {
        "rows": total_rows,
        "identity_before": total_identity_before,
        "identity_after": total_identity_after,
        "identity_rate_before": total_identity_before / total_rows if total_rows else 0.0,
        "identity_rate_after": total_identity_after / total_rows if total_rows else 0.0,
        "changed_identity_rows": total_changed,
        "protected_nonidentity_rows": protected,
        "protected_nonidentity_rows_unchanged": protected_unchanged,
        "changed_reduction_mean": weighted_reduction,
        "changed_reduction_median": reduction_median,
        "political_changed_rate": political_changed / political_rows if political_rows else 0.0,
        "nonpolitical_changed_rate": (
            nonpolitical_changed / nonpolitical_rows if nonpolitical_rows else 0.0
        ),
        "errors": dict(total_errors),
    }
    final_report = {
        "passed": not total_errors and protected == protected_unchanged,
        "teacher_model": "gpt-5.6-sol",
        "removed_rows": sum(len(lines) for lines in removed_lines.values()),
        "total": total,
        "splits": split_reports,
        "performance": {
            "elapsed_seconds": elapsed,
            "rows_per_second": total_rows / elapsed if elapsed else 0.0,
        },
    }
    output_dir = args.distilled_dir.resolve()
    (output_dir / "extreme_data_audit.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "extreme_data_audit.md").write_text(render_markdown(final_report), encoding="utf-8")
    write_jsonl(output_dir / "changed_samples.jsonl", samples)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    if not final_report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
