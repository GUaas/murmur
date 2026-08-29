from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

from muddywater.text_simplification.identity_repair import compact_text
from muddywater.text_simplification.sol_distillation import read_jsonl, write_jsonl


SOURCE_BINS = (
    ("1–20", 1, 20),
    ("21–40", 21, 40),
    ("41–60", 41, 60),
    ("61–80", 61, 80),
    ("81–120", 81, 120),
    ("121–200", 121, 200),
    (">200", 201, math.inf),
)


@dataclass(frozen=True)
class LengthRow:
    split: str
    line_number: int
    source: str
    target: str
    source_chars: int
    target_chars: int
    teacher_changed: bool

    @property
    def key(self) -> str:
        return f"{self.split}:{self.line_number}"

    @property
    def difference(self) -> int:
        return self.target_chars - self.source_chars

    @property
    def ratio(self) -> float:
        return self.target_chars / self.source_chars if self.source_chars else 0.0

    @property
    def relation(self) -> str:
        if self.target_chars < self.source_chars:
            return "shorter"
        if self.target_chars > self.source_chars:
            return "longer"
        return "equal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze source/target length distributions.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--removed-rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_plotting() -> None:
    for candidate in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def load_removed_lines(path: Path | None) -> dict[str, set[int]]:
    removed = {"train": set(), "validation": set()}
    if path is None:
        return removed
    for row in read_jsonl(path):
        split = str(row["split"])
        removed[split].add(int(row["line_number"]))
    return removed


def load_rows(
    dataset_dir: Path,
    baseline_dir: Path | None,
    removed_lines: dict[str, set[int]],
) -> list[LengthRow]:
    rows: list[LengthRow] = []
    for split in ("train", "validation"):
        current = read_jsonl(dataset_dir / f"{split}.jsonl")
        baseline = read_jsonl(baseline_dir / f"{split}.jsonl") if baseline_dir else current
        if baseline_dir and removed_lines[split]:
            baseline = [
                row
                for line_number, row in enumerate(baseline, start=1)
                if line_number not in removed_lines[split]
            ]
        if len(current) != len(baseline):
            raise ValueError(f"{split}: current rows={len(current)}, baseline rows={len(baseline)}")
        for line_number, (current_row, baseline_row) in enumerate(
            zip(current, baseline, strict=True), start=1
        ):
            source = str(current_row.get("source", ""))
            target = str(current_row.get("target", ""))
            if source != str(baseline_row.get("source", "")):
                raise ValueError(f"{split}:{line_number}: source differs from baseline")
            rows.append(
                LengthRow(
                    split=split,
                    line_number=line_number,
                    source=source,
                    target=target,
                    source_chars=len(compact_text(source)),
                    target_chars=len(compact_text(target)),
                    teacher_changed=target != str(baseline_row.get("target", "")),
                )
            )
    return rows


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), fraction))


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def relation_counts(rows: Iterable[LengthRow]) -> dict[str, int]:
    counter = Counter(row.relation for row in rows)
    return {name: counter[name] for name in ("shorter", "equal", "longer")}


def summarize(rows: list[LengthRow], removed_rows: int = 0) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in ("train", "validation", "all"):
        selected = rows if split == "all" else [row for row in rows if row.split == split]
        relations = relation_counts(selected)
        splits[split] = {
            "rows": len(selected),
            "source_characters": describe([row.source_chars for row in selected]),
            "target_characters": describe([row.target_chars for row in selected]),
            "relations": relations,
            "relation_rates": {
                name: count / len(selected) if selected else 0.0
                for name, count in relations.items()
            },
        }

    longer = [row for row in rows if row.relation == "longer"]
    extra_bins = {
        "1": sum(row.difference == 1 for row in longer),
        "2–3": sum(2 <= row.difference <= 3 for row in longer),
        "4–5": sum(4 <= row.difference <= 5 for row in longer),
        "6–10": sum(6 <= row.difference <= 10 for row in longer),
        "11–20": sum(11 <= row.difference <= 20 for row in longer),
        ">20": sum(row.difference > 20 for row in longer),
    }
    ratio_bins = {
        "1.00–1.05": sum(row.ratio <= 1.05 for row in longer),
        "1.05–1.10": sum(1.05 < row.ratio <= 1.10 for row in longer),
        "1.10–1.20": sum(1.10 < row.ratio <= 1.20 for row in longer),
        "1.20–1.50": sum(1.20 < row.ratio <= 1.50 for row in longer),
        ">1.50": sum(row.ratio > 1.50 for row in longer),
    }
    by_source_length: list[dict[str, Any]] = []
    for label, lower, upper in SOURCE_BINS:
        selected = [row for row in rows if lower <= row.source_chars <= upper]
        counts = relation_counts(selected)
        by_source_length.append(
            {
                "label": label,
                "rows": len(selected),
                "counts": counts,
                "rates": {
                    name: counts[name] / len(selected) if selected else 0.0
                    for name in counts
                },
            }
        )
    teacher_changed = [row for row in rows if row.teacher_changed]
    teacher_relations = relation_counts(teacher_changed)
    return {
        "removed_rows": removed_rows,
        "splits": splits,
        "total_source_characters": sum(row.source_chars for row in rows),
        "total_target_characters": sum(row.target_chars for row in rows),
        "total_character_reduction_rate": (
            1 - sum(row.target_chars for row in rows) / sum(row.source_chars for row in rows)
        ),
        "longer_than_source": {
            "rows": len(longer),
            "rate": len(longer) / len(rows),
            "extra_characters": describe([row.difference for row in longer]),
            "extra_bins": extra_bins,
            "ratio_bins": ratio_bins,
            "ratio_over_1_2": sum(row.ratio > 1.2 for row in longer),
            "ratio_over_1_5": sum(row.ratio > 1.5 for row in longer),
        },
        "by_source_length": by_source_length,
        "teacher_changed": {
            "rows": len(teacher_changed),
            "relations": teacher_relations,
            "relation_rates": {
                name: count / len(teacher_changed) if teacher_changed else 0.0
                for name, count in teacher_relations.items()
            },
        },
    }


def annotate_bars(axis: Any, bars: Iterable[Any], values: Iterable[float], *, percent: bool = False) -> None:
    for bar, value in zip(bars, values, strict=True):
        text = f"{value:.1%}" if percent else f"{int(value):,}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def render_charts(rows: list[LengthRow], report: dict[str, Any], assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    relation = report["splits"]["all"]["relations"]
    labels = ["简化更短", "长度相同", "简化更长"]
    values = [relation["shorter"], relation["equal"], relation["longer"]]
    bars = axis.bar(labels, values, color=["#4472C4", "#A5A5A5", "#ED7D31"])
    annotate_bars(axis, bars, values)
    axis.set_title("原文与简化文本的长度关系")
    axis.set_ylabel("样本数")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(assets / "length_relation.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    maximum = 300
    bins = np.arange(0, maximum + 10, 10)
    source_lengths = [min(row.source_chars, maximum) for row in rows]
    target_lengths = [min(row.target_chars, maximum) for row in rows]
    axis.hist(source_lengths, bins=bins, histtype="step", linewidth=2, label="原文")
    axis.hist(target_lengths, bins=bins, histtype="step", linewidth=2, label="简化文本")
    axis.set_title("字符长度分布（超过300字符归入末端）")
    axis.set_xlabel("去空白字符数")
    axis.set_ylabel("样本数")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(assets / "source_target_length_distribution.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    bucket_labels = [item["label"] for item in report["by_source_length"]]
    shorter = np.asarray([item["rates"]["shorter"] for item in report["by_source_length"]])
    equal = np.asarray([item["rates"]["equal"] for item in report["by_source_length"]])
    longer = np.asarray([item["rates"]["longer"] for item in report["by_source_length"]])
    axis.bar(bucket_labels, shorter, label="简化更短", color="#4472C4")
    axis.bar(bucket_labels, equal, bottom=shorter, label="长度相同", color="#A5A5A5")
    axis.bar(bucket_labels, longer, bottom=shorter + equal, label="简化更长", color="#ED7D31")
    axis.set_title("不同原文长度区间的结果分布")
    axis.set_xlabel("原文字符数")
    axis.set_ylabel("区间内占比")
    axis.set_ylim(0, 1)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(assets / "relation_by_source_length.png", bbox_inches="tight")
    plt.close(figure)

    longer_rows = [row for row in rows if row.relation == "longer"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    extra = report["longer_than_source"]["extra_bins"]
    extra_values = list(extra.values())
    bars = axes[0].bar(list(extra), extra_values, color="#ED7D31")
    annotate_bars(axes[0], bars, extra_values)
    axes[0].set_title("多出的字符数")
    axes[0].set_xlabel("目标比原文多出的字符")
    axes[0].set_ylabel("样本数")
    axes[0].grid(axis="y", alpha=0.25)
    ratios = report["longer_than_source"]["ratio_bins"]
    ratio_values = list(ratios.values())
    bars = axes[1].bar(list(ratios), ratio_values, color="#70AD47")
    annotate_bars(axes[1], bars, ratio_values)
    axes[1].set_title("目标/原文长度比")
    axes[1].set_xlabel("长度比")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(f"简化文本超过原文的 {len(longer_rows):,} 条样本")
    figure.tight_layout()
    figure.savefig(assets / "longer_than_source_breakdown.png", bbox_inches="tight")
    plt.close(figure)

    upper = int(math.ceil(max(
        percentile([row.source_chars for row in rows], 0.99),
        percentile([row.target_chars for row in rows], 0.99),
    ) / 25) * 25)
    plotted = [row for row in rows if row.source_chars <= upper and row.target_chars <= upper]
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    image = axis.hexbin(
        [row.source_chars for row in plotted],
        [row.target_chars for row in plotted],
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="Blues",
    )
    axis.plot([0, upper], [0, upper], linestyle="--", color="#C00000", linewidth=1.2, label="长度相同")
    axis.set_xlim(0, upper)
    axis.set_ylim(0, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"原文与简化长度关系（覆盖99%+样本，≤{upper}字符）")
    axis.set_xlabel("原文字符数")
    axis.set_ylabel("简化文本字符数")
    axis.legend()
    figure.colorbar(image, ax=axis, label="样本密度（对数）")
    figure.tight_layout()
    figure.savefig(assets / "source_target_length_hexbin.png", bbox_inches="tight")
    plt.close(figure)


def render_markdown(report: dict[str, Any]) -> str:
    all_rows = report["splits"]["all"]
    longer = report["longer_than_source"]
    changed = report["teacher_changed"]
    return "\n".join(
        [
            "# 数据集分布与长度关系报告",
            "",
            f"- 数据总量：{all_rows['rows']:,} 条（训练集 {report['splits']['train']['rows']:,}，验证集 {report['splits']['validation']['rows']:,}）。",
            f"- 原文长度：均值 {all_rows['source_characters']['mean']:.2f}，中位数 {all_rows['source_characters']['median']:.0f}，P95 {all_rows['source_characters']['p95']:.0f}，最大 {all_rows['source_characters']['max']:,} 字符。",
            f"- 简化长度：均值 {all_rows['target_characters']['mean']:.2f}，中位数 {all_rows['target_characters']['median']:.0f}，P95 {all_rows['target_characters']['p95']:.0f}，最大 {all_rows['target_characters']['max']:,} 字符。",
            f"- 全数据字符总量下降：{report['total_character_reduction_rate']:.2%}。",
            f"- 简化更短：{all_rows['relations']['shorter']:,} 条（{all_rows['relation_rates']['shorter']:.2%}）。",
            f"- 长度相同：{all_rows['relations']['equal']:,} 条（{all_rows['relation_rates']['equal']:.2%}）。",
            f"- 简化更长：{all_rows['relations']['longer']:,} 条（{all_rows['relation_rates']['longer']:.2%}）。",
            "",
            "## 简化文本超过原文",
            "",
            f"确实存在，共 {longer['rows']:,} 条。多出字符数的中位数为 {longer['extra_characters']['median']:.0f}，均值 {longer['extra_characters']['mean']:.2f}，最大 {longer['extra_characters']['max']:.0f}。",
            f"其中 {longer['extra_bins']['1'] + longer['extra_bins']['2–3']:,} 条只多 1–3 个字符；{longer['ratio_over_1_2']:,} 条超过原文 20%；{longer['ratio_over_1_5']:,} 条超过原文 50%。",
            f"GPT-5.6-sol 本轮改写的 {changed['rows']:,} 条中，更短 {changed['relations']['shorter']:,} 条、等长 {changed['relations']['equal']:,} 条、更长 {changed['relations']['longer']:,} 条（{changed['relation_rates']['longer']:.2%}）。其余更长样本来自原数据。",
            "",
            "## 图表",
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
            f"`longer_than_source_top100.jsonl` 保存超出字符数最多的 100 个样本；`longer_than_source_over_20pct.jsonl` 保存长度超过原文 20% 的 {longer['ratio_over_1_2']:,} 条风险样本。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    configure_plotting()
    dataset_dir = args.dataset_dir.resolve()
    baseline_dir = args.baseline_dir.resolve() if args.baseline_dir else None
    removed_rows_path = args.removed_rows.resolve() if args.removed_rows else None
    removed_lines = load_removed_lines(removed_rows_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(dataset_dir, baseline_dir, removed_lines)
    report = summarize(rows, sum(len(lines) for lines in removed_lines.values()))
    render_charts(rows, report, output_dir / "dataset_distribution_assets")
    (output_dir / "dataset_distribution_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_distribution_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    top_longer = sorted(
        (row for row in rows if row.relation == "longer"),
        key=lambda row: (row.difference, row.ratio),
        reverse=True,
    )[:100]
    write_jsonl(
        output_dir / "longer_than_source_top100.jsonl",
        (
            {
                **asdict(row),
                "key": row.key,
                "difference": row.difference,
                "ratio": row.ratio,
            }
            for row in top_longer
        ),
    )
    risk_rows = sorted(
        (row for row in rows if row.relation == "longer" and row.ratio > 1.2),
        key=lambda row: (row.ratio, row.difference),
        reverse=True,
    )
    write_jsonl(
        output_dir / "longer_than_source_over_20pct.jsonl",
        (
            {
                **asdict(row),
                "key": row.key,
                "difference": row.difference,
                "ratio": row.ratio,
            }
            for row in risk_rows
        ),
    )
    write_jsonl(
        output_dir / "gpt56sol_changed_longer_than_source.jsonl",
        (
            {
                **asdict(row),
                "key": row.key,
                "difference": row.difference,
                "ratio": row.ratio,
            }
            for row in sorted(
                (row for row in rows if row.teacher_changed and row.relation == "longer"),
                key=lambda item: (item.difference, item.ratio),
                reverse=True,
            )
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
