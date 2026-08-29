from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "long_text_chunking_evaluation_20260814"
DEFAULT_RESULT = DEFAULT_OUTPUT / "raw" / "long_text_chunking_eval_results.json"

NAVY = "#19324D"
BLUE = "#3B82C4"
TEAL = "#1E8A8A"
GREEN = "#4E9A68"
ORANGE = "#E58A2B"
RED = "#C34A45"
GRAY = "#7B8794"
LIGHT = "#E8EEF3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build charts and CSV tables for long-text A/B evaluation.")
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def configure_style() -> None:
    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B8C2CC",
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": LIGHT,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _annotate(ax, bars, *, digits: int = 3, offset: float = 0.015) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def quality_ab(data: dict[str, Any], charts_dir: Path) -> None:
    direct = data["direct"]["summary"]
    chunked = data["chunked"]["summary"]
    labels = ["ROUGE-L", "chrF", "SARI"]
    keys = ["rouge_l_f1", "chrf", "sari"]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.3, 5.2))
    bars1 = ax.bar(x - width / 2, [direct[key] for key in keys], width, color=GRAY, label="旧：整篇直接推理")
    bars2 = ax.bar(x + width / 2, [chunked[key] for key in keys], width, color=TEAL, label="新：分句分块推理")
    _annotate(ax, bars1)
    _annotate(ax, bars2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("得分（越高越好）")
    ax.set_title("16 篇独立长文档：新算法显著提升参考质量")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, charts_dir / "01_quality_ab.png")


def reliability_ab(data: dict[str, Any], charts_dir: Path) -> None:
    direct = data["direct"]["summary"]
    chunked = data["chunked"]["summary"]
    labels = ["数字召回", "约束项召回", "尾部事实", "正常结束", "换行结构"]
    keys = [
        "number_recall",
        "constraint_item_recall",
        "tail_document_pass_rate",
        "all_chunks_finished_rate",
        "newline_layout_exact_rate",
    ]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    bars1 = ax.bar(x - width / 2, [direct[key] for key in keys], width, color=GRAY, label="旧：整篇直接推理")
    bars2 = ax.bar(x + width / 2, [chunked[key] for key in keys], width, color=GREEN, label="新：分句分块推理")
    _annotate(ax, bars1)
    _annotate(ax, bars2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("通过率 / 召回率")
    ax.set_title("可靠性提升：尾部事实、结束状态与版式全部恢复")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    _save(fig, charts_dir / "02_reliability_ab.png")


def latency_vs_length(data: dict[str, Any], charts_dir: Path) -> None:
    direct = data["direct"]["records"]
    chunked = data["chunked"]["records"]
    x = np.asarray([row["source_prompt_tokens"] for row in direct])
    order = np.argsort(x)
    direct_seconds = np.asarray([row["latency_ms"] for row in direct]) / 1000.0
    chunked_seconds = np.asarray([row["latency_ms"] for row in chunked]) / 1000.0
    fig, ax = plt.subplots(figsize=(10.4, 5.5))
    ax.plot(x[order], direct_seconds[order], color=GRAY, marker="o", linewidth=2, label="旧：直接推理")
    ax.plot(x[order], chunked_seconds[order], color=ORANGE, marker="o", linewidth=2, label="新：分块推理")
    ax.axvline(160, color=TEAL, linestyle="--", linewidth=1.4, label="默认分块预算 160")
    ax.set_xlabel("整篇提示词元数")
    ax.set_ylabel("端到端延迟（秒）")
    ax.set_title("延迟成本：完整输出越长，分块路径越慢")
    ax.legend(loc="upper left")
    _save(fig, charts_dir / "03_latency_vs_length.png")


def quality_by_length(data: dict[str, Any], charts_dir: Path) -> None:
    ordered = ["short_long", "medium_long", "long", "extreme_long"]
    translations = ["短长文", "中长文", "长文", "极长文"]
    direct = data["direct"]["by_length_tier"]
    chunked = data["chunked"]["by_length_tier"]
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axes[0].plot(x, [direct[key]["sari"] for key in ordered], color=GRAY, marker="o", linewidth=2.2, label="旧路径")
    axes[0].plot(x, [chunked[key]["sari"] for key in ordered], color=TEAL, marker="o", linewidth=2.2, label="新路径")
    axes[0].set_xticks(x, translations)
    axes[0].set_ylim(0.15, 0.72)
    axes[0].set_ylabel("SARI")
    axes[0].set_title("长度越大，质量收益越明显")
    axes[0].legend()
    axes[1].plot(x, [direct[key]["tail_document_pass_rate"] for key in ordered], color=GRAY, marker="D", linewidth=2.2, label="旧路径")
    axes[1].plot(x, [chunked[key]["tail_document_pass_rate"] for key in ordered], color=GREEN, marker="D", linewidth=2.2, label="新路径")
    axes[1].set_xticks(x, translations)
    axes[1].set_ylim(-0.03, 1.08)
    axes[1].set_ylabel("尾部事实保留率")
    axes[1].set_title("中长文以上：尾部事实从不稳定到 100%")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, charts_dir / "04_quality_by_length.png")


def budget_sweep(data: dict[str, Any], charts_dir: Path) -> None:
    summaries = data["budget_sweep"]["by_budget"]
    budgets = sorted(int(key) for key in summaries)
    sari = [summaries[str(key)]["sari"] for key in budgets]
    rouge = [summaries[str(key)]["rouge_l_f1"] for key in budgets]
    latency = [summaries[str(key)]["total_latency_seconds"] for key in budgets]
    chunks = [summaries[str(key)]["chunk_count_percentiles"]["mean"] for key in budgets]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axes[0].plot(budgets, sari, color=TEAL, marker="o", linewidth=2.2, label="SARI")
    axes[0].plot(budgets, rouge, color=BLUE, marker="o", linewidth=2.2, label="ROUGE-L")
    axes[0].set_xlabel("每块提示词元预算")
    axes[0].set_ylabel("质量得分")
    axes[0].set_title("96 / 160 / 224：160 的综合质量最好")
    axes[0].legend()
    bars = axes[1].bar([str(key) for key in budgets], latency, color=[BLUE, TEAL, ORANGE], alpha=0.9)
    axes[1].set_xlabel("每块提示词元预算")
    axes[1].set_ylabel("4 篇代表文档总延迟（秒）")
    axes[1].set_title("160 词元总延迟最低")
    ax2 = axes[1].twinx()
    ax2.plot([str(key) for key in budgets], chunks, color=RED, marker="D", linewidth=2, label="平均块数")
    ax2.set_ylabel("平均块数", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    for bar, value in zip(bars, latency):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.3, f"{value:.1f}s", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, charts_dir / "05_budget_sweep.png")


def segmentation_and_planning(data: dict[str, Any], charts_dir: Path) -> None:
    audit = data["segmentation_audit"]
    labels = ["人工边界", "人工无损", "保护项", "随机分割", "随机规划", "预算上限"]
    values = [
        audit["handcrafted_unit_count_pass_rate"],
        audit["handcrafted_lossless_rate"],
        audit["handcrafted_protected_pass_rate"],
        audit["fuzz_lossless_split_rate"],
        audit["fuzz_lossless_plan_rate"],
        audit["fuzz_budget_pass_rate"],
    ]
    bench = data["planning_benchmarks"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))
    bars = axes[0].barh(np.arange(len(labels)), values, color=GREEN)
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].set_xlim(0, 1.08)
    axes[0].set_xlabel("通过率")
    axes[0].set_title("12 项人工边界 + 2,000 份随机文档")
    axes[0].invert_yaxis()
    for bar, value in zip(bars, values):
        axes[0].text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.0%}", va="center")
    for shape, color, label in (("punctuated", BLUE, "有标点"), ("no_punctuation", ORANGE, "无标点")):
        rows = [row for row in bench if row["shape"] == shape]
        axes[1].plot(
            [row["characters"] for row in rows],
            [row["elapsed_ms"] for row in rows],
            marker="o",
            linewidth=2.2,
            color=color,
            label=label,
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("文档字符数（对数轴）")
    axes[1].set_ylabel("规划耗时 ms（对数轴）")
    axes[1].set_title("10 万字符规划低于 0.33 秒")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, charts_dir / "06_segmentation_planning.png")


def per_document_delta(data: dict[str, Any], charts_dir: Path) -> None:
    rows = data["comparison"]["details"]
    labels = [row["document_id"].replace("composite_", "C").replace("long_", "L") for row in rows]
    sari_delta = [row["sari_delta"] for row in rows]
    latency_ratio = [1.0 / row["latency_speedup"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.4), sharex=True)
    colors = [GREEN if value > 0 else RED if value < 0 else GRAY for value in sari_delta]
    axes[0].bar(x, sari_delta, color=colors)
    axes[0].axhline(0, color=NAVY, linewidth=0.8)
    axes[0].set_ylabel("新 - 旧 SARI")
    axes[0].set_title("13/16 文档 SARI 提升，2 篇持平，1 篇轻微下降")
    colors = [GREEN if value < 1 else ORANGE for value in latency_ratio]
    axes[1].bar(x, latency_ratio, color=colors)
    axes[1].axhline(1, color=NAVY, linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("新延迟 / 旧延迟")
    axes[1].set_title("仅 5/16 文档更快；极长完整输出可能慢 5-9 倍")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, charts_dir / "07_per_document_delta.png")


def chunk_count_cost(data: dict[str, Any], charts_dir: Path) -> None:
    direct = {row["document_id"]: row for row in data["direct"]["records"]}
    rows = data["chunked"]["records"]
    layout_colors = {
        "continuous": BLUE,
        "line_per_sentence": RED,
        "paragraphs": TEAL,
        "mixed": ORANGE,
    }
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    for layout, color in layout_colors.items():
        selected = [row for row in rows if row["layout"] == layout]
        ax.scatter(
            [row["chunk_count"] for row in selected],
            [row["latency_ms"] / direct[row["document_id"]]["latency_ms"] for row in selected],
            s=62,
            color=color,
            alpha=0.85,
            label=layout,
        )
    for row in rows:
        ratio = row["latency_ms"] / direct[row["document_id"]]["latency_ms"]
        if row["chunk_count"] >= 15 or ratio >= 4:
            ax.annotate(row["document_id"], (row["chunk_count"], ratio), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(1, color=NAVY, linestyle="--", linewidth=1.2)
    ax.set_xlabel("分块数量")
    ax.set_ylabel("新延迟 / 旧延迟")
    ax.set_title("逐行文本的块数膨胀是主要性能风险")
    ax.legend(title="版式")
    _save(fig, charts_dir / "08_chunk_count_cost.png")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_tables(data: dict[str, Any], tables_dir: Path) -> None:
    direct_by_id = {row["document_id"]: row for row in data["direct"]["records"]}
    chunked_by_id = {row["document_id"]: row for row in data["chunked"]["records"]}
    ab_rows = []
    for document_id, direct in direct_by_id.items():
        chunked = chunked_by_id[document_id]
        ab_rows.append(
            {
                "document_id": document_id,
                "category": direct["category"],
                "length_tier": direct["length_tier"],
                "layout": direct["layout"],
                "source_prompt_tokens": direct["source_prompt_tokens"],
                "chunk_count": chunked["chunk_count"],
                "direct_latency_ms": direct["latency_ms"],
                "chunked_latency_ms": chunked["latency_ms"],
                "latency_speedup": direct["latency_ms"] / chunked["latency_ms"],
                "direct_rouge_l": direct["rouge_l_f1"],
                "chunked_rouge_l": chunked["rouge_l_f1"],
                "direct_chrf": direct["chrf"],
                "chunked_chrf": chunked["chrf"],
                "direct_sari": direct["sari"],
                "chunked_sari": chunked["sari"],
                "direct_number_recall": direct["number_recall"],
                "chunked_number_recall": chunked["number_recall"],
                "direct_tail_pass": direct["tail_constraints"]["pass"],
                "chunked_tail_pass": chunked["tail_constraints"]["pass"],
                "direct_all_finished": direct["all_chunks_finished"],
                "chunked_all_finished": chunked["all_chunks_finished"],
            }
        )
    _write_csv(tables_dir / "ab_document_results.csv", ab_rows, list(ab_rows[0]))

    quality_rows = [
        {"mode": "direct", **{k: v for k, v in data["direct"]["summary"].items() if not isinstance(v, (dict, list))}},
        {"mode": "chunked", **{k: v for k, v in data["chunked"]["summary"].items() if not isinstance(v, (dict, list))}},
    ]
    fields = sorted({key for row in quality_rows for key in row})
    _write_csv(tables_dir / "quality_summary.csv", quality_rows, fields)

    tier_rows = []
    for mode in ("direct", "chunked"):
        for tier, summary in data[mode]["by_length_tier"].items():
            tier_rows.append({"mode": mode, "length_tier": tier, **{k: v for k, v in summary.items() if not isinstance(v, (dict, list))}})
    fields = sorted({key for row in tier_rows for key in row})
    _write_csv(tables_dir / "length_tier_summary.csv", tier_rows, fields)

    budget_fields = [
        "document_id", "chunk_budget", "source_prompt_tokens", "chunk_count", "latency_ms",
        "rouge_l_f1", "chrf", "sari", "number_recall", "repetition_ratio",
        "all_chunks_finished", "newline_layout_exact",
    ]
    _write_csv(tables_dir / "budget_sweep.csv", data["budget_sweep"]["records"], budget_fields)
    _write_csv(
        tables_dir / "segmentation_handcrafted.csv",
        data["segmentation_audit"]["handcrafted"],
        ["id", "expected_units", "actual_units", "unit_count_pass", "lossless", "protected_pass", "units"],
    )
    _write_csv(
        tables_dir / "planning_benchmarks.csv",
        data["planning_benchmarks"],
        ["shape", "characters", "chunks", "elapsed_ms", "characters_per_second", "max_chunk_prompt_tokens", "budget_pass"],
    )
    failure_rows = [
        {
            "document_id": row["document_id"],
            "missing": " | ".join(row["constraints"]["missing"]),
            "kept": row["constraints"]["kept"],
            "total": row["constraints"]["total"],
            "number_recall": row["number_recall"],
        }
        for row in data["chunked"]["records"]
        if not row["constraints"]["pass"]
    ]
    _write_csv(tables_dir / "chunked_constraint_failures.csv", failure_rows, list(failure_rows[0]))


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    charts_dir = output_dir / "charts"
    configure_style()
    quality_ab(data, charts_dir)
    reliability_ab(data, charts_dir)
    latency_vs_length(data, charts_dir)
    quality_by_length(data, charts_dir)
    budget_sweep(data, charts_dir)
    segmentation_and_planning(data, charts_dir)
    per_document_delta(data, charts_dir)
    chunk_count_cost(data, charts_dir)
    export_tables(data, output_dir / "tables")
    print(json.dumps({"charts": len(list(charts_dir.glob('*.png'))), "tables": len(list((output_dir/'tables').glob('*.csv'))), "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
