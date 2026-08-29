from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = PROJECT_ROOT / "output" / "extreme_evaluation_20260814" / "raw" / "extreme_eval_results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "extreme_evaluation_20260814"

NAVY = "#19324D"
TEAL = "#1E8A8A"
BLUE = "#3B82C4"
ORANGE = "#E58A2B"
RED = "#C34A45"
GREEN = "#4E9A68"
GRAY = "#7B8794"
LIGHT = "#E8EEF3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate charts and CSV tables for the extreme evaluation.")
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def configure_style() -> None:
    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = family
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
            "grid.alpha": 0.9,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _annotate_bars(ax, bars, *, digits: int = 3) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=NAVY,
        )


def quality_overview(data: dict[str, Any], charts_dir: Path) -> None:
    validation = data["validation"]["summary"]
    copy = data["validation"]["copy_baseline_summary"]
    stress = data["stress"]["summary"]
    labels = ["ROUGE-L", "chrF", "SARI"]
    keys = ["rouge_l_f1", "chrf", "sari"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    bars1 = ax.bar(x - width, [validation[key] for key in keys], width, label="验证集复核 (500)", color=BLUE)
    bars2 = ax.bar(x, [copy[key] for key in keys], width, label="原样复制基线", color=GRAY)
    bars3 = ax.bar(x + width, [stress[key] for key in keys], width, label="独立压力集 (135)", color=ORANGE)
    _annotate_bars(ax, bars1)
    _annotate_bars(ax, bars2)
    _annotate_bars(ax, bars3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("得分 (越高越好)")
    ax.set_title("质量总览：同分布表现强，独立压力集明显回落")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    _save(fig, charts_dir / "01_quality_overview.png")


def length_degradation(data: dict[str, Any], charts_dir: Path) -> None:
    groups = data["validation"]["stratified"]["source_length"]
    ordered = sorted(groups.items())
    labels = [name.split("_", 1)[1].replace("<=", "≤").replace(">", ">") for name, _ in ordered]
    rouge = [summary["rouge_l_f1"] for _, summary in ordered]
    sari = [summary["sari"] for _, summary in ordered]
    latency = [summary["avg_latency_ms"] / 1000.0 for _, summary in ordered]
    repetition = [summary["repetition_ratio"] for _, summary in ordered]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    ax = axes[0]
    ax.plot(x, rouge, marker="o", linewidth=2.2, color=BLUE, label="ROUGE-L")
    ax.plot(x, sari, marker="o", linewidth=2.2, color=TEAL, label="SARI")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.55, 1.0)
    ax.set_xlabel("输入字符数")
    ax.set_ylabel("质量得分")
    ax.set_title("质量随输入长度下降")
    ax.legend()
    ax = axes[1]
    bars = ax.bar(x, latency, color=ORANGE, alpha=0.88, label="平均延迟")
    ax.set_xticks(x, labels)
    ax.set_xlabel("输入字符数")
    ax.set_ylabel("平均延迟 (秒)")
    ax.set_title("长文本的延迟与重复率上升")
    ax2 = ax.twinx()
    ax2.plot(x, repetition, color=RED, marker="D", linewidth=2, label="重复率")
    ax2.set_ylabel("4-gram 重复率", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, f"{bar.get_height():.1f}s", ha="center", fontsize=8)
    fig.suptitle("长度分层：>256 字符后风险加速", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, charts_dir / "02_length_degradation.png")


def stress_categories(data: dict[str, Any], charts_dir: Path) -> None:
    categories = data["stress"]["by_category"]
    excluded = {"perturbation", "identity"}
    rows = [
        (name, summary)
        for name, summary in categories.items()
        if name not in excluded
    ]
    rows.sort(key=lambda item: item[1]["sari"])
    translations = {
        "academic": "学术",
        "entity": "实体",
        "formal": "正式文体",
        "injection": "提示注入",
        "legal": "法律",
        "long_context": "长文本",
        "mixed": "中英混合",
        "negation": "否定",
        "news": "新闻",
        "noisy": "噪声文本",
        "numbers": "数字",
        "spoken": "口语冗余",
        "technical": "技术",
        "traditional": "繁体",
    }
    labels = [translations.get(name, name) for name, _ in rows]
    sari = [summary["sari"] for _, summary in rows]
    copy_rate = [summary["unchanged_copy_rate"] for _, summary in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    colors = [RED if value < 0.4 else ORANGE if value < 0.55 else TEAL for value in sari]
    bars = ax.barh(y, sari, color=colors, alpha=0.9, label="SARI")
    ax.scatter(copy_rate, y, marker="D", color=NAVY, s=38, label="原样复制率", zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("得分 / 比率")
    ax.set_title("独立压力集分类型表现")
    ax.legend(loc="upper right")
    for bar, value in zip(bars, sari):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    _save(fig, charts_dir / "03_stress_categories.png")


def latency_distribution(data: dict[str, Any], charts_dir: Path) -> None:
    validation = np.asarray([row["latency_ms"] for row in data["validation"]["records"]]) / 1000.0
    stress = np.asarray([row["latency_ms"] for row in data["stress"]["records"]]) / 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    bins = np.linspace(0, min(12, max(validation.max(), stress.max())), 32)
    axes[0].hist(validation, bins=bins, color=BLUE, alpha=0.8, label="验证集复核")
    axes[0].hist(stress, bins=bins, color=ORANGE, alpha=0.55, label="独立压力集")
    axes[0].set_xlabel("单条延迟 (秒)")
    axes[0].set_ylabel("样本数")
    axes[0].set_title("延迟分布（截取 12 秒内）")
    axes[0].legend()
    for values, label, color in ((validation, "验证集复核", BLUE), (stress, "独立压力集", ORANGE)):
        sorted_values = np.sort(values)
        percentile = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axes[1].plot(sorted_values, percentile, color=color, linewidth=2.2, label=label)
    axes[1].set_xlim(0, 12)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_xlabel("单条延迟 (秒)")
    axes[1].set_ylabel("累计比例")
    axes[1].set_title("延迟累计分布 (CDF)")
    axes[1].legend()
    fig.suptitle("CPU 端到端延迟：长尾明显", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, charts_dir / "04_latency_distribution.png")


def performance_scaling(data: dict[str, Any], charts_dir: Path) -> None:
    micro = data["microbench"]
    seq = micro["forward_by_sequence_length"]
    batch = micro["forward_by_batch_size"]
    threads = micro["forward_by_thread_count"]
    cache = micro["kv_cache_comparison"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    ax = axes[0, 0]
    ax.plot([row["sequence_length"] for row in seq], [row["latency_ms"] for row in seq], color=BLUE, marker="o", linewidth=2.2)
    ax.set_xlabel("序列长度 (tokens)")
    ax.set_ylabel("前向延迟 (ms)")
    ax.set_title("序列长度扩展")
    ax = axes[0, 1]
    bars = ax.bar([str(row["batch_size"]) for row in batch], [row["tokens_per_second"] for row in batch], color=[BLUE, TEAL, GREEN])
    ax.set_xlabel("Batch size (seq=128)")
    ax.set_ylabel("前向吞吐 (tokens/s)")
    ax.set_title("批量吞吐")
    _annotate_bars(ax, bars, digits=0)
    ax = axes[1, 0]
    ax.plot([row["threads"] for row in threads], [row["tokens_per_second"] for row in threads], color=TEAL, marker="o", linewidth=2.2)
    ax.set_xlabel("PyTorch 线程数")
    ax.set_ylabel("前向吞吐 (tokens/s)")
    ax.set_title("线程伸缩（本机 4 线程最佳）")
    ax = axes[1, 1]
    cached = np.mean([row["cached_ms"] for row in cache])
    uncached = np.mean([row["uncached_ms"] for row in cache])
    bars = ax.bar(["KV cache 开", "KV cache 关"], [cached, uncached], color=[GREEN, RED])
    ax.set_ylabel("平均生成延迟 (ms)")
    ax.set_title(f"KV cache 平均加速 {micro['kv_cache_mean_speedup']:.2f}×")
    _annotate_bars(ax, bars, digits=0)
    fig.suptitle("CPU 性能微基准", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, charts_dir / "05_performance_scaling.png")


def reliability_scorecard(data: dict[str, Any], charts_dir: Path) -> None:
    validation = data["validation"]["summary"]
    constraints = data["stress"]["constraints"]
    perturb = data["stress"]["perturbation_consistency"]["overall"]["mean"]
    items = [
        ("正常结束率", validation["eos_or_stop_hit_rate"]),
        ("数字召回", validation["number_recall"]),
        ("关键信息约束", constraints["must_keep_pass_rate"]),
        ("简单句不乱改", constraints["identity_exact_preservation_rate"]),
        ("基础注入未执行", constraints["injection_forbidden_exact_pass_rate"]),
        ("扰动输出一致性", perturb),
        ("重复确定性", data["determinism"]["exact_determinism_rate"]),
    ]
    labels = [item[0] for item in items]
    values = [float(item[1]) for item in items]
    colors = [GREEN if value >= 0.95 else ORANGE if value >= 0.8 else RED for value in values]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.barh(np.arange(len(items)), values, color=colors)
    ax.set_yticks(np.arange(len(items)), labels)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("通过率 / 一致性")
    ax.set_title("可靠性指标卡（注意：注入项仅为基础行为测试）")
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9)
    _save(fig, charts_dir / "06_reliability_scorecard.png")


def training_curve(charts_dir: Path) -> None:
    log_path = PROJECT_ROOT / "outputs" / "sft_203m_text_simplification_pass_filtered" / "train.log"
    train_steps: list[int] = []
    train_loss: list[float] = []
    learning_rates: list[float] = []
    eval_steps: list[int] = []
    eval_loss: list[float] = []
    train_pattern = re.compile(r"step=(\d+) ce_loss=([0-9.]+).*?lr=([0-9.eE+-]+)")
    eval_pattern = re.compile(r"(?:eval step|final eval step)=(\d+) val_ce_loss=([0-9.]+)")
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = train_pattern.search(line)
        if match:
            train_steps.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            learning_rates.append(float(match.group(3)))
        match = eval_pattern.search(line)
        if match:
            eval_steps.append(int(match.group(1)))
            eval_loss.append(float(match.group(2)))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    axes[0].plot(train_steps, train_loss, color=BLUE, linewidth=1.8, label="训练 CE")
    axes[0].set_xlabel("训练步")
    axes[0].set_ylabel("CE loss")
    axes[0].set_title("训练损失持续下降")
    ax_lr = axes[0].twinx()
    ax_lr.plot(train_steps, learning_rates, color=GRAY, alpha=0.6, linewidth=1.2, label="学习率")
    ax_lr.set_ylabel("学习率", color=GRAY)
    axes[1].plot(eval_steps, eval_loss, color=ORANGE, marker="o", linewidth=2.2)
    if eval_loss:
        best_index = int(np.argmin(eval_loss))
        axes[1].scatter([eval_steps[best_index]], [eval_loss[best_index]], color=GREEN, s=70, zorder=3)
        axes[1].annotate(
            f"最佳 step={eval_steps[best_index]}\nloss={eval_loss[best_index]:.4f}",
            (eval_steps[best_index], eval_loss[best_index]),
            xytext=(18, 14),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": GREEN},
        )
    axes[1].set_xlabel("训练步")
    axes[1].set_ylabel("验证 CE loss")
    axes[1].set_title("step 350 后出现过拟合")
    fig.suptitle("训练历史：最佳权重选择合理，但最终轮验证损失回升", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, charts_dir / "07_training_curve.png")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_tables(data: dict[str, Any], tables_dir: Path) -> None:
    quality_rows = []
    for name, summary in (
        ("validation_model_500", data["validation"]["summary"]),
        ("validation_copy_baseline", data["validation"]["copy_baseline_summary"]),
        ("independent_stress_135", data["stress"]["summary"]),
    ):
        quality_rows.append({"dataset": name, **{key: value for key, value in summary.items() if not isinstance(value, (dict, list))}})
    _write_csv(tables_dir / "quality_summary.csv", quality_rows, list(quality_rows[0].keys()))

    category_rows = [{"category": name, **summary} for name, summary in data["stress"]["by_category"].items()]
    _write_csv(tables_dir / "stress_category_summary.csv", category_rows, list(category_rows[0].keys()))

    performance_rows = []
    for row in data["microbench"]["forward_by_sequence_length"]:
        performance_rows.append({"benchmark": "sequence_length", **row})
    for row in data["microbench"]["forward_by_batch_size"]:
        performance_rows.append({"benchmark": "batch_size", **row})
    for row in data["microbench"]["forward_by_thread_count"]:
        performance_rows.append({"benchmark": "thread_count", **row})
    fields = sorted({key for row in performance_rows for key in row})
    _write_csv(tables_dir / "performance_microbench.csv", performance_rows, fields)

    stress_fields = [
        "case_id", "case_category", "source", "target", "prediction", "rouge_l_f1", "chrf", "sari",
        "compression_ratio", "finish_reason", "generated_tokens", "latency_ms", "repetition_ratio",
    ]
    _write_csv(tables_dir / "stress_case_results.csv", data["stress"]["records"], stress_fields)
    validation_fields = [
        "index", "source", "target", "prediction", "rouge_l_f1", "chrf", "sari", "compression_ratio",
        "number_precision", "number_recall", "finish_reason", "generated_tokens", "latency_ms", "repetition_ratio",
    ]
    _write_csv(tables_dir / "validation_review_500.csv", data["validation"]["records"], validation_fields)


def main() -> None:
    args = parse_args()
    result_path = Path(args.result).resolve()
    output_dir = Path(args.output_dir).resolve()
    charts_dir = output_dir / "charts"
    tables_dir = output_dir / "tables"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    configure_style()
    quality_overview(data, charts_dir)
    length_degradation(data, charts_dir)
    stress_categories(data, charts_dir)
    latency_distribution(data, charts_dir)
    performance_scaling(data, charts_dir)
    reliability_scorecard(data, charts_dir)
    training_curve(charts_dir)
    export_tables(data, tables_dir)
    print(json.dumps({"charts": len(list(charts_dir.glob('*.png'))), "tables": len(list(tables_dir.glob('*.csv'))), "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
