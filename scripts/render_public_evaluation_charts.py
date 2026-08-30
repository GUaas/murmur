from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "evaluation" / "results" / "version_comparison_summary.json"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "evaluation" / "comparison"
COLORS = {
    "previous": "#6B7280",
    "current": "#0F766E",
    "positive": "#16A34A",
    "negative": "#DC2626",
    "grid": "#E5E7EB",
    "text": "#17324D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render public first-version vs Murmur 203M comparison charts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"scorecard", "core_quality", "reliability", "efficiency", "category_sari"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Comparison summary is missing fields: {sorted(missing)}")
    return payload


def configure_style() -> None:
    for family in (
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "SimHei",
        "Arial Unicode MS",
    ):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = family
        break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.edgecolor": "#B6C2CF",
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_horizontal_bars(ax: plt.Axes, bars: Iterable[Any], digits: int = 1) -> None:
    for bar in bars:
        width = float(bar.get_width())
        ax.text(
            width + ax.get_xlim()[1] * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.{digits}f}",
            va="center",
            fontsize=9,
        )


def render_scorecard(data: dict[str, Any], output_dir: Path) -> None:
    rows = data["scorecard"]
    labels = [row["metric"] for row in rows]
    previous = np.array([row["previous"] for row in rows])
    current = np.array([row["current"] for row in rows])
    positions = np.arange(len(rows))
    height = 0.36
    fig, ax = plt.subplots(figsize=(12, 7))
    left = ax.barh(
        positions + height / 2,
        previous,
        height,
        color=COLORS["previous"],
        label="旧版 140M",
    )
    right = ax.barh(
        positions - height / 2,
        current,
        height,
        color=COLORS["current"],
        label="当前 203M",
    )
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("评分（0–100，越高越好）")
    ax.set_title("旧版 140M vs 当前 203M：加权评分卡")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.legend(loc="lower right")
    label_horizontal_bars(ax, left, digits=2)
    label_horizontal_bars(ax, right, digits=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "01_scorecard.png")


def render_core_quality(data: dict[str, Any], output_dir: Path) -> None:
    core = data["core_quality"]
    keys = ["sari", "rouge_l", "chrf"]
    labels = ["SARI", "ROUGE-L", "chrF"]
    previous = np.array([core["previous"][key] for key in keys])
    current = np.array([core["current"][key] for key in keys])
    positions = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    left = ax.bar(
        positions - width / 2,
        previous,
        width,
        color=COLORS["previous"],
        label="旧版 140M",
    )
    right = ax.bar(
        positions + width / 2,
        current,
        width,
        color=COLORS["current"],
        label="当前 203M",
    )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("字符级指标（越高越好）")
    ax.set_title("51 条独立核心样本：简化质量")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend()
    ax.bar_label(left, fmt="%.3f", padding=3)
    ax.bar_label(right, fmt="%.3f", padding=3)
    sari = core["paired_current_minus_previous"]["sari"]
    ax.text(
        0.5,
        -0.17,
        "当前−旧版 SARI = "
        f"{sari['mean']:+.3f}，95% CI [{sari['lower_95']:+.3f}, {sari['upper_95']:+.3f}]",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "02_core_quality.png")


def render_reliability(data: dict[str, Any], output_dir: Path) -> None:
    rows = data["reliability"]
    labels = [row["metric"] for row in rows]
    previous = np.array([row["previous"] * 100 for row in rows])
    current = np.array([row["current"] * 100 for row in rows])
    positions = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    left = ax.bar(
        positions - width / 2,
        previous,
        width,
        color=COLORS["previous"],
        label="旧版 140M",
    )
    right = ax.bar(
        positions + width / 2,
        current,
        width,
        color=COLORS["current"],
        label="当前 203M",
    )
    ax.set_xticks(positions, labels, rotation=12)
    ax.set_ylim(0, 112)
    ax.set_ylabel("通过率 / 一致性（%）")
    ax.set_title("独立压力集：事实保持与可靠性")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(loc="lower right")
    ax.bar_label(left, fmt="%.1f", padding=3)
    ax.bar_label(right, fmt="%.1f", padding=3)
    fig.tight_layout()
    save_figure(fig, output_dir, "03_reliability.png")


def render_efficiency(data: dict[str, Any], output_dir: Path) -> None:
    rows = data["efficiency"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    flattened = list(axes.flat)
    for ax, row in zip(flattened, rows):
        values = [row["previous"], row["current"]]
        bars = ax.bar(
            ["旧版 140M", "当前 203M"],
            values,
            color=[COLORS["previous"], COLORS["current"]],
        )
        ax.set_title(row["metric"], fontsize=11)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.tick_params(axis="x", labelrotation=12)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_ylim(0, max(values) * 1.22)
    flattened[-1].axis("off")
    fig.suptitle("同机 CPU 性能与资源对比（延迟/资源越低越好，吞吐越高越好）", fontsize=16)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_efficiency.png")


def render_category_delta(data: dict[str, Any], output_dir: Path) -> None:
    rows = data["category_sari"]
    rows = sorted(rows, key=lambda row: row["current"] - row["previous"])
    labels = [row["category"] for row in rows]
    deltas = np.array([row["current"] - row["previous"] for row in rows])
    colors = [COLORS["positive"] if delta >= 0 else COLORS["negative"] for delta in deltas]
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(positions, deltas, color=colors)
    ax.set_yticks(positions, labels)
    ax.axvline(0, color=COLORS["text"], linewidth=1)
    ax.set_xlabel("当前 203M − 旧版 140M 的 SARI")
    ax.set_title("分类差异：绿色为当前版更高，红色为旧版更高")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    for bar, delta in zip(bars, deltas):
        offset = 0.006 if delta >= 0 else -0.006
        ax.text(
            delta + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{delta:+.3f}",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, output_dir, "05_category_delta.png")


def main() -> None:
    args = parse_args()
    configure_style()
    data = load_summary(args.input)
    render_scorecard(data, args.output_dir)
    render_core_quality(data, args.output_dir)
    render_reliability(data, args.output_dir)
    render_efficiency(data, args.output_dir)
    render_category_delta(data, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
