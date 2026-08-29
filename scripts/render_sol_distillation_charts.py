from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

from muddywater.text_simplification.identity_repair import topic_flags
from muddywater.text_simplification.sol_distillation import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GPT-5.6-sol distillation metric charts.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_plotting() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def collect_rows(work_dir: Path) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    combined: list[dict[str, Any]] = []
    shard_rates: list[float] = []
    reduction_rates: list[float] = []
    for shard in manifest["shards"]:
        inputs = read_jsonl(work_dir / shard["input"])
        outputs = read_jsonl(work_dir / shard["output"])
        simplified_count = 0
        for source_row, output_row in zip(inputs, outputs, strict=True):
            source = str(source_row["source"])
            target = str(output_row["target"])
            decision = str(output_row["decision"])
            row = {
                "key": source_row["key"],
                "source": source,
                "target": target,
                "decision": decision,
                "political": bool(topic_flags(source)),
            }
            combined.append(row)
            if decision == "simplified":
                simplified_count += 1
                if source:
                    reduction_rates.append((len(source) - len(target)) / len(source))
        shard_rates.append(simplified_count / len(inputs) if inputs else 0.0)
    return combined, shard_rates, reduction_rates


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, percent: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    bars = axis.bar(labels, values, color=["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"][: len(values)])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    maximum = max(values, default=1.0)
    axis.set_ylim(0, maximum * 1.18 if maximum else 1.0)
    for bar, value in zip(bars, values, strict=True):
        label = f"{value:.1%}" if percent else f"{int(value):,}"
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def render_charts(work_dir: Path, output_dir: Path) -> list[Path]:
    rows, shard_rates, reduction_rates = collect_rows(work_dir)
    assets = output_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    simplified = sum(row["decision"] == "simplified" for row in rows)
    kept = len(rows) - simplified
    political = [row for row in rows if row["political"]]
    non_political = [row for row in rows if not row["political"]]
    political_rate = sum(row["decision"] == "simplified" for row in political) / len(political) if political else 0.0
    non_political_rate = sum(row["decision"] == "simplified" for row in non_political) / len(non_political) if non_political else 0.0

    paths = [
        assets / "decision_distribution.png",
        assets / "identity_rate_before_after.png",
        assets / "compression_distribution.png",
        assets / "political_comparison.png",
        assets / "shard_simplification_rate.png",
    ]
    save_bar(paths[0], ["重新简化", "保留原文"], [simplified, kept], "GPT-5.6-sol 决策分布", "样本数")
    report = json.loads((output_dir / "distillation_report.json").read_text(encoding="utf-8-sig"))
    total_dataset_rows = sum(int(item["rows"]) for item in report["splits"].values())
    save_bar(
        paths[1],
        ["蒸馏前", "蒸馏后"],
        [len(rows) / total_dataset_rows, kept / total_dataset_rows],
        "全数据集 source=target 比例",
        "比例",
        percent=True,
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.hist(reduction_rates, bins=30, color="#4472C4", edgecolor="white")
    axis.axvline(0, color="#C00000", linestyle="--", linewidth=1)
    axis.set_title("重新简化样本的字符压缩率分布")
    axis.set_xlabel("(原文字符数 - 简化字符数) / 原文字符数")
    axis.set_ylabel("样本数")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths[2], bbox_inches="tight")
    plt.close(figure)

    save_bar(
        paths[3],
        ["非政治关键词", "政治关键词"],
        [non_political_rate, political_rate],
        "不同主题的重新简化率",
        "比例",
        percent=True,
    )

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.plot(range(len(shard_rates)), shard_rates, color="#4472C4", linewidth=1.4)
    axis.axhline(sum(shard_rates) / len(shard_rates), color="#ED7D31", linestyle="--", label="分片均值")
    axis.set_title("各分片重新简化率")
    axis.set_xlabel("分片编号")
    axis.set_ylabel("重新简化率")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths[4], bbox_inches="tight")
    plt.close(figure)
    return paths


def append_charts_to_report(output_dir: Path, paths: list[Path]) -> None:
    report_path = output_dir / "distillation_report.md"
    lines = [
        "## 指标图",
        "",
        "![决策分布](report_assets/decision_distribution.png)",
        "",
        "![恒等比例变化](report_assets/identity_rate_before_after.png)",
        "",
        "![字符压缩率](report_assets/compression_distribution.png)",
        "",
        "![主题对比](report_assets/political_comparison.png)",
        "",
        "![分片趋势](report_assets/shard_simplification_rate.png)",
        "",
    ]
    del paths
    existing = report_path.read_text(encoding="utf-8")
    marker = "\n## 指标图\n"
    while marker in existing:
        start = existing.index(marker)
        next_section = existing.find("\n## ", start + len(marker))
        if next_section == -1:
            existing = existing[:start]
        else:
            existing = existing[:start] + existing[next_section:]
    report_path.write_text(existing.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_plotting()
    output_dir = args.output_dir.resolve()
    paths = render_charts(args.work_dir.resolve(), output_dir)
    append_charts_to_report(output_dir, paths)
    print(json.dumps({"charts": [str(path) for path in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
