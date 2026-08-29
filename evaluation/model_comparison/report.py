from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .constraints import semantic_constraint_audit
from .paths import EvaluationPaths
from .scoring import DIMENSION_WEIGHTS, paired_statistics, score_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final comparison scorecard.")
    parser.add_argument("--output-dir", default="model_comparison_results")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _category_rows(v1: dict[str, Any], v2: dict[str, Any]) -> list[dict[str, Any]]:
    categories = sorted(set(v1["stress"]["by_category"]) | set(v2["stress"]["by_category"]))
    rows = []
    for category in categories:
        left = v1["stress"]["by_category"][category]
        right = v2["stress"]["by_category"][category]
        rows.append(
            {
                "类别": category,
                "样本数": left["count"],
                "V1_SARI": round(left["sari"], 6),
                "V2_SARI": round(right["sari"], 6),
                "SARI差值_V2-V1": round(right["sari"] - left["sari"], 6),
                "V1_ROUGE-L": round(left["rouge_l_f1"], 6),
                "V2_ROUGE-L": round(right["rouge_l_f1"], 6),
                "V1_压缩比": round(left["compression_ratio"], 6),
                "V2_压缩比": round(right["compression_ratio"], 6),
            }
        )
    return rows


def _largest_differences(v1: dict[str, Any], v2: dict[str, Any], count: int = 8) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded = {"identity", "injection", "long_context", "perturbation"}
    left = {
        row["case_id"]: row
        for row in v1["stress"]["records"]
        if row["case_category"] not in excluded
    }
    right = {
        row["case_id"]: row
        for row in v2["stress"]["records"]
        if row["case_category"] not in excluded
    }
    rows = []
    for case_id in sorted(set(left) & set(right)):
        rows.append(
            {
                "case_id": case_id,
                "category": left[case_id]["case_category"],
                "source": left[case_id]["source"],
                "target": left[case_id]["target"],
                "v1_prediction": left[case_id]["prediction"],
                "v2_prediction": right[case_id]["prediction"],
                "sari_delta_v2_minus_v1": right[case_id]["sari"] - left[case_id]["sari"],
            }
        )
    return (
        sorted(rows, key=lambda row: row["sari_delta_v2_minus_v1"], reverse=True)[:count],
        sorted(rows, key=lambda row: row["sari_delta_v2_minus_v1"])[:count],
    )


def _constraint_review_rows(v1: dict[str, Any], v2: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result, label in ((v1, "第一版"), (v2, "第二版")):
        for record in result["stress"]["records"]:
            if not record["constraints"]["must_keep_total"] or record["case_category"] == "perturbation":
                continue
            audit = semantic_constraint_audit(record)
            rows.append(
                {
                    "模型": label,
                    "案例": record["case_id"],
                    "类别": record["case_category"],
                    "严格通过": audit["strict_pass"],
                    "语义归一后通过": audit["semantic_pass"],
                    "严格缺失": " | ".join(audit["strict_missing"]),
                    "语义缺失": " | ".join(audit["semantic_missing"]),
                    "归一化或同义恢复": " | ".join(
                        item["required"] for item in audit["recovered_by_normalization_or_alias"]
                    ),
                    "输出": record["prediction"],
                }
            )
    return rows


def build_report(paths: EvaluationPaths) -> Path:
    v1 = _read_json(paths.raw_dir / "v1_results.json")
    v2 = _read_json(paths.raw_dir / "v2_results.json")
    scores = score_models([v1, v2])
    paired = paired_statistics(v1, v2)
    winner = max(scores["models"], key=lambda model_id: scores["models"][model_id]["overall"])
    loser = "v1" if winner == "v2" else "v2"
    winner_score = scores["models"][winner]["overall"]
    loser_score = scores["models"][loser]["overall"]

    score_rows = []
    for dimension, weight in DIMENSION_WEIGHTS.items():
        v1_score = scores["models"]["v1"]["dimensions"][dimension]["score"]
        v2_score = scores["models"]["v2"]["dimensions"][dimension]["score"]
        score_rows.append(
            {
                "评分维度": dimension,
                "权重": weight,
                "第一版": v1_score,
                "第二版": v2_score,
                "领先": "第一版" if v1_score > v2_score else ("第二版" if v2_score > v1_score else "持平"),
            }
        )
    score_rows.append(
        {
            "评分维度": "总分",
            "权重": 100,
            "第一版": scores["models"]["v1"]["overall"],
            "第二版": scores["models"]["v2"]["overall"],
            "领先": "第一版" if winner == "v1" else "第二版",
        }
    )
    _write_csv(
        paths.tables_dir / "final_scorecard.csv",
        ["评分维度", "权重", "第一版", "第二版", "领先"],
        score_rows,
    )
    category_rows = _category_rows(v1, v2)
    _write_csv(
        paths.tables_dir / "category_metrics.csv",
        list(category_rows[0]),
        category_rows,
    )
    constraint_rows = _constraint_review_rows(v1, v2)
    _write_csv(
        paths.tables_dir / "constraint_semantic_review.csv",
        list(constraint_rows[0]),
        constraint_rows,
    )

    v1_perf = scores["models"]["v1"]["dimensions"]["推理效率"]["details"]
    v2_perf = scores["models"]["v2"]["dimensions"]["推理效率"]["details"]
    v1_core = scores["models"]["v1"]["dimensions"]["简化质量"]["details"]
    v2_core = scores["models"]["v2"]["dimensions"]["简化质量"]["details"]
    paired_sari = paired["metrics"]["sari"]
    v2_wins, v1_wins = _largest_differences(v1, v2)

    lines = [
        "# 两版中文文本简化模型极致对比评估",
        "",
        f"评测结论：按本报告的均衡权重，**{scores['models'][winner]['display_name']} 小幅领先**，总分 "
        f"**{winner_score:.2f} vs {loser_score:.2f}**，仅差 {winner_score - loser_score:.2f} 分；"
        "应判为**综合近似持平、优势方向不同**，不是全面代际升级。",
        "",
        "## 最终评分表",
        "",
        "| 评分维度 | 权重 | 第一版 | 第二版 | 领先 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in score_rows:
        lines.append(
            f"| {row['评分维度']} | {row['权重']}% | {float(row['第一版']):.2f} | "
            f"{float(row['第二版']):.2f} | {row['领先']} |"
        )
    lines.extend(
        [
            "",
            f"- {winner_score - loser_score:.2f} 分的总分差会随业务权重改变，不能解读为所有场景都由同一版本胜出。",
            "- 总分只使用独立压力集与同机性能数据；第二版训练时使用过的验证集不计入总分。",
            "",
            "## 选型结论",
            "",
            "- **优先选第一版**：主要目标是短中句删冗、正式文本/新闻/口语改写，且在意模型体积、内存、加载速度和吞吐。",
            "- **优先选第二版**：数字、实体、否定和末尾事实不能丢，输入中已有简单句不应被改写，并愿意接受更保守、经常欠简化的输出。",
            "- **长文本暂不建议任一版直接无守门上线**：语义复核后两版都保住了末尾否定事实，但都存在严重跨块复读，尚未解决全局去重与压缩。",
            "",
            "## 核心质量与统计显著性",
            "",
            f"主质量集共 {paired['case_count']} 条，排除了同一性、提示注入、扰动和人工重复长文本，"
            "避免不同任务互相稀释。",
            "",
            "| 指标 | 第一版 | 第二版 | 第二版-第一版 | 95% 配对 Bootstrap CI | 第二版胜率 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| SARI | {v1_core['sari']:.3f} | {v2_core['sari']:.3f} | "
            f"{paired_sari['v2_minus_v1']:+.3f} | "
            f"[{paired_sari['lower_95']:+.3f}, {paired_sari['upper_95']:+.3f}] | "
            f"{paired_sari['v2_win_rate']:.1%} |",
        ]
    )
    for key, label, detail_key in (("rouge_l_f1", "ROUGE-L", "rouge_l"), ("chrf", "chrF", "chrf")):
        stats = paired["metrics"][key]
        lines.append(
            f"| {label} | {v1_core[detail_key]:.3f} | {v2_core[detail_key]:.3f} | "
            f"{stats['v2_minus_v1']:+.3f} | [{stats['lower_95']:+.3f}, {stats['upper_95']:+.3f}] | "
            f"{stats['v2_win_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 客观性能",
            "",
            "以下延迟取同一台 CPU、8 线程、同一批 500 条验证输入；两边均使用随包交付的默认解码策略。",
            "",
            "| 指标 | 第一版 | 第二版 |",
            "|---|---:|---:|",
            f"| 参数量 | {v1['runtime']['model_parameters']:,} | {v2['runtime']['model_parameters']:,} |",
            f"| 权重体积 | {v1_perf['weight_mib']:.1f} MiB | {v2_perf['weight_mib']:.1f} MiB |",
            f"| 模型加载 | {v1_perf['load_seconds']:.3f}s | {v2_perf['load_seconds']:.3f}s |",
            f"| 进程峰值工作集 | {v1_perf['peak_working_set_mib']:.1f} MiB | "
            f"{v2_perf['peak_working_set_mib']:.1f} MiB |",
            f"| 单条延迟 p50 | {v1_perf['latency_p50_ms']:.1f}ms | {v2_perf['latency_p50_ms']:.1f}ms |",
            f"| 单条延迟 p95 | {v1_perf['latency_p95_ms']:.1f}ms | {v2_perf['latency_p95_ms']:.1f}ms |",
            f"| 近似解码吞吐 | {v1_perf['decode_tokens_per_second']:.1f} token/s | "
            f"{v2_perf['decode_tokens_per_second']:.1f} token/s |",
            f"| 原生长文切块 | 100 字符阈值、每块最多 128 token | 160 prompt-token 阈值、模型窗口 896 token |",
            "",
            "## 辅助验证集（不计总分）",
            "",
            "该 500 条样本以固定随机种子从第二版随包验证集抽取。第二版训练选点使用过这份验证集，"
            "因此这里能反映同分布能力，但不能作为两版无偏泛化结论。",
            "",
            "| 指标 | 第一版 | 第二版 |",
            "|---|---:|---:|",
            f"| SARI | {v1['validation']['summary']['sari']:.3f} | {v2['validation']['summary']['sari']:.3f} |",
            f"| ROUGE-L | {v1['validation']['summary']['rouge_l_f1']:.3f} | "
            f"{v2['validation']['summary']['rouge_l_f1']:.3f} |",
            f"| chrF | {v1['validation']['summary']['chrf']:.3f} | {v2['validation']['summary']['chrf']:.3f} |",
            f"| 数字精确率 | {v1['validation']['summary']['number_precision']:.1%} | "
            f"{v2['validation']['summary']['number_precision']:.1%} |",
            f"| 数字召回率 | {v1['validation']['summary']['number_recall']:.1%} | "
            f"{v2['validation']['summary']['number_recall']:.1%} |",
            f"| 原文照抄率 | {v1['validation']['summary']['unchanged_copy_rate']:.1%} | "
            f"{v2['validation']['summary']['unchanged_copy_rate']:.1%} |",
            "",
            "## 关键可靠性指标",
            "",
            "| 指标 | 第一版 | 第二版 |",
            "|---|---:|---:|",
            f"| 独立数字集：数字 F1 | "
            f"{scores['models']['v1']['dimensions']['事实与关键信息保持']['details']['number_f1']:.1%} | "
            f"{scores['models']['v2']['dimensions']['事实与关键信息保持']['details']['number_f1']:.1%} |",
            f"| 语义归一 must-keep 通过率 | "
            f"{scores['models']['v1']['dimensions']['事实与关键信息保持']['details']['must_keep_pass_rate']:.1%} | "
            f"{scores['models']['v2']['dimensions']['事实与关键信息保持']['details']['must_keep_pass_rate']:.1%} |",
            f"| 简单句原样保持 | {v1['stress']['constraints']['identity_exact_preservation_rate']:.1%} | "
            f"{v2['stress']['constraints']['identity_exact_preservation_rate']:.1%} |",
            f"| 扰动输出一致性 | "
            f"{v1['stress']['perturbation_consistency']['overall']['mean']:.1%} | "
            f"{v2['stress']['perturbation_consistency']['overall']['mean']:.1%} |",
            f"| 长文本末尾事实保留（语义归一） | "
            f"{scores['models']['v1']['dimensions']['长文本能力']['details']['must_keep_pass_rate']:.1%} | "
            f"{scores['models']['v2']['dimensions']['长文本能力']['details']['must_keep_pass_rate']:.1%} |",
            "",
            "## 分类结果",
            "",
            "| 类别 | N | 第一版 SARI | 第二版 SARI | 差值 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in category_rows:
        lines.append(
            f"| {row['类别']} | {row['样本数']} | {row['V1_SARI']:.3f} | "
            f"{row['V2_SARI']:.3f} | {row['SARI差值_V2-V1']:+.3f} |"
        )

    def append_examples(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}", ""])
        for row in rows:
            lines.extend(
                [
                    f"### {row['case_id']} / {row['category']} / ΔSARI {row['sari_delta_v2_minus_v1']:+.3f}",
                    "",
                    f"- 原文：{row['source']}",
                    f"- 参考：{row['target']}",
                    f"- 第一版：{row['v1_prediction']}",
                    f"- 第二版：{row['v2_prediction']}",
                ]
            )

    append_examples("第二版优势最大的案例", v2_wins[:5])
    append_examples("第一版优势最大的案例", v1_wins[:5])
    lines.extend(
        [
            "",
            "## 评分公式与限制",
            "",
            "- 简化质量（30%）：独立核心集上的 50% SARI + 30% ROUGE-L + 20% chrF。",
            "- 事实保持（20%）：数字 F1 与语义归一后的 must-keep 约束通过率各占一半。"
            "归一化只处理 Unicode/空白/标点和少量明确同义否定，不放宽数字或实体。",
            "- 简化力度（10%）：压缩比贴近参考答案 60%，非照抄率 40%。",
            "- 鲁棒性（15%）：空格/标点/换行扰动一致性 50%，短句保守性与提示注入各 25%。",
            "- 长文本（10%）：SARI、末尾事实、压缩效率各 25%，低重复 20%，非空 5%。",
            "- 推理效率（10%）：权重 10%、加载 10%、峰值内存 15%、p50 20%、p95 25%、近似吞吐 20%的同机相对分。",
            "- 稳定性（5%）：确定性、非空、无特殊标记泄漏、低重复。",
            "",
            "自动指标不能代替双盲人工评审；中文 SARI/ROUGE/chrF 采用字符级实现，单参考答案也可能惩罚合理改写。"
            "验证集对第二版存在选点偏置，因此仅保留在原始结果中供诊断，不计总分。"
            "第一版原生接口不暴露 EOS/长度终止原因，这一项没有被纳入评分。",
        ]
    )
    report_path = paths.output_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path = paths.raw_dir / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "winner": winner,
                "scores": scores,
                "paired_statistics": paired,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    paths = EvaluationPaths.discover(parse_args().output_dir)
    print(build_report(paths))


if __name__ == "__main__":
    main()
