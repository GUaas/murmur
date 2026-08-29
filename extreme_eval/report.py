from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "extreme_evaluation_20260814"
DEFAULT_RESULT = DEFAULT_OUTPUT / "raw" / "extreme_eval_results.json"
DEFAULT_PREFLIGHT = DEFAULT_OUTPUT / "raw" / "preflight_observations.json"
DEFAULT_PDF = PROJECT_ROOT / "output" / "pdf" / "murmur_203m_text_simplification_extreme_evaluation_20260814.pdf"

NAVY = colors.HexColor("#19324D")
TEAL = colors.HexColor("#1E8A8A")
BLUE = colors.HexColor("#3B82C4")
ORANGE = colors.HexColor("#E58A2B")
RED = colors.HexColor("#C34A45")
GREEN = colors.HexColor("#4E9A68")
GRAY = colors.HexColor("#7B8794")
LIGHT = colors.HexColor("#E8EEF3")
PALE_BLUE = colors.HexColor("#EEF5FB")
PALE_ORANGE = colors.HexColor("#FFF3E7")
PALE_RED = colors.HexColor("#FCECEB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Markdown and PDF reports for the extreme evaluation.")
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--preflight", default=str(DEFAULT_PREFLIGHT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    return parser.parse_args()


def pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{100 * float(value):.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def seconds_from_ms(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value) / 1000.0:.{digits}f} s"


def gib(value: int | None) -> str:
    return "-" if value is None else f"{float(value) / (1024 ** 3):.2f} GiB"


def _ci(summary: dict[str, Any], key: str) -> str:
    ci = summary["bootstrap_95_ci"][key]
    return f"{ci['lower_95']:.3f}-{ci['upper_95']:.3f}"


def build_markdown(data: dict[str, Any], preflight: dict[str, Any], output_dir: Path) -> str:
    validation = data["validation"]["summary"]
    copy = data["validation"]["copy_baseline_summary"]
    stress = data["stress"]["summary"]
    constraints = data["stress"]["constraints"]
    micro = data["microbench"]
    audit = data["dataset_audit"]
    runtime = data["runtime"]
    training = data["historical"]["training_summary"]
    long_rows = [row for row in data["stress"]["records"] if row["case_category"] == "long_context"]
    must_keep_failures = [
        row for row in data["stress"]["records"]
        if row["constraints"]["must_keep_total"] and not row["constraints"]["must_keep_pass"]
    ]
    lines = [
        "# Murmur 203M 中文文本简化模型极限评测报告",
        "",
        "**评测日期：** 2026-08-14  ",
        "**总体结论：** **B- / 条件通过**。适合短到中等长度、一般风险的中文简化；上线前必须增加数字/实体校验、长文本分块、重复检测和超时/长度保护。不得把当前结果视为高风险事实文本或任意长文本的无条件放行。",
        "",
        "> 重要口径：打包验证集参与过最佳权重选择，因此这里只复核 500 条，不作为独立泛化结论。真正的域外证据来自 135 条独立压力样例。",
        "",
        "## 一页结论",
        "",
        f"- 同分布验证复核：ROUGE-L {validation['rouge_l_f1']:.3f}、chrF {validation['chrf']:.3f}、SARI {validation['sari']:.3f}；SARI 比原样复制基线高 {data['validation']['paired_vs_copy']['sari']['mean_delta']:+.3f}。",
        f"- 独立压力集：ROUGE-L {stress['rouge_l_f1']:.3f}、SARI {stress['sari']:.3f}，输出压缩比 {stress['compression_ratio']:.3f}，而参考答案为 {stress['reference_compression_ratio']:.3f}，显示明显欠简化。",
        f"- 可靠性：简单句原样保持 {pct(constraints['identity_exact_preservation_rate'])}；关键信息约束通过 {pct(constraints['must_keep_pass_rate'])}，排除 3 个长文本失效后为 95.0%。",
        f"- CPU 性能：500 条稳态平均 {seconds_from_ms(validation['avg_latency_ms'])}/条，p95 {seconds_from_ms(validation['latency_ms_percentiles']['p95'])}，生成吞吐 {validation['decode_tokens_per_second']:.2f} tokens/s；进程内加载 {runtime['load_seconds']:.2f} s，冷启动命令行 {preflight['smoke_inference']['cold_cli_wall_seconds']:.3f} s。",
        f"- 长文本：重复型输入从 243 prompt tokens 起出现强复读；475/823 tokens 时均触及 256-token 输出上限，EOS 命中率仅 50%。",
        "",
        "![质量总览](charts/01_quality_overview.png)",
        "",
        "## 1. 范围与方法",
        "",
        "本次覆盖：归档安全与 SHA-256 完整性、源码测试、数据泄漏/近重复、500 条验证集复核、135 条独立压力样例、24 次确定性重复、CPU 延迟/吞吐/内存、序列长度/批量/线程/KV cache 微基准，以及训练历史与过拟合检查。",
        "",
        f"归档包含 {preflight['archive']['entries']} 个条目，无危险路径和链接；便携包 {preflight['bundle_verification']['hashes_checked']} 个文件哈希全部通过。权重 SHA-256 为 `{runtime['checkpoint_sha256']}`。",
        "",
        "## 2. 质量报表",
        "",
        "| 集合 | N | ROUGE-L | chrF | SARI | 压缩比 | 原样复制率 | EOS/Stop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| 验证集复核 | 500 | {validation['rouge_l_f1']:.3f} | {validation['chrf']:.3f} | {validation['sari']:.3f} | {validation['compression_ratio']:.3f} | {pct(validation['unchanged_copy_rate'])} | {pct(validation['eos_or_stop_hit_rate'])} |",
        f"| 原样复制基线 | 500 | {copy['rouge_l_f1']:.3f} | {copy['chrf']:.3f} | {copy['sari']:.3f} | 1.000 | 100.0% | - |",
        f"| 独立压力集 | 135 | {stress['rouge_l_f1']:.3f} | {stress['chrf']:.3f} | {stress['sari']:.3f} | {stress['compression_ratio']:.3f} | {pct(stress['unchanged_copy_rate'])} | {pct(stress['eos_or_stop_hit_rate'])} |",
        "",
        f"验证集 SARI 的 95% bootstrap CI 为 {_ci(validation, 'sari')}；独立压力集为 {_ci(stress, 'sari')}。验证集指标较高，但其参考目标平均压缩比 {validation['reference_compression_ratio']:.3f}，本身接近原文；因此 ROUGE-L/chrF 会高估“复制输入”的能力，SARI 和相对复制基线更有解释力。",
        "",
        "![长度退化](charts/02_length_degradation.png)",
        "",
        "## 3. 独立压力与鲁棒性",
        "",
        f"独立压力集原样复制率为 {pct(stress['unchanged_copy_rate'])}，法律样例为 100%，数字样例为 75%。模型在域外数据上常选择保守复制，而不是执行充分简化。空格扰动输出一致性为 {data['stress']['perturbation_consistency']['by_perturbation']['spaces']['mean']:.3f}，去标点为 {data['stress']['perturbation_consistency']['by_perturbation']['punct_removed']['mean']:.3f}，换行为 {data['stress']['perturbation_consistency']['by_perturbation']['linebreaks']['mean']:.3f}。",
        "",
        "![压力集分类](charts/03_stress_categories.png)",
        "",
        "![可靠性指标](charts/06_reliability_scorecard.png)",
        "",
        "## 4. 事实性与失败案例",
        "",
        f"验证集数字精确率/召回率为 {pct(validation['number_precision'])}/{pct(validation['number_recall'])}，低于原样复制基线的 {pct(copy['number_precision'])}/{pct(copy['number_recall'])}。独立样例 `num_03` 删除了原发车时间 `14:35`，只保留 `15:10`；验证集还出现删除年份 `1934`、删除多个比例数字等情况。",
        "",
        "关键信息约束失败共 4/23：`num_03` 丢失 14:35，另外 3 个均为长重复文本丢失末尾否定事实“不会取消”。基础注入样例 8/8 未退化成攻击者要求的单一 payload，但多数只是复制或轻改指令文本；这不是完整安全评估。",
        "",
        "## 5. 长文本边界",
        "",
        "| case | prompt tokens | 生成 tokens | 结束原因 | 重复率 | 关键事实 | 延迟 |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for row in long_rows:
        missing = "通过" if not row["constraints"]["must_keep_missing"] else "丢失: " + ",".join(row["constraints"]["must_keep_missing"])
        lines.append(
            f"| {row['case_id']} | {row['prompt_tokens']} | {row['generated_tokens']} | {row['finish_reason']} | {row['repetition_ratio']:.3f} | {missing} | {row['latency_ms'] / 1000:.2f}s |"
        )
    lines.extend(
        [
            "",
            "结论：配置支持 896 tokens，不等于在整个窗口内都具有可靠简化能力。对重复型输入，实际可靠边界远低于配置上限。生产环境应在约 200 tokens 以上启用分块/摘要策略，并对 4-gram 重复率、EOS、输出长度和末尾事实做守门。",
            "",
            "## 6. 性能报表",
            "",
            f"测试机为 13th Gen Intel Core i5-13420H、16 GiB RAM、CPU 版 PyTorch 2.12.0，无 CUDA/ROCm。模型参数 {runtime['model_parameters']:,}，权重 {runtime['checkpoint_bytes'] / (1024**2):.1f} MiB；加载后工作集 {gib(runtime['memory_after_load']['working_set'])}，峰值 {gib(data['microbench']['memory_after_microbench']['peak_working_set'])}。",
            "",
            f"- 验证集稳态平均 {seconds_from_ms(validation['avg_latency_ms'])}，p50 {seconds_from_ms(validation['latency_ms_percentiles']['p50'])}，p95 {seconds_from_ms(validation['latency_ms_percentiles']['p95'])}，p99 {seconds_from_ms(validation['latency_ms_percentiles']['p99'])}。",
            f"- KV cache 平均加速 {micro['kv_cache_mean_speedup']:.2f}x，6/6 输出完全一致。",
            f"- batch=4、seq=128 的前向吞吐 {micro['forward_by_batch_size'][-1]['tokens_per_second']:.0f} tokens/s，比 batch=1 提升 {(micro['forward_by_batch_size'][-1]['tokens_per_second']/micro['forward_by_batch_size'][0]['tokens_per_second']-1)*100:.1f}%，但单批延迟更高。",
            f"- 本机线程测试中 4 线程最佳 ({max(micro['forward_by_thread_count'], key=lambda row: row['tokens_per_second'])['tokens_per_second']:.0f} tokens/s)，8 线程并非最优，混合大小核调度会影响结果。",
            "",
            "![性能微基准](charts/05_performance_scaling.png)",
            "",
            "![延迟分布](charts/04_latency_distribution.png)",
            "",
            "## 7. 数据、训练与工程质量",
            "",
            f"数据共 59,419 训练 + 3,119 验证。严格源文本无交集，但 NFKC/去标点归一化后有 {audit['normalized_source_overlap']} 条重合；字符 4-gram Jaccard >=0.85 的近重复有 {audit['near_duplicate']['near_match_count']} 条 ({pct(audit['near_duplicate']['near_match_rate'], 2)})。这会轻微抬高验证结果。",
            "",
            f"训练最佳验证 loss {training['best_val_loss']:.4f} 出现在 step 350 附近，最终 loss {training['final_metrics']['loss']:.4f}，较最佳点回升 {(training['final_metrics']['loss']/training['best_val_loss']-1)*100:.1f}%。交付的是最佳权重而非最终轮，选择正确。",
            "",
            "![训练曲线](charts/07_training_curve.png)",
            "",
            f"源码测试 70/71 通过。唯一失败是便携包保留了 540M 预训练变体对比测试，却未携带所需的两个配置文件。`run_simplify.ps1` 可正常推理；但 `run_eval_portable.sh` 引用不存在的 `best.pt` 和拆分包中不存在的基础预训练权重，不能按说明直接复评。`README_00_SPLIT_PACKAGE_ZH.md` 正确说明拆包结构，但 `README_PORTABLE_ZH.md` 仍声称已包含基础预训练权重，文档互相矛盾。",
            "",
            "## 8. 优点",
            "",
            "1. 同分布质量强，SARI 相对复制基线提升明显，且 64.6% 样本优于复制基线。",
            "2. 简单句保守性好，独立 12/12 完全不乱改；无空输出、无保留标签泄漏。",
            "3. 短中等输入的数字与实体大多能保留，非长文本关键信息约束通过 19/20。",
            "4. 贪心解码完全确定，8 组各 3 次输出一致；KV cache 开关输出一致。",
            "5. 203M 规模在 CPU 可运行，内存约 1.6-1.9 GiB，KV cache 加速有效。",
            "6. 交付权重哈希、Tokenizer 哈希和 392 个文件完整性全部通过。",
            "",
            "## 9. 缺点",
            "",
            "1. 独立压力集显著欠简化：输出压缩比 0.923，而参考为 0.700；法律和数字类别大量复制。",
            "2. 事实保护不够硬：数字召回低于复制基线，存在删除年份、时间和比例列表的案例。",
            "3. 长重复文本失效严重：复读、遗漏末尾否定事实、触及长度上限且不产生 EOS。",
            "4. 域外类别样本较少，压力集 135 条只能做风险发现，不能替代大规模独立盲测。",
            "5. 验证集用于选点且有少量近重复，已有高分不能当作无偏泛化估计。",
            "6. 便携复评入口、测试依赖和 README 存在打包一致性问题。",
            "7. 权重为 FP32，812 MiB；未提供量化或 ONNX/DirectML 部署路径。",
            "",
            "## 10. 上线建议",
            "",
            "- 推荐场景：短到中等中文句段、低到中等事实风险、允许保守复制的离线或交互式简化。",
            "- 必须守门：数字/日期/单位/否定词差异检测；实体保留；输出为空/EOS/长度上限；重复率；输入 token 长度。",
            "- 长文本：超过约 200 tokens 时分块，保留末尾窗口事实，并对跨块实体与数字做合并校验。",
            "- 性能：保持 KV cache；本机优先尝试 4 线程；批处理只用于吞吐优先场景。",
            "- 发布前：另建冻结的独立盲测集，至少 1,000 条，按新闻/法律/技术/口语/数字/否定/长文本分层，并加入人工事实性评分。",
            "- 工程：修复 portable eval 的 checkpoint 路径，删除或补齐不相关测试配置，统一两份 README 的拆包说明。",
            "",
            "## 11. 口径与限制",
            "",
            "ROUGE-L/chrF 衡量与单一参考答案的表面相似度，无法证明语义完全等价；SARI 为中文字符 n-gram 版本，也不能替代人工可读性与事实性审查。基础注入测试只验证模型是否直接执行少数 payload，不构成安全认证。CPU 数据仅代表本机，不外推到 GPU。",
            "",
            "## 产物索引",
            "",
            "- 原始结果：`raw/extreme_eval_results.json`",
            "- 预检证据：`raw/preflight_observations.json`",
            "- 指标图片：`charts/`",
            "- CSV 报表：`tables/`",
            "- 可复现代码：项目根目录 `extreme_eval/` 与 `run_extreme_eval.py`",
            "",
        ]
    )
    markdown = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    return markdown


def register_fonts() -> tuple[str, str]:
    regular_path = Path("C:/Windows/Fonts/simhei.ttf")
    if not regular_path.exists():
        raise FileNotFoundError(f"Chinese font not found: {regular_path}")
    pdfmetrics.registerFont(TTFont("ReportChinese", str(regular_path)))
    return "ReportChinese", "ReportChinese"


def build_styles(font_name: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontName=bold_font, fontSize=26, leading=34,
            textColor=NAVY, alignment=TA_LEFT, spaceAfter=14,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Normal"], fontName=font_name, fontSize=12, leading=20,
            textColor=GRAY, alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName=bold_font, fontSize=18, leading=24,
            textColor=NAVY, spaceBefore=3, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName=bold_font, fontSize=13, leading=18,
            textColor=TEAL, spaceBefore=8, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName=font_name, fontSize=9.4, leading=15,
            textColor=NAVY, spaceAfter=6, alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName=font_name, fontSize=7.8, leading=11,
            textColor=GRAY, spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "Table", parent=base["BodyText"], fontName=font_name, fontSize=7.5, leading=10,
            textColor=NAVY, alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=base["BodyText"], fontName=bold_font, fontSize=7.5, leading=10,
            textColor=colors.white, alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName=font_name, fontSize=10.2, leading=16,
            textColor=NAVY, leftIndent=5, rightIndent=5, spaceBefore=5, spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName=font_name, fontSize=9.2, leading=14,
            textColor=NAVY, leftIndent=14, firstLineIndent=-8, bulletIndent=3, spaceAfter=4,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def callout(text: str, styles: dict[str, ParagraphStyle], *, background=PALE_BLUE, border=BLUE) -> Table:
    table = Table([[para(text, styles["callout"])]], colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 1, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def bullet_items(items: Iterable[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [Paragraph(f"- {item}", styles["bullet"]) for item in items]


def make_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    styles: dict[str, ParagraphStyle],
    *,
    font_size: float = 7.5,
) -> Table:
    table_style = styles["table"]
    if font_size != table_style.fontSize:
        table_style = ParagraphStyle("TableDynamic", parent=table_style, fontSize=font_size, leading=font_size + 2.5)
    payload = [[para(html.escape(str(value)), styles["table_header"]) for value in headers]]
    for row in rows:
        payload.append([para(html.escape(str(value)), table_style) for value in row])
    table = Table(payload, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD4DD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def chart(path: Path, *, max_width: float = 174 * mm, max_height: float = 110 * mm) -> Image:
    width, height = ImageReader(str(path)).getSize()
    scale = min(max_width / width, max_height / height)
    image = Image(str(path), width=width * scale, height=height * scale)
    image.hAlign = "CENTER"
    return image


def page_decor(canvas, document) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, height - 15 * mm, width - 20 * mm, height - 15 * mm)
    canvas.setFont("ReportChinese", 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(20 * mm, height - 11 * mm, "Murmur 203M 中文文本简化模型极限评测")
    canvas.drawRightString(width - 20 * mm, 10 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def build_pdf(data: dict[str, Any], preflight: dict[str, Any], output_dir: Path, pdf_path: Path) -> None:
    font_name, bold_font = register_fonts()
    styles = build_styles(font_name, bold_font)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=16 * mm,
        title="Murmur 203M 中文文本简化模型极限评测报告",
        author="Codex",
        subject="质量、鲁棒性、性能、数据与工程完整性评估",
    )
    story: list[Any] = []
    charts_dir = output_dir / "charts"
    validation = data["validation"]["summary"]
    copy = data["validation"]["copy_baseline_summary"]
    stress = data["stress"]["summary"]
    constraints = data["stress"]["constraints"]
    audit = data["dataset_audit"]
    micro = data["microbench"]
    runtime = data["runtime"]
    training = data["historical"]["training_summary"]

    # Cover
    story.extend(
        [
            Spacer(1, 23 * mm),
            para("Murmur 203M", styles["cover_subtitle"]),
            para("中文文本简化模型<br/>极限评测报告", styles["cover_title"]),
            Spacer(1, 4 * mm),
            callout(
                "<b>总体评级：B- / 条件通过</b><br/>适合短到中等长度、一般风险的中文简化；生产部署必须加入事实校验、长文本分块、重复检测和长度保护。",
                styles,
                background=PALE_ORANGE,
                border=ORANGE,
            ),
            Spacer(1, 10 * mm),
            make_table(
                ["质量", "可靠性", "CPU 性能", "工程完整性"],
                [["同分布 A- / 域外 C+", "短文本 B+ / 长文本 F", "可用但有长尾", "主体可用，复评入口需修复"]],
                [43.5 * mm] * 4,
                styles,
                font_size=8.5,
            ),
            Spacer(1, 14 * mm),
            para("评测日期：2026-08-14", styles["cover_subtitle"]),
            para("模型：203,037,056 参数 | 权重：812,221,131 bytes | 设备：Intel i5-13420H CPU", styles["cover_subtitle"]),
            para("证据：500 条验证复核 + 135 条独立压力样例 + 24 次确定性重复 + 微基准 + 数据/代码/归档审计", styles["cover_subtitle"]),
            Spacer(1, 15 * mm),
            para("重要说明：验证集参与过最佳权重选择，因此验证复核不作为独立泛化结论。", styles["small"]),
            PageBreak(),
        ]
    )

    # Executive summary
    story.extend(
        [
            para("执行摘要", styles["h1"]),
            callout(
                f"<b>结论：</b>模型在同分布短中等文本上效果好（SARI {validation['sari']:.3f}），但独立压力集降至 {stress['sari']:.3f}，并出现欠简化、数字删除和长文本复读。可上线，但只能在防护措施到位后条件放行。",
                styles,
            ),
            Spacer(1, 3 * mm),
        ]
    )
    story += bullet_items(
        [
            f"验证复核 SARI 比原样复制高 {data['validation']['paired_vs_copy']['sari']['mean_delta']:+.3f}；64.6% 样本获胜，9.4% 退步。",
            f"独立压力集输出压缩比 {stress['compression_ratio']:.3f}，参考为 {stress['reference_compression_ratio']:.3f}，说明域外输入明显欠简化。",
            f"简单句 12/12 完全不乱改；基础注入 8/8 未变成单一攻击 payload；8 组重复生成完全一致。",
            f"关键信息约束 19/23 通过；4 个失败中 3 个来自长重复文本，另 1 个删除时间 14:35。",
            f"CPU 稳态平均 {validation['avg_latency_ms']/1000:.2f}s/条，p95 {validation['latency_ms_percentiles']['p95']/1000:.2f}s，生成吞吐 {validation['decode_tokens_per_second']:.2f} tokens/s。",
            "243 prompt tokens 的重复型输入已出现强复读；475/823 tokens 时打满 256-token 输出上限并丢失末尾否定事实。",
        ],
        styles,
    )
    story.extend([Spacer(1, 3 * mm), chart(charts_dir / "01_quality_overview.png", max_height=95 * mm), PageBreak()])

    # Method and quality
    story.extend(
        [
            para("1. 评测范围与证据强度", styles["h1"]),
            para("测试覆盖归档、数据、模型行为、性能、训练历史和工程入口。以下证据按可信度分层：", styles["body"]),
            make_table(
                ["层级", "样本/测试", "用途", "限制"],
                [
                    ["验证复核", "500 条", "检查与打包报告是否一致、稳定统计", "参与过选点，非独立盲测"],
                    ["独立压力", "135 条", "发现域外、扰动、注入、长文本风险", "规模有限、单一人工参考"],
                    ["性能", "CPU 本机", "延迟、吞吐、内存、伸缩", "不外推到 GPU"],
                    ["静态审计", "全量数据/392 哈希/71 测试", "完整性、泄漏、工程问题", "不等价于人工代码审计"],
                ],
                [24 * mm, 28 * mm, 61 * mm, 61 * mm],
                styles,
            ),
            Spacer(1, 4 * mm),
            para("2. 质量报表", styles["h1"]),
            make_table(
                ["集合", "N", "ROUGE-L", "chrF", "SARI", "压缩比", "复制率", "EOS/Stop"],
                [
                    ["验证复核", 500, num(validation["rouge_l_f1"]), num(validation["chrf"]), num(validation["sari"]), num(validation["compression_ratio"]), pct(validation["unchanged_copy_rate"]), pct(validation["eos_or_stop_hit_rate"])],
                    ["复制基线", 500, num(copy["rouge_l_f1"]), num(copy["chrf"]), num(copy["sari"]), "1.000", "100.0%", "-"],
                    ["独立压力", 135, num(stress["rouge_l_f1"]), num(stress["chrf"]), num(stress["sari"]), num(stress["compression_ratio"]), pct(stress["unchanged_copy_rate"]), pct(stress["eos_or_stop_hit_rate"])],
                ],
                [31 * mm, 12 * mm, 22 * mm, 20 * mm, 20 * mm, 22 * mm, 22 * mm, 25 * mm],
                styles,
            ),
            Spacer(1, 4 * mm),
            para(
                f"验证 SARI 95% bootstrap CI 为 {_ci(validation, 'sari')}；独立压力集为 {_ci(stress, 'sari')}。验证目标本身平均压缩比 {validation['reference_compression_ratio']:.3f}，复制输入已能获得 ROUGE-L {copy['rouge_l_f1']:.3f}，所以必须结合 SARI 与复制基线解读。",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Length stratification
    length_groups = sorted(data["validation"]["stratified"]["source_length"].items())
    story.extend(
        [
            para("3. 长度分层与延迟长尾", styles["h1"]),
            chart(charts_dir / "02_length_degradation.png", max_height=102 * mm),
            Spacer(1, 3 * mm),
            make_table(
                ["输入字符", "N", "ROUGE-L", "SARI", "平均延迟", "重复率", "EOS/Stop"],
                [
                    [name.split("_", 1)[1], summary["count"], num(summary["rouge_l_f1"]), num(summary["sari"]), seconds_from_ms(summary["avg_latency_ms"]), num(summary["repetition_ratio"]), pct(summary["eos_or_stop_hit_rate"])]
                    for name, summary in length_groups
                ],
                [28 * mm, 14 * mm, 24 * mm, 22 * mm, 28 * mm, 25 * mm, 28 * mm],
                styles,
            ),
            Spacer(1, 4 * mm),
            callout(">256 字符组的 ROUGE-L 降至 0.819，平均延迟 6.74s，EOS/Stop 为 90%，重复率显著上升。配置窗口 896 tokens 不能等同于可靠能力边界。", styles, background=PALE_RED, border=RED),
            PageBreak(),
        ]
    )

    # Stress categories
    category_rows = []
    translations = {"formal": "正式", "spoken": "口语", "news": "新闻", "technical": "技术", "legal": "法律", "academic": "学术", "negation": "否定", "entity": "实体", "mixed": "混合", "noisy": "噪声", "traditional": "繁体", "numbers": "数字", "long_context": "长文本", "injection": "注入"}
    for name, summary in data["stress"]["by_category"].items():
        if name in {"identity", "perturbation"}:
            continue
        category_rows.append((translations.get(name, name), summary["count"], num(summary["sari"]), pct(summary["unchanged_copy_rate"]), num(summary["compression_ratio"]), pct(summary["eos_or_stop_hit_rate"])))
    category_rows.sort(key=lambda row: float(row[2]))
    story.extend(
        [
            para("4. 独立压力集与域外泛化", styles["h1"]),
            para(
                f"独立集 SARI {stress['sari']:.3f}，输出压缩比 {stress['compression_ratio']:.3f}，参考为 {stress['reference_compression_ratio']:.3f}。原样复制率 {pct(stress['unchanged_copy_rate'])}，主要弱项是法律、数字、新闻和繁体。",
                styles["body"],
            ),
            chart(charts_dir / "03_stress_categories.png", max_height=128 * mm),
            PageBreak(),
            para("4-1 分类型明细", styles["h1"]),
            make_table(
                ["类别", "N", "SARI", "复制率", "输出压缩比", "EOS/Stop"],
                category_rows,
                [36 * mm, 16 * mm, 25 * mm, 28 * mm, 34 * mm, 34 * mm],
                styles,
            ),
            Spacer(1, 5 * mm),
            para("扰动一致性", styles["h2"]),
            make_table(
                ["扰动", "输出 ROUGE-L 一致性"],
                [
                    ["逐字空格", num(data["stress"]["perturbation_consistency"]["by_perturbation"]["spaces"]["mean"])],
                    ["去标点", num(data["stress"]["perturbation_consistency"]["by_perturbation"]["punct_removed"]["mean"])],
                    ["插入换行", num(data["stress"]["perturbation_consistency"]["by_perturbation"]["linebreaks"]["mean"])],
                ],
                [80 * mm, 70 * mm],
                styles,
            ),
            Spacer(1, 5 * mm),
            callout("高扰动一致性不必然代表高质量：逐字空格输入经常被原样复制，因此要同时查看 SARI 和复制率。", styles, background=PALE_ORANGE, border=ORANGE),
            PageBreak(),
        ]
    )

    # Reliability and factuality
    story.extend(
        [
            para("5. 可靠性、事实性与注入", styles["h1"]),
            chart(charts_dir / "06_reliability_scorecard.png", max_height=102 * mm),
            Spacer(1, 3 * mm),
            make_table(
                ["检查", "结果", "解释"],
                [
                    ["简单句不乱改", "12/12", "独立 identity 样例完全保持"],
                    ["关键信息约束", "19/23", "1 个时间删除 + 3 个长文本末尾事实丢失"],
                    ["非长文本约束", "19/20", "95.0%"],
                    ["验证数字召回", pct(validation["number_recall"]), f"低于复制基线 {pct(copy['number_recall'])}"],
                    ["基础注入", "8/8 未执行 payload", "行为探针，不是安全认证"],
                    ["重复确定性", "24/24", "8 个输入各重复 3 次"],
                ],
                [45 * mm, 40 * mm, 89 * mm],
                styles,
            ),
            Spacer(1, 5 * mm),
            callout("关键事实失败示例：列车 G1234 原定 14:35、推迟至 15:10，模型输出删除了 14:35。验证样例还出现删除年份 1934、删除比例列表等情况。高风险事实文本必须做输入输出数字/日期/单位差异校验。", styles, background=PALE_RED, border=RED),
            PageBreak(),
        ]
    )

    # Long context
    long_rows = [row for row in data["stress"]["records"] if row["case_category"] == "long_context"]
    story.extend(
        [
            para("6. 长文本失效模式", styles["h1"]),
            para("重复型长文本用于放大模型在上下文边界附近的复读、截断和末尾事实遗失。结果显示实际可靠边界远低于配置窗口。", styles["body"]),
            make_table(
                ["case", "prompt tokens", "生成", "结束", "重复率", "末尾事实", "延迟"],
                [
                    [row["case_id"], row["prompt_tokens"], row["generated_tokens"], row["finish_reason"], num(row["repetition_ratio"]), "通过" if not row["constraints"]["must_keep_missing"] else "丢失", f"{row['latency_ms']/1000:.2f}s"]
                    for row in long_rows
                ],
                [24 * mm, 28 * mm, 21 * mm, 23 * mm, 24 * mm, 28 * mm, 25 * mm],
                styles,
            ),
            Spacer(1, 6 * mm),
        ]
    )
    story += bullet_items(
        [
            "127 prompt tokens：能够结束，但 4-gram 重复率已达 0.416。",
            "243 prompt tokens：虽产生 EOS，重复率 0.831，并丢失末尾“不会取消”事实。",
            "475/823 prompt tokens：均生成 256 tokens 后因 length 结束，重复率 0.890，末尾事实丢失。",
            "建议：约 200 tokens 以上强制分块；输出达到上限或重复率异常时拒绝自动交付并回退。",
        ],
        styles,
    )
    story.extend([Spacer(1, 4 * mm), callout("生产禁区：不要把 max_seq_len=896 当作可直接处理 896 tokens 的质量承诺。", styles, background=PALE_RED, border=RED), PageBreak()])

    # Performance
    story.extend(
        [
            para("7. CPU 性能报表", styles["h1"]),
            make_table(
                ["项目", "结果"],
                [
                    ["稳态平均 / p50 / p95 / p99", f"{validation['avg_latency_ms']/1000:.2f}s / {validation['latency_ms_percentiles']['p50']/1000:.2f}s / {validation['latency_ms_percentiles']['p95']/1000:.2f}s / {validation['latency_ms_percentiles']['p99']/1000:.2f}s"],
                    ["生成吞吐", f"{validation['decode_tokens_per_second']:.2f} tokens/s"],
                    ["进程内加载 / 冷启动 CLI", f"{runtime['load_seconds']:.2f}s / {preflight['smoke_inference']['cold_cli_wall_seconds']:.3f}s"],
                    ["加载后工作集 / 峰值", f"{gib(runtime['memory_after_load']['working_set'])} / {gib(micro['memory_after_microbench']['peak_working_set'])}"],
                    ["KV cache", f"{micro['kv_cache_mean_speedup']:.2f}x 加速，输出 6/6 一致"],
                    ["最佳线程数（本机前向）", f"4 threads, {max(micro['forward_by_thread_count'], key=lambda row: row['tokens_per_second'])['tokens_per_second']:.0f} tokens/s"],
                ],
                [65 * mm, 105 * mm],
                styles,
            ),
            Spacer(1, 4 * mm),
            chart(charts_dir / "05_performance_scaling.png", max_height=132 * mm),
            PageBreak(),
            para("7-1 延迟分布", styles["h1"]),
            chart(charts_dir / "04_latency_distribution.png", max_height=105 * mm),
            Spacer(1, 4 * mm),
            para(
                "平均值会掩盖长尾：验证复核 p95 为 4.43s、p99 为 6.32s，最大 11.57s；独立压力集含长文本，最大 14.76s。交互式产品应按 p95/p99 设计超时与回退。",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Training and data
    story.extend(
        [
            para("8. 数据泄漏、训练与模型结构", styles["h1"]),
            make_table(
                ["项目", "结果"],
                [
                    ["训练 / 验证", f"{audit['train_rows']:,} / {audit['validation_rows']:,}"],
                    ["严格源文本重合", str(audit["exact_source_overlap"])],
                    ["归一化源文本重合", str(audit["normalized_source_overlap"])],
                    ["近重复 (Jaccard >=0.85)", f"{audit['near_duplicate']['near_match_count']} ({pct(audit['near_duplicate']['near_match_rate'], 2)})"],
                    ["验证 identity 对", f"{audit['validation_identity_pairs']} ({pct(audit['validation_identity_pairs']/audit['validation_rows'])})"],
                    ["最佳 / 最终验证 loss", f"{training['best_val_loss']:.4f} / {training['final_metrics']['loss']:.4f}"],
                    ["有效参数 / state dict 元素", f"{runtime['model_parameters']:,} / {runtime['state_dict_elements']:,}"],
                ],
                [72 * mm, 98 * mm],
                styles,
            ),
            Spacer(1, 4 * mm),
            para("state dict 元素比有效参数多 28,672,000，来自权重绑定的 embedding/output alias 被重复计数，不代表实际有效参数增加。", styles["small"]),
            chart(charts_dir / "07_training_curve.png", max_height=105 * mm),
            Spacer(1, 3 * mm),
            callout(f"最佳验证 loss 出现在 step 350 附近；最终 loss 较最佳点回升 {(training['final_metrics']['loss']/training['best_val_loss']-1)*100:.1f}%。交付最佳权重而非最终轮是正确选择。", styles, background=PALE_ORANGE, border=ORANGE),
            PageBreak(),
        ]
    )

    # Engineering quality
    story.extend(
        [
            para("9. 工程与便携包审计", styles["h1"]),
            make_table(
                ["检查", "结果", "评价"],
                [
                    ["归档路径安全", "427 条目，0 危险路径，0 链接", "通过"],
                    ["文件完整性", "392 个 SHA-256 全通过", "通过"],
                    ["Smoke 推理", "Windows CPU 成功", "通过"],
                    ["源码测试", "70 通过 / 1 失败", "打包缺陷"],
                    ["run_simplify.ps1", "可用", "通过"],
                    ["run_eval_portable.sh", "引用缺失 best.pt 与基础权重", "不可直接复评"],
                    ["README 一致性", "拆包说明与便携 README 矛盾", "需修复"],
                ],
                [46 * mm, 78 * mm, 46 * mm],
                styles,
            ),
            Spacer(1, 6 * mm),
            para("唯一测试失败", styles["h2"]),
            para("`test_variants_are_parameter_matched` 需要 `pretrain_fineweb_v21_20b_540m_wide.yaml` 和 `..._deep.yaml`，但拆分包未包含。这不是 203M 简化模型逻辑失败，却说明交付测试集没有按拆包范围裁剪。", styles["body"]),
            para("文档/入口问题", styles["h2"]),
            para("`README_00_SPLIT_PACKAGE_ZH.md` 正确说明不含基础预训练权重；`README_PORTABLE_ZH.md` 却声称已经包含。便携评估配置还指向未交付的 `best.pt` 与基础模型，导致官方复评入口不能直接运行。", styles["body"]),
            Spacer(1, 5 * mm),
            callout("结论：推理交付可用，复评/再训练交付需要修复路径与文档；不要把 70/71 简化成“测试基本全过”而忽略可复现性问题。", styles, background=PALE_ORANGE, border=ORANGE),
            PageBreak(),
        ]
    )

    # Pros/cons and recommendations
    story.extend([para("10. 优缺点与部署建议", styles["h1"]), para("优点", styles["h2"])])
    story += bullet_items(
        [
            "同分布质量强，SARI 相对复制基线提升 +0.233，且大多数样本不退步。",
            "简单句保守性好；无空输出、无保留标签泄漏；贪心生成完全确定。",
            "短中等输入的数字、实体和否定大多保留，非长文本约束通过 95%。",
            "203M 规模可在普通 CPU 运行，内存约 1.6-1.9 GiB；KV cache 加速有效。",
            "最佳权重选择合理，完整性哈希、Tokenizer 和 Smoke 推理均通过。",
        ],
        styles,
    )
    story.extend([para("缺点", styles["h2"])])
    story += bullet_items(
        [
            "域外欠简化明显，法律/数字/新闻等类别常原样复制。",
            "存在删除年份、时间和比例列表的事实性错误，不适合无守门的高风险文本。",
            "长重复文本复读、丢失末尾否定事实并触及长度上限，长上下文能力不可靠。",
            "验证集参与选点且有少量近重复，已有高分不是无偏泛化估计。",
            "FP32 权重 812 MiB，未提供量化、ONNX 或硬件加速部署包。",
            "复评脚本、测试依赖和 README 存在一致性缺陷。",
        ],
        styles,
    )
    story.extend([para("建议的上线门槛", styles["h2"])])
    story += bullet_items(
        [
            "输入约 200 tokens 以上强制分块；对末尾窗口事实单独校验。",
            "比较输入/输出的数字、日期、时间、单位、百分比、实体和否定词；差异超阈值时拒绝自动交付。",
            "监控 EOS、输出长度上限和字符 4-gram 重复率；异常则回退为保守复制或人工复核。",
            "保持 KV cache；本机优先 4 线程；批处理仅用于吞吐优先的离线任务。",
            "建立至少 1,000 条冻结独立盲测集并加入人工事实性/可读性评分，再决定高风险场景放行。",
            "修复 portable eval 路径、裁剪不相关测试、统一拆包 README，再发布新包。",
        ],
        styles,
    )
    story.extend([Spacer(1, 5 * mm), callout("最终判定：条件通过。短中等、低风险中文简化可进入受控试用；长文本与高事实风险场景暂不放行。", styles, background=PALE_ORANGE, border=ORANGE), PageBreak()])

    # Limitations and artifact index
    story.extend(
        [
            para("11. 指标口径、限制与产物", styles["h1"]),
            para("指标口径", styles["h2"]),
            para("ROUGE-L/chrF 衡量与单一参考答案的表面相似度；SARI 为中文字符 n-gram 的增删保留指标。它们都不能证明语义完全等价。数字召回只覆盖正则识别到的阿拉伯数字；中文数字与实体依赖额外约束探针。", styles["body"]),
            para("主要限制", styles["h2"]),
        ]
    )
    story += bullet_items(
        [
            "验证集参与过 checkpoint 选择；本报告明确降级为复核证据。",
            "独立压力集 135 条规模有限，且每条只有一个人工参考。",
            "基础注入样例不能替代专门的红队与系统级安全评估。",
            "CPU 性能只代表本机；历史 GPU 训练硬件名称在 run manifest 中为空。",
            "未做人类双盲可读性评分，也未调用外部大模型裁判，避免把主观自动评分冒充金标准。",
        ],
        styles,
    )
    story.extend(
        [
            para("产物索引", styles["h2"]),
            make_table(
                ["产物", "位置"],
                [
                    ["PDF 报告", "output/pdf/murmur_203m_text_simplification_extreme_evaluation_20260814.pdf"],
                    ["Markdown 报告", "output/extreme_evaluation_20260814/report.md"],
                    ["原始 JSON", "output/extreme_evaluation_20260814/raw/extreme_eval_results.json"],
                    ["指标图片", "output/extreme_evaluation_20260814/charts/"],
                    ["CSV 报表", "output/extreme_evaluation_20260814/tables/"],
                    ["可复现代码", "extreme_eval/ 与 run_extreme_eval.py"],
                ],
                [45 * mm, 125 * mm],
                styles,
                font_size=7.0,
            ),
            Spacer(1, 6 * mm),
            para("报告生成：2026-08-14 | 所有结论均可追溯到本地原始 JSON、训练日志、配置和逐条输出。", styles["small"]),
        ]
    )

    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)


def main() -> None:
    args = parse_args()
    result_path = Path(args.result).resolve()
    preflight_path = Path(args.preflight).resolve()
    output_dir = Path(args.output_dir).resolve()
    pdf_path = Path(args.pdf).resolve()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    build_markdown(data, preflight, output_dir)
    build_pdf(data, preflight, output_dir, pdf_path)
    print(json.dumps({"markdown": str(output_dir / 'report.md'), "pdf": str(pdf_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
