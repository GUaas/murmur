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
from reportlab.lib.utils import ImageReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "long_text_chunking_evaluation_20260814"
DEFAULT_RESULT = DEFAULT_OUTPUT / "raw" / "long_text_chunking_eval_results.json"
DEFAULT_PDF = PROJECT_ROOT / "output" / "pdf" / "murmur_203m_long_text_chunking_extreme_evaluation_20260814.pdf"

NAVY = colors.HexColor("#19324D")
TEAL = colors.HexColor("#1E8A8A")
BLUE = colors.HexColor("#3B82C4")
GREEN = colors.HexColor("#4E9A68")
ORANGE = colors.HexColor("#E58A2B")
RED = colors.HexColor("#C34A45")
GRAY = colors.HexColor("#6F7D89")
LIGHT = colors.HexColor("#E8EEF3")
PALE_BLUE = colors.HexColor("#EEF6FC")
PALE_GREEN = colors.HexColor("#EDF8F1")
PALE_ORANGE = colors.HexColor("#FFF5E8")
PALE_RED = colors.HexColor("#FCEEEE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Markdown and PDF long-text evaluation reports.")
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    return parser.parse_args()


def pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def sec(value_ms: float | None, digits: int = 2) -> str:
    return "-" if value_ms is None else f"{value_ms / 1000.0:.{digits}f}s"


def delta(new: float, old: float, digits: int = 3) -> str:
    return f"{new - old:+.{digits}f}"


def build_markdown(data: dict[str, Any], output_dir: Path) -> str:
    direct = data["direct"]["summary"]
    chunked = data["chunked"]["summary"]
    comparison = data["comparison"]
    audit = data["segmentation_audit"]
    tests = data["test_suite"]
    lines = [
        "# Murmur 203M 长文本分块算法极限评估报告",
        "",
        "**总体评级：B / 条件通过。** 新算法已经把长文本从容易截断、漏尾的状态提升到可受控使用，但它用更高端到端延迟换取完整性，且数字与跨块一致性仍需守门。",
        "",
        "## 1. 结论摘要",
        "",
        "| 指标 | 旧：整篇直接推理 | 新：分句分块 | 变化 |",
        "|---|---:|---:|---:|",
        f"| ROUGE-L | {direct['rouge_l_f1']:.3f} | {chunked['rouge_l_f1']:.3f} | {delta(chunked['rouge_l_f1'], direct['rouge_l_f1'])} |",
        f"| chrF | {direct['chrf']:.3f} | {chunked['chrf']:.3f} | {delta(chunked['chrf'], direct['chrf'])} |",
        f"| SARI | {direct['sari']:.3f} | {chunked['sari']:.3f} | {delta(chunked['sari'], direct['sari'])} |",
        f"| 数字召回 | {pct(direct['number_recall'])} | {pct(chunked['number_recall'])} | {pct(chunked['number_recall']-direct['number_recall'])} |",
        f"| 尾部事实保留 | {pct(direct['tail_document_pass_rate'])} | {pct(chunked['tail_document_pass_rate'])} | {pct(chunked['tail_document_pass_rate']-direct['tail_document_pass_rate'])} |",
        f"| 全部正常结束 | {pct(direct['all_chunks_finished_rate'])} | {pct(chunked['all_chunks_finished_rate'])} | {pct(chunked['all_chunks_finished_rate']-direct['all_chunks_finished_rate'])} |",
        f"| 16 篇总延迟 | {direct['total_latency_seconds']:.1f}s | {chunked['total_latency_seconds']:.1f}s | {chunked['total_latency_seconds']/direct['total_latency_seconds']:.2f}x |",
        "",
        "![质量 A/B](charts/01_quality_ab.png)",
        "",
        "![可靠性 A/B](charts/02_reliability_ab.png)",
        "",
        "## 2. 测试范围与口径",
        "",
        "- 16 篇独立长文档，提示长度 127 到 1,186 词元。",
        "- 4 篇固定重复长文回归 + 12 篇由独立人工压力句确定性组合的跨领域文档。",
        "- 未读取或重跑打包训练集、验证集；验证集没有参与本报告指标。",
        "- 旧路径和新路径逐文档交替先后执行，降低热机和顺序偏差。",
        "- 另测 96、160、224 三档预算，12 项人工边界，2,000 份随机文档，10 万字符规划和 3 篇各 3 次确定性。",
        "",
        "## 3. 算法与边界正确性",
        "",
        "算法流程：保留式分句 -> 按提示词元预算装块 -> 超长单句软切分/字符兜底 -> 复用同一模型顺序推理 -> 按原顺序和段落分隔符合并。预审中发现 `e.g.` 被误分句，已补充常见英文缩写保护；无标点超长句的前缀搜索也已从近似二次扫描优化为有界扩展搜索。",
        "",
        f"人工边界通过 {pct(audit['handcrafted_unit_count_pass_rate'])}；2,000 份随机文档分割重建、规划重建和词元上限均为 100%。10 万字符有标点/无标点规划分别约 {data['planning_benchmarks'][2]['elapsed_ms']:.1f}ms / {data['planning_benchmarks'][5]['elapsed_ms']:.1f}ms。",
        "",
        "![分割与规划](charts/06_segmentation_planning.png)",
        "",
        "## 4. 质量与长度",
        "",
        f"新算法在 ROUGE-L、chrF、SARI 上分别提升 {delta(chunked['rouge_l_f1'], direct['rouge_l_f1'])}、{delta(chunked['chrf'], direct['chrf'])}、{delta(chunked['sari'], direct['sari'])}。SARI 13/16 提升、2/16 持平、1/16 轻微下降。短长文几乎没有收益，真正收益集中在 200 词元以上。",
        "",
        "![按长度质量](charts/04_quality_by_length.png)",
        "",
        "![逐文档变化](charts/07_per_document_delta.png)",
        "",
        "## 5. 事实完整性与失败案例",
        "",
        f"新路径尾部事实 16/16 保留，全部块 100% 正常结束，数字召回为 {pct(chunked['number_recall'])}。严格的整篇约束全通过率为 {pct(chunked['constraint_document_pass_rate'])}，但逐项召回为 {pct(chunked['constraint_item_recall'])}：5 篇文档至少漏 1 项，其中 4 篇重复出现漏掉原发车时间 `14:35`；`composite_10` 还漏了 8 个数字/时间项。",
        "",
        "`long_02` 的旧输出没有末尾“不取消原计划”，新输出恢复该事实。算法解决了窗口截断，但没有解决模型本身删除数字的倾向。",
        "",
        "## 6. 性能报表",
        "",
        f"16 篇总延迟从 {direct['total_latency_seconds']:.1f}s 增加到 {chunked['total_latency_seconds']:.1f}s，约 {chunked['total_latency_seconds']/direct['total_latency_seconds']:.2f}x；只有 {pct(comparison['chunked_faster_rate'])} 文档更快。新路径平均 {chunked['chunk_count_percentiles']['mean']:.2f} 块，p95 {chunked['chunk_count_percentiles']['p95']:.1f}，最大 48 块。",
        "",
        "![延迟与长度](charts/03_latency_vs_length.png)",
        "",
        "![块数成本](charts/08_chunk_count_cost.png)",
        "",
        "## 7. 分块预算",
        "",
        "160 词元是本机和本测试集上的最佳综合默认值：96 词元增加块数和延迟；224 词元虽然块数略少，但 SARI 降低且代表样本尾部通过率降至 75%。",
        "",
        "![预算扫描](charts/05_budget_sweep.png)",
        "",
        "## 8. 优点",
        "",
        "1. 显著降低长文截断、漏尾和无法正常结束。",
        "2. 长度越大，质量收益越明显；极长文 SARI 从约 0.271 提升到 0.564。",
        "3. 保留段落与换行结构，2,000 份随机文档规划无损。",
        "4. 规划开销很低，10 万字符低于 0.33 秒。",
        "5. 复用单次模型加载，贪心解码 3 篇各 3 次完全一致。",
        "6. 对重复型长文可同时提质和提速。",
        "",
        "## 9. 缺点",
        "",
        "1. 不是通用提速：完整输出通常更长，总延迟约翻倍。",
        "2. 逐行文本会产生大量小块，48 行样本产生 48 次推理，延迟约为旧路径 5.1 倍。",
        "3. 不能消除模型本身的数字删除问题，`14:35` 仍反复丢失。",
        f"4. 仍偏保守，输出压缩比 {chunked['compression_ratio']:.3f}，参考为 {chunked['reference_compression_ratio']:.3f}。",
        f"5. 不能做跨块全局去重，重复率 {chunked['repetition_ratio']:.3f}，与旧路径 {direct['repetition_ratio']:.3f} 基本相同。",
        "6. 跨块代词、承接关系和全局摘要不受保证。",
        "",
        "## 10. 上线建议",
        "",
        "- 保持 `mode: auto` 和 `max_prompt_tokens_per_chunk: 160`。160 以下继续走直接路径。",
        "- 数字、日期、时间、单位、实体和否定词必须做输入输出差异守门；缺失时回退原文或人工复核。",
        "- 对逐行列表设置最大块数或先安全合并相邻短行，否则延迟可能膨胀。",
        "- 设置整篇超时、最大块数和最大总输出词元；面向交互场景应流式返回每块结果。",
        "- 法律、医疗、金融等高风险文本仍不能无人审核自动发布。",
        "- 下一步性能优化应做长度分桶批处理或安全并行，而不是继续放大单块窗口。",
        "",
        "## 11. 测试状态与局限",
        "",
        f"代码测试 {tests['passed']} 通过、{tests['failed']} 失败；唯一失败仍是便携包缺少无关的 540M 预训练对比配置。新增长文本测试全部通过。",
        "",
        "本报告只有 16 篇长文档，其中 12 篇是由独立短压力句组合得到；单参考自动指标不等于人工语义审校。100% 尾部保留和正常结束仅代表本测试集，不是无限文本保证。CPU 数据只适用于本机。",
        "",
        "## 12. 产物索引",
        "",
        "- 原始结果：`raw/long_text_chunking_eval_results.json`",
        "- 指标图片：`charts/`",
        "- CSV 报表：`tables/`",
        "- 可复现代码：`long_text_eval/` 与 `run_long_text_eval.py`",
        "",
    ]
    markdown = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    return markdown


def register_font() -> str:
    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    if not font_path.exists():
        raise FileNotFoundError(f"Chinese font not found: {font_path}")
    pdfmetrics.registerFont(TTFont("ReportChinese", str(font_path)))
    return "ReportChinese"


def build_styles(font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle("CoverTitle", parent=base["Title"], fontName=font, fontSize=24, leading=33, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12),
        "cover_subtitle": ParagraphStyle("CoverSubtitle", parent=base["Normal"], fontName=font, fontSize=11.5, leading=18, textColor=GRAY),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName=font, fontSize=17, leading=23, textColor=NAVY, spaceBefore=2, spaceAfter=9),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName=font, fontSize=12.5, leading=17, textColor=TEAL, spaceBefore=7, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName=font, fontSize=9.3, leading=14.5, textColor=NAVY, spaceAfter=6),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName=font, fontSize=7.7, leading=11, textColor=GRAY, spaceAfter=3),
        "table": ParagraphStyle("Table", parent=base["BodyText"], fontName=font, fontSize=7.2, leading=9.5, textColor=NAVY),
        "table_header": ParagraphStyle("TableHeader", parent=base["BodyText"], fontName=font, fontSize=7.2, leading=9.5, textColor=colors.white, alignment=TA_CENTER),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName=font, fontSize=10, leading=15.5, textColor=NAVY, leftIndent=4, rightIndent=4, spaceAfter=2),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName=font, fontSize=9, leading=13.5, textColor=NAVY, leftIndent=13, firstLineIndent=-8, spaceAfter=3),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def callout(text: str, styles: dict[str, ParagraphStyle], *, background=PALE_BLUE, border=BLUE) -> Table:
    table = Table([[para(text, styles["callout"])]], colWidths=[174 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 1, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def bullet_items(items: Iterable[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [Paragraph(f"- {html.escape(item)}", styles["bullet"]) for item in items]


def make_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    styles: dict[str, ParagraphStyle],
    *,
    font_size: float = 7.2,
) -> Table:
    table_style = styles["table"]
    if font_size != table_style.fontSize:
        table_style = ParagraphStyle("TableDynamic", parent=table_style, fontSize=font_size, leading=font_size + 2.3)
    payload = [[para(html.escape(str(value)), styles["table_header"]) for value in headers]]
    payload.extend([[para(html.escape(str(value)), table_style) for value in row] for row in rows])
    table = Table(payload, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD4DD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def chart(path: Path, *, max_width: float = 174 * mm, max_height: float = 103 * mm) -> Image:
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
    canvas.drawString(20 * mm, height - 11 * mm, "Murmur 203M 长文本分块算法极限评估")
    canvas.drawRightString(width - 20 * mm, 10 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def build_pdf(data: dict[str, Any], output_dir: Path, pdf_path: Path) -> None:
    font = register_font()
    styles = build_styles(font)
    charts = output_dir / "charts"
    direct = data["direct"]["summary"]
    chunked = data["chunked"]["summary"]
    comparison = data["comparison"]
    audit = data["segmentation_audit"]
    tests = data["test_suite"]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(pdf_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=16 * mm,
        title="Murmur 203M 长文本分块算法极限评估报告",
        author="Codex",
        subject="长文本分句分块算法的质量、可靠性、性能与工程评估",
    )
    story: list[Any] = []

    # Cover
    story.extend([
        Spacer(1, 22 * mm),
        para("Murmur 203M 中文文本简化系统", styles["cover_subtitle"]),
        para("长文本分句分块算法<br/>极限评估报告", styles["cover_title"]),
        Spacer(1, 5 * mm),
        callout("<b>总体评级：B / 条件通过</b><br/>长文质量、尾部事实和结束可靠性显著提升；端到端延迟约翻倍，数字事实和逐行块数仍需生产守门。", styles, background=PALE_ORANGE, border=ORANGE),
        Spacer(1, 10 * mm),
        make_table(
            ["长文质量", "事实可靠性", "CPU 延迟", "算法工程"],
            [["B+", "A-（仍需数字守门）", "C", "A-"]],
            [43.5 * mm] * 4, styles, font_size=8.5,
        ),
        Spacer(1, 14 * mm),
        para("评估日期：2026-08-14", styles["cover_subtitle"]),
        para("模型：203,037,056 参数 | CPU：Intel i5-13420H | PyTorch 线程：4", styles["cover_subtitle"]),
        para("证据：16 篇长文 A/B + 三档预算 + 12 项人工边界 + 2,000 份随机文档 + 10 万字符规划 + 确定性复测", styles["cover_subtitle"]),
        Spacer(1, 14 * mm),
        para("重要口径：本报告未读取或重跑打包训练集、验证集。所有质量结论来自独立长文本压力文档。", styles["small"]),
        PageBreak(),
    ])

    # Executive summary
    story.extend([
        para("1. 执行摘要", styles["h1"]),
        callout(
            f"新算法将 SARI 从 <b>{direct['sari']:.3f}</b> 提升到 <b>{chunked['sari']:.3f}</b>，尾部事实从 <b>{pct(direct['tail_document_pass_rate'])}</b> 提升到 <b>{pct(chunked['tail_document_pass_rate'])}</b>，但 16 篇总延迟从 <b>{direct['total_latency_seconds']:.1f}s</b> 增加到 <b>{chunked['total_latency_seconds']:.1f}s</b>。它是可靠性优化，不是通用提速器。",
            styles, background=PALE_GREEN, border=GREEN,
        ),
        Spacer(1, 4 * mm),
        make_table(
            ["指标", "旧：直接推理", "新：分块推理", "变化"],
            [
                ["ROUGE-L", f"{direct['rouge_l_f1']:.3f}", f"{chunked['rouge_l_f1']:.3f}", delta(chunked['rouge_l_f1'], direct['rouge_l_f1'])],
                ["chrF", f"{direct['chrf']:.3f}", f"{chunked['chrf']:.3f}", delta(chunked['chrf'], direct['chrf'])],
                ["SARI", f"{direct['sari']:.3f}", f"{chunked['sari']:.3f}", delta(chunked['sari'], direct['sari'])],
                ["数字召回", pct(direct['number_recall']), pct(chunked['number_recall']), f"+{(chunked['number_recall']-direct['number_recall'])*100:.1f}pp"],
                ["尾部事实", pct(direct['tail_document_pass_rate']), pct(chunked['tail_document_pass_rate']), f"+{(chunked['tail_document_pass_rate']-direct['tail_document_pass_rate'])*100:.1f}pp"],
                ["正常结束", pct(direct['all_chunks_finished_rate']), pct(chunked['all_chunks_finished_rate']), f"+{(chunked['all_chunks_finished_rate']-direct['all_chunks_finished_rate'])*100:.1f}pp"],
                ["总延迟", f"{direct['total_latency_seconds']:.1f}s", f"{chunked['total_latency_seconds']:.1f}s", f"{chunked['total_latency_seconds']/direct['total_latency_seconds']:.2f}x"],
            ],
            [44 * mm, 43 * mm, 43 * mm, 35 * mm], styles,
        ),
        Spacer(1, 5 * mm),
        *bullet_items([
            "建议保留自动模式：160 词元以内不分块，超过后启用。",
            "长文可受控上线，但数字、日期、时间、单位、实体和否定词仍必须校验。",
            "逐行列表应设置最大块数或安全合并，否则延迟可能增加 5 倍以上。",
            "高风险行业仍需人工复核，不能把分块等同于事实安全。",
        ], styles),
        PageBreak(),
    ])

    # Methodology
    story.extend([
        para("2. 测试设计与算法", styles["h1"]),
        para("本轮专门比较加入长文本算法前后的整套系统，没有重复验证集评估。16 篇文档覆盖 127 到 1,186 个提示词元，包括固定重复长文、数字、否定、法律、技术、新闻和跨领域多段落。旧/新路径按文档交替先运行，减少热机顺序偏差。", styles["body"]),
        para("2.1 算法流程", styles["h2"]),
        make_table(
            ["1 分句", "2 装块", "3 兜底", "4 推理", "5 合并"],
            [["保留标点和空白", "提示不超过 160", "软标点或字符切分", "单次加载顺序执行", "恢复顺序和段落"]],
            [34.8 * mm] * 5, styles, font_size=7.5,
        ),
        Spacer(1, 5 * mm),
        para("分句保护小数、版本号、网址和闭合引号。预审发现常见英文缩写 e.g. 会误分句，已加入缩写保护；超长无标点文本的前缀查找也已改为有界扩展搜索，避免近似二次扫描。", styles["body"]),
        para("2.2 证据矩阵", styles["h2"]),
        make_table(
            ["测试", "规模", "目的"],
            [
                ["长文 A/B", "16 篇，127-1,186 词元", "质量、事实、延迟、长度退化"],
                ["预算扫描", "96 / 160 / 224，4 篇代表文档", "寻找质量和性能平衡点"],
                ["边界审计", "12 项人工 + 2,000 随机", "无损、标点保护、预算上限"],
                ["规划微基准", "1千 / 1万 / 10万字符", "有标点和无标点复杂度"],
                ["确定性", "3 篇各 3 次", "贪心输出与合并稳定性"],
                ["代码测试", f"{tests['passed']} 通过 / {tests['failed']} 失败", "回归和工程完整性"],
            ],
            [48 * mm, 55 * mm, 71 * mm], styles,
        ),
        Spacer(1, 5 * mm),
        callout("指标说明：ROUGE-L/chrF 衡量参考表面相似度；SARI 衡量保留、删除和新增。单参考自动指标不能替代人工语义审查。", styles),
        PageBreak(),
    ])

    # Quality
    story.extend([
        para("3. 长文质量 A/B", styles["h1"]),
        chart(charts / "01_quality_ab.png", max_height=94 * mm),
        Spacer(1, 3 * mm),
        para(f"三项参考质量均显著上升。配对结果中，ROUGE-L 和 SARI 都是 13/16 提升、2/16 持平、1/16 下降；chrF 为 14/16 提升、2/16 持平、无下降。平均 SARI 增量为 {comparison['reference_metrics']['sari']['mean_delta']:+.3f}。", styles["body"]),
        callout(f"注意压缩率：新输出为 {chunked['compression_ratio']:.3f}，参考为 {chunked['reference_compression_ratio']:.3f}，仍明显欠简化。旧路径的 {direct['compression_ratio']:.3f} 看似接近参考，部分来自截断，不代表更好的简化。", styles, background=PALE_ORANGE, border=ORANGE),
        Spacer(1, 4 * mm),
        make_table(
            ["配对指标", "平均增量", "胜 / 平 / 负"],
            [
                ["ROUGE-L", f"{comparison['reference_metrics']['rouge_l_f1']['mean_delta']:+.3f}", "13 / 2 / 1"],
                ["chrF", f"{comparison['reference_metrics']['chrf']['mean_delta']:+.3f}", "14 / 2 / 0"],
                ["SARI", f"{comparison['reference_metrics']['sari']['mean_delta']:+.3f}", "13 / 2 / 1"],
            ],
            [58 * mm, 58 * mm, 58 * mm], styles,
        ),
        PageBreak(),
    ])

    # Reliability
    failures = [row for row in data["chunked"]["records"] if not row["constraints"]["pass"]]
    failure_rows = [[row["document_id"], "、".join(row["constraints"]["missing"]), f"{row['constraints']['kept']}/{row['constraints']['total']}", pct(row["number_recall"])] for row in failures]
    story.extend([
        para("4. 事实完整性与结束可靠性", styles["h1"]),
        chart(charts / "02_reliability_ab.png", max_height=91 * mm),
        Spacer(1, 3 * mm),
        para("新路径 16/16 保留指定尾部事实，16/16 的所有块都以 EOS/停止串正常结束，且换行数量精确保留。没有空输出回退，也没有触发长度上限。", styles["body"]),
        para(f"但严格的整篇约束全通过只有 {pct(chunked['constraint_document_pass_rate'])}。原因是全通过口径很严：逐项实际保留 {pct(chunked['constraint_item_recall'])}，只要漏 1 项整篇就失败。", styles["body"]),
        make_table(["文档", "漏失项", "保留项", "数字召回"], failure_rows, [30 * mm, 91 * mm, 24 * mm, 29 * mm], styles, font_size=6.8),
        Spacer(1, 4 * mm),
        callout("最稳定的残留故障是原发车时间 14:35：在 4 篇组合文档中再次被删除。分块消除了窗口截断，但没有改变模型本身偏向只保留新时间 15:10 的行为。", styles, background=PALE_RED, border=RED),
        PageBreak(),
    ])

    # Length
    tier_rows = []
    tier_names = {"short_long": "短长文", "medium_long": "中长文", "long": "长文", "extreme_long": "极长文"}
    for tier in ("short_long", "medium_long", "long", "extreme_long"):
        old = data["direct"]["by_length_tier"][tier]
        new = data["chunked"]["by_length_tier"][tier]
        tier_rows.append([tier_names[tier], old["count"], f"{old['sari']:.3f}", f"{new['sari']:.3f}", pct(old["tail_document_pass_rate"]), pct(new["tail_document_pass_rate"])])
    story.extend([
        para("5. 长度退化是否被修复", styles["h1"]),
        chart(charts / "04_quality_by_length.png", max_height=91 * mm),
        Spacer(1, 3 * mm),
        make_table(["长度层", "篇数", "旧 SARI", "新 SARI", "旧尾部", "新尾部"], tier_rows, [35 * mm, 20 * mm, 30 * mm, 30 * mm, 29 * mm, 30 * mm], styles),
        Spacer(1, 5 * mm),
        *bullet_items([
            "短长文：SARI 轻微下降，说明不应对 160 词元以内输入强制分块。",
            "中长文：质量明显提高且总延迟略降，是当前算法最理想区间。",
            "长文和极长文：尾部保留恢复到 100%，但延迟约 2-3 倍。",
            "极长文 SARI 从 0.271 提升到 0.564，证明窗口截断是旧路径的主要瓶颈。",
        ], styles),
        PageBreak(),
    ])

    # Performance
    story.extend([
        para("6. 性能报表", styles["h1"]),
        chart(charts / "03_latency_vs_length.png", max_height=86 * mm),
        Spacer(1, 3 * mm),
        para(f"16 篇旧路径总延迟 {direct['total_latency_seconds']:.1f}s，新路径 {chunked['total_latency_seconds']:.1f}s，为旧路径的 {chunked['total_latency_seconds']/direct['total_latency_seconds']:.2f} 倍。平均单篇从 {sec(direct['avg_latency_ms'])} 增加到 {sec(chunked['avg_latency_ms'])}；新路径仅 {pct(comparison['chunked_faster_rate'])} 文档更快。", styles["body"]),
        make_table(
            ["性能项", "旧路径", "新路径"],
            [
                ["平均单篇延迟", sec(direct["avg_latency_ms"]), sec(chunked["avg_latency_ms"])],
                ["p95 单篇延迟", sec(direct["latency_ms_percentiles"]["p95"]), sec(chunked["latency_ms_percentiles"]["p95"])],
                ["解码吞吐", f"{direct['corpus_generated_tokens_per_second']:.2f} tok/s", f"{chunked['corpus_generated_tokens_per_second']:.2f} tok/s"],
                ["平均 / p95 / 最大块数", "1 / 1 / 1", f"{chunked['chunk_count_percentiles']['mean']:.2f} / {chunked['chunk_count_percentiles']['p95']:.1f} / {chunked['chunk_count_percentiles']['max']:.0f}"],
                ["工作集峰值", "共享同一进程", f"{data['final_process_memory']['peak_working_set']/(1024**3):.2f} GiB"],
                ["模型加载", "共享", f"{data['runtime']['load_seconds']:.2f}s"],
            ],
            [68 * mm, 53 * mm, 53 * mm], styles,
        ),
        Spacer(1, 4 * mm),
        callout("解码吞吐没有下降，慢的根本原因是新路径生成了更完整、更多的总输出，并需要多次块级前向。旧路径对极长文很快，往往是因为截断后只生成了局部内容。", styles),
        PageBreak(),
    ])

    # Per-document cost
    selected_ids = {"long_02", "long_03", "long_04", "composite_07", "composite_09", "composite_11", "composite_12"}
    paired = {row["document_id"]: row for row in comparison["details"]}
    per_rows = []
    for record in data["chunked"]["records"]:
        if record["document_id"] not in selected_ids:
            continue
        row = paired[record["document_id"]]
        per_rows.append([record["document_id"], record["source_prompt_tokens"], record["chunk_count"], f"{row['sari_delta']:+.3f}", f"{row['latency_speedup']:.2f}x", "是" if record["tail_constraints"]["pass"] else "否"])
    story.extend([
        para("7. 逐文档收益与成本", styles["h1"]),
        chart(charts / "07_per_document_delta.png", max_height=98 * mm),
        Spacer(1, 3 * mm),
        make_table(["文档", "整篇词元", "块数", "SARI 增量", "旧/新速度比", "尾部保留"], per_rows, [36 * mm, 29 * mm, 22 * mm, 29 * mm, 32 * mm, 26 * mm], styles),
        Spacer(1, 4 * mm),
        para("重复型 long_02/03 同时提质和提速；long_04 仍提质，但 6 块带来约 21% 额外延迟。composite_12 的旧路径只处理局部内容，因此看似 4.95 秒；新路径完整处理 60 句需要 43.82 秒。", styles["body"]),
        PageBreak(),
    ])

    # Chunk count cost
    story.extend([
        para("8. 块数膨胀风险", styles["h1"]),
        chart(charts / "08_chunk_count_cost.png", max_height=96 * mm),
        Spacer(1, 4 * mm),
        para("当前实现把换行作为强边界，以保证版式精确恢复。这个选择对段落安全，但逐行列表会退化为每行一次推理：composite_07 产生 24 块，composite_11 产生 48 块。后者延迟约为旧路径 5.1 倍。", styles["body"]),
        callout("生产建议：增加最大块数守门；对逐行短句提供可选的“单换行可合并、双换行强分段”模式，或做长度分桶批推理。不能直接并发写同一个结果流而不保序。", styles, background=PALE_ORANGE, border=ORANGE),
        Spacer(1, 6 * mm),
        para("当前顺序执行的优点是确定、内存平稳、结果天然有序；缺点是多块延迟线性累积。下一阶段性能优化应围绕批处理和安全并行，而不是把窗口上限从 160 盲目放大。", styles["body"]),
        PageBreak(),
    ])

    # Budget
    budget_rows = []
    for budget in (96, 160, 224):
        summary = data["budget_sweep"]["by_budget"][str(budget)]
        budget_rows.append([budget, f"{summary['rouge_l_f1']:.3f}", f"{summary['sari']:.3f}", pct(summary['tail_document_pass_rate']), f"{summary['total_latency_seconds']:.1f}s", f"{summary['chunk_count_percentiles']['mean']:.2f}"])
    story.extend([
        para("9. 分块预算：为什么选择 160", styles["h1"]),
        chart(charts / "05_budget_sweep.png", max_height=91 * mm),
        Spacer(1, 3 * mm),
        make_table(["预算", "ROUGE-L", "SARI", "尾部通过", "4 篇总延迟", "平均块数"], budget_rows, [25 * mm, 29 * mm, 29 * mm, 31 * mm, 35 * mm, 25 * mm], styles),
        Spacer(1, 5 * mm),
        *bullet_items([
            "96：SARI 略高，但块数最多、ROUGE-L 最低、总延迟最高。",
            "160：ROUGE-L 最高、尾部 100%、总延迟最低，是最佳综合点。",
            "224：块数略少，但 SARI 明显下降，代表文档尾部通过率只有 75%。",
            "预算结论只代表当前 CPU、模型和测试集；换模型后需重新扫描。",
        ], styles),
        PageBreak(),
    ])

    # Segmentation
    planning_rows = [["有标点" if row["shape"] == "punctuated" else "无标点", f"{row['characters']:,}", row["chunks"], f"{row['elapsed_ms']:.2f}ms", f"{row['characters_per_second']:,.0f}", "通过" if row["budget_pass"] else "失败"] for row in data["planning_benchmarks"]]
    story.extend([
        para("10. 分句与规划工程质量", styles["h1"]),
        chart(charts / "06_segmentation_planning.png", max_height=86 * mm),
        Spacer(1, 3 * mm),
        para(f"12/12 人工边界通过；2,000/2,000 随机文档分割重建、规划重建和预算上限全部通过，处理速度约 {audit['fuzz_documents_per_second']:,.0f} 文档/秒。", styles["body"]),
        make_table(["形态", "字符", "块数", "耗时", "字符/秒", "预算"], planning_rows, [31 * mm, 29 * mm, 24 * mm, 32 * mm, 35 * mm, 23 * mm], styles, font_size=6.9),
        Spacer(1, 4 * mm),
        callout(f"确定性：3 篇代表长文各运行 3 次，逐字一致率 {pct(data['determinism']['exact_determinism_rate'])}。空输出回退 0 次，长度上限结束 0 块。", styles, background=PALE_GREEN, border=GREEN),
        PageBreak(),
    ])

    # Pros/cons and deployment
    story.extend([
        para("11. 优缺点与上线建议", styles["h1"]),
        para("11.1 优点", styles["h2"]),
        *bullet_items([
            "长文 SARI 提升 0.158，尾部事实、正常结束和换行版式达到 100%。",
            "数字召回从 40.8% 提升到 96.1%，极长文不再只输出窗口末端或中途截断。",
            "分割和规划无损，10 万字符规划低于 0.33 秒。",
            "同一模型只加载一次，内存没有随文档或块数持续增长。",
            "默认 160 词元有实测预算扫描支撑。",
        ], styles),
        para("11.2 缺点", styles["h2"]),
        *bullet_items([
            "总延迟约翻倍；逐行极长文可能慢 5-9 倍。",
            "仍会漏数字和时间，14:35 是稳定复现的失败案例。",
            "输出压缩比 0.791，参考为 0.590，仍明显欠简化。",
            "跨块全局去重、代词承接和摘要一致性没有解决。",
            "16 篇样本规模有限，不能替代大规模人工盲测。",
        ], styles),
        para("11.3 生产配置", styles["h2"]),
        make_table(
            ["控制项", "建议"],
            [
                ["触发", "mode=auto；提示 >160 才分块"],
                ["事实守门", "数字/日期/时间/单位/实体/否定词差异；失败回退原文"],
                ["性能守门", "整篇超时、最大块数、最大总输出；逐块流式返回"],
                ["逐行文本", "最大块数或安全合并单换行；优先长度分桶批处理"],
                ["高风险内容", "法律、医疗、金融必须人工复核"],
                ["发布策略", "低风险灰度上线，记录块数、漏事实、长度结束与延迟"],
            ],
            [45 * mm, 129 * mm], styles,
        ),
        PageBreak(),
    ])

    # Limitations and artifact index
    story.extend([
        para("12. 结论、限制与产物", styles["h1"]),
        callout("<b>最终结论：可以作为默认长文本保护层上线，但必须受控。</b><br/>它已经解决本模型最严重的长文截断和漏尾问题；它没有把模型变成全文摘要器，也没有消除数字事实风险。", styles, background=PALE_GREEN, border=GREEN),
        Spacer(1, 5 * mm),
        para("12.1 局限", styles["h2"]),
        *bullet_items([
            "12 篇组合长文来自独立压力句的确定性拼接，真实业务文档结构可能更复杂。",
            "ROUGE-L、chrF、SARI 基于单一参考，不能证明语义完全正确。",
            "尾部 100% 和正常结束 100% 仅适用于这 16 篇文档。",
            "CPU 性能仅适用于 Intel i5-13420H、4 个 PyTorch 线程和当前软件版本。",
            "本轮没有验证并行推理、GPU、量化、批处理或多用户并发。",
        ], styles),
        para("12.2 代码状态", styles["h2"]),
        para(f"测试结果：{tests['passed']} 通过、{tests['failed']} 失败。唯一失败是便携包仍保留 540M 预训练变体对比测试，却缺少对应配置文件；与长文本算法无关。新增 8 项长文本测试全部通过。", styles["body"]),
        para("12.3 产物索引", styles["h2"]),
        make_table(
            ["产物", "位置"],
            [
                ["原始 JSON", "output/long_text_chunking_evaluation_20260814/raw/long_text_chunking_eval_results.json"],
                ["Markdown", "output/long_text_chunking_evaluation_20260814/report.md"],
                ["8 张指标图", "output/long_text_chunking_evaluation_20260814/charts/"],
                ["7 份 CSV", "output/long_text_chunking_evaluation_20260814/tables/"],
                ["评估代码", "long_text_eval/ 与 run_long_text_eval.py"],
                ["算法代码", "muddywater/text_simplification/chunking.py 与 inference.py"],
            ],
            [43 * mm, 131 * mm], styles, font_size=6.9,
        ),
        Spacer(1, 7 * mm),
        para("报告生成时间：2026-08-14 | 原始证据保留完整逐文档输入、参考、输出、分块信息、结束原因和耗时。", styles["small"]),
    ])

    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    pdf_path = Path(args.pdf).resolve()
    build_markdown(data, output_dir)
    build_pdf(data, output_dir, pdf_path)
    print(json.dumps({"markdown": str(output_dir / 'report.md'), "pdf": str(pdf_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
