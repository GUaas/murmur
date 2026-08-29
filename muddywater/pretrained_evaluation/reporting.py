from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def render_markdown(results: dict[str, Any]) -> str:
    artifact = results["artifact"]
    checks = results["functional_checks"]
    coverage = results["coverage"]
    quality = results.get("text_quality", {}).get("aggregate", {})
    benchmark_names = ", ".join(results.get("benchmarks", {})) or "未运行"
    lines = [
        "# Murmur 预训练模型评估报告",
        "",
        f"- 生成时间：{results['run']['completed_at']}",
        f"- 设备：{results['run']['device']}",
        f"- 权重 SHA-256：`{artifact['checkpoint']['sha256']}`",
        f"- 唯一参数量：{artifact['model']['unique_parameters']:,}",
        f"- 严格加载：{'通过' if artifact['model']['strict_load_passed'] else '失败'}",
        f"- 权重有限值检查：{'通过' if artifact['state']['all_finite'] else '失败'}",
        f"- Tokenizer 哈希：{'匹配' if artifact['tokenizer']['sha256_matches_checkpoint'] else '未验证'}",
        "",
        "## 功能正确性",
        "",
        f"- 输出有限值：{'通过' if checks['forward_logits_all_finite'] else '失败'}",
        f"- 确定性：{'通过' if checks['determinism']['passed'] else '失败'}",
        f"- 因果隔离：{'通过' if checks['causal_isolation']['passed'] else '失败'}",
        f"- KV 缓存一致性：{'通过' if checks['kv_cache_parity']['passed'] else '失败'}",
        f"- 越界上下文拒绝：{'通过' if checks['context_contract']['over_limit_rejected'] else '失败'}",
        "",
        "## 质量与覆盖",
        "",
        f"- 内置短文本探针 PPL：{quality.get('perplexity', '未运行')}",
        f"- 标准基准：{benchmark_names}",
        f"- 原始同源验证集：{'已复测' if coverage['heldout_validation_retested'] else '缺少验证缓存，未复测'}",
        f"- 2048 全上下文实算：{'已运行' if coverage['full_context_executed'] else '未运行'}",
        "",
        "> 权重内记录的 best_val_loss 仅作为训练元数据展示，不等同于本机独立复测。",
        "",
    ]
    return "\n".join(lines)


def write_reports(output_dir: str | Path, results: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = json_safe(results)
    json_path = output_dir / "evaluation_results.json"
    markdown_path = output_dir / "evaluation_summary.md"
    json_path.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(safe), encoding="utf-8")
    return json_path, markdown_path


def write_partial_report(output_dir: str | Path, results: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation_results.partial.json"
    path.write_text(
        json.dumps(json_safe(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
