from __future__ import annotations

import random
import time
from statistics import mean
from typing import Callable

from muddywater.text_simplification.chunking import (
    plan_inference_chunks,
    reconstruct_source,
    split_sentences,
)


TokenCounter = Callable[[str], int]


def _handcrafted_cases() -> list[dict[str, object]]:
    return [
        {"id": "decimal", "text": "温度是3.14摄氏度。下一句。", "expected_units": 2, "protected": ("3.14",)},
        {"id": "version", "text": "版本v1.2.3已经发布。请升级。", "expected_units": 2, "protected": ("v1.2.3",)},
        {"id": "url", "text": "访问https://a.cn/v1.2查看。然后登录。", "expected_units": 2, "protected": ("https://a.cn/v1.2",)},
        {"id": "quoted", "text": "他说：“可以！”随后离开。", "expected_units": 2, "protected": ("可以！",)},
        {"id": "ellipsis", "text": "我们还在考虑……暂时不决定。", "expected_units": 2, "protected": ()},
        {"id": "crlf", "text": "第一行没有句号\r\n第二行。\r\n", "expected_units": 2, "protected": ()},
        {"id": "paragraphs", "text": "第一段。\n\n第二段。", "expected_units": 2, "protected": ()},
        {"id": "semicolon", "text": "先检查设备；再启动服务；最后确认。", "expected_units": 3, "protected": ()},
        {"id": "ascii", "text": "It is version 2.0. It works.", "expected_units": 2, "protected": ("2.0",)},
        {"id": "abbreviation", "text": "例如 e.g. example should stay together.", "expected_units": 1, "protected": ("e.g.",)},
        {"id": "emoji", "text": "结果不错🙂！继续测试。", "expected_units": 2, "protected": ("🙂",)},
        {"id": "no_punctuation", "text": "这是一个完全没有标点但是非常长的句子", "expected_units": 1, "protected": ()},
    ]


def _fuzz_documents(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    atoms = [
        "温度为3.14℃",
        "网址https://a.cn/v1.2可访问",
        "列车G1234在14:35出发",
        "他说：“不会取消！”",
        "版本v2.0.1已经上线",
        "CPU使用率为92%",
        "这是普通中文句子",
        "混合English与中文",
    ]
    endings = ["。", "！", "？", "；", "... ", "\n", "\r\n", "\n\n"]
    documents: list[str] = []
    for _ in range(count):
        pieces = [rng.choice(atoms) + rng.choice(endings) for _ in range(rng.randint(1, 12))]
        prefix = rng.choice(["", " ", "\n", "\n\n"])
        documents.append(prefix + "".join(pieces))
    return documents


def run_segmentation_audit(
    token_count: TokenCounter,
    *,
    max_prompt_tokens: int = 160,
    fuzz_count: int = 2000,
) -> dict[str, object]:
    handcrafted_rows = []
    for case in _handcrafted_cases():
        text = str(case["text"])
        leading, units = split_sentences(text)
        unit_texts = [unit.text for unit in units]
        protected = tuple(case["protected"])
        row = {
            "id": case["id"],
            "expected_units": case["expected_units"],
            "actual_units": len(units),
            "unit_count_pass": len(units) == int(case["expected_units"]),
            "lossless": reconstruct_source(leading, units) == text,
            "protected_pass": all(any(item in unit for unit in unit_texts) for item in protected),
            "units": unit_texts,
        }
        handcrafted_rows.append(row)

    fuzz_rows = []
    started = time.perf_counter()
    for index, text in enumerate(_fuzz_documents(fuzz_count, seed=20260814), start=1):
        leading, units = split_sentences(text)
        plan = plan_inference_chunks(text, token_count, max_prompt_tokens=max_prompt_tokens)
        rebuilt_plan = plan.leading_whitespace + "".join(
            chunk.source + chunk.separator_after for chunk in plan.chunks
        )
        fuzz_rows.append(
            {
                "index": index,
                "lossless_split": reconstruct_source(leading, units) == text,
                "lossless_plan": rebuilt_plan == text,
                "budget_pass": all(chunk.prompt_tokens <= max_prompt_tokens for chunk in plan.chunks),
                "chunks": len(plan.chunks),
            }
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "max_prompt_tokens": max_prompt_tokens,
        "handcrafted": handcrafted_rows,
        "handcrafted_count": len(handcrafted_rows),
        "handcrafted_unit_count_pass_rate": round(mean(row["unit_count_pass"] for row in handcrafted_rows), 6),
        "handcrafted_lossless_rate": round(mean(row["lossless"] for row in handcrafted_rows), 6),
        "handcrafted_protected_pass_rate": round(mean(row["protected_pass"] for row in handcrafted_rows), 6),
        "fuzz_count": fuzz_count,
        "fuzz_elapsed_ms": round(elapsed_ms, 6),
        "fuzz_documents_per_second": round(fuzz_count / (elapsed_ms / 1000.0), 6),
        "fuzz_lossless_split_rate": round(mean(row["lossless_split"] for row in fuzz_rows), 6),
        "fuzz_lossless_plan_rate": round(mean(row["lossless_plan"] for row in fuzz_rows), 6),
        "fuzz_budget_pass_rate": round(mean(row["budget_pass"] for row in fuzz_rows), 6),
        "fuzz_chunk_percentiles": _percentiles([int(row["chunks"]) for row in fuzz_rows]),
        "failures": [row for row in handcrafted_rows if not all((row["unit_count_pass"], row["lossless"], row["protected_pass"]))],
    }


def _percentiles(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def value_at(q: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * q))
        return float(ordered[index])

    return {
        "mean": round(mean(ordered), 6),
        "p50": value_at(0.50),
        "p95": value_at(0.95),
        "max": float(ordered[-1]),
    }


def run_planning_benchmarks(
    token_count: TokenCounter,
    *,
    max_prompt_tokens: int = 160,
) -> list[dict[str, object]]:
    punctuated_clause = "项目组完成检查后，决定先修复故障，再安排升级。"
    unpunctuated_clause = "这是用于测试超长无标点文本规划性能的连续内容"
    rows: list[dict[str, object]] = []
    for shape, clause in (("punctuated", punctuated_clause), ("no_punctuation", unpunctuated_clause)):
        for target_chars in (1_000, 10_000, 100_000):
            text = (clause * (target_chars // len(clause) + 1))[:target_chars]
            started = time.perf_counter()
            plan = plan_inference_chunks(text, token_count, max_prompt_tokens=max_prompt_tokens)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                {
                    "shape": shape,
                    "characters": len(text),
                    "chunks": len(plan.chunks),
                    "elapsed_ms": round(elapsed_ms, 6),
                    "characters_per_second": round(len(text) / (elapsed_ms / 1000.0), 6),
                    "max_chunk_prompt_tokens": max(
                        (chunk.prompt_tokens for chunk in plan.chunks), default=0
                    ),
                    "budget_pass": all(
                        chunk.prompt_tokens <= max_prompt_tokens for chunk in plan.chunks
                    ),
                }
            )
    return rows
