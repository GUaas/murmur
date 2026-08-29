from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    key: str
    label: str
    sources: frozenset[str] = frozenset()
    category_terms: tuple[str, ...] = ()

    def matches(self, source: str, category: str) -> bool:
        return source in self.sources or any(term in category for term in self.category_terms)


SCENARIOS = (
    ScenarioSpec(
        "conversation",
        "日常、多轮与任务型对话",
        frozenset({"smoltalk_chinese", "duconv", "kdconv", "crosswoz", "oasst2_zh"}),
        ("everyday", "task_oriented", "knowledge_dialogue", "human_feedback_dialogue"),
    ),
    ScenarioSpec(
        "knowledge_qa",
        "中文知识问答与解释",
        frozenset({"coig_cqia"}),
        ("knowledge", "information-seeking", "问答", "概念解析", "百科"),
    ),
    ScenarioSpec(
        "reading_comprehension",
        "材料阅读、理解与抽取式问答",
        frozenset({"cmrc2018", "drcd", "dolly_closed_qa"}),
        ("reading_comprehension", "document-qa", "understanding", "english_closed_qa"),
    ),
    ScenarioSpec(
        "writing_revision",
        "写作、改写、纠错与润色",
        frozenset({"doit_creative_writing", "dolly_creative_writing", "dolly_brainstorming"}),
        ("writing", "editing", "rewrite", "brainstorming", "写", "纠错", "扩写", "对联"),
    ),
    ScenarioSpec(
        "summarization_translation",
        "摘要、翻译与简繁转换",
        category_terms=("summary", "summarization", "摘要", "translate", "翻译", "简繁", "繁体"),
    ),
    ScenarioSpec(
        "math",
        "算术、应用题与数学推理",
        frozenset({"gsm8k_zh", "synthetic_math_verified"}),
        ("arithmetic", "add_subtract", "multiply_divide", "percentage", "数学"),
    ),
    ScenarioSpec(
        "programming",
        "编程、算法、SQL 与代码解释",
        frozenset({"coig_leetcode", "doit_code"}),
        ("code", "代码", "SQL"),
    ),
    ScenarioSpec(
        "reasoning",
        "逻辑推理、分析与判断",
        frozenset({"doit_reasoning"}),
        ("reasoning", "推理", "因果分析", "判断"),
    ),
    ScenarioSpec(
        "nlp_structured_output",
        "分类、抽取与结构化输出",
        frozenset(
            {
                "synthetic_instruction_verified",
                "dolly_classification",
                "dolly_information_extraction",
            }
        ),
        ("classification", "information_extraction", "分类", "抽取", "json", "fixed_list", "key_value", "table"),
    ),
    ScenarioSpec(
        "strict_instruction_following",
        "约束遵循与可验证格式",
        frozenset({"synthetic_instruction_verified", "tulu_instruction_following"}),
        ("strict_instruction_following",),
    ),
    ScenarioSpec(
        "planning_advice",
        "规划、建议与问题解决",
        category_terms=("planning", "advice-seeking", "wikihow"),
    ),
    ScenarioSpec(
        "safety_values",
        "安全边界、隐私与价值对齐",
        frozenset({"coig_human_value", "synthetic_safety_verified"}),
        ("safe", "privacy", "safety", "credentials", "phishing", "fraud", "human_value"),
    ),
    ScenarioSpec(
        "identity",
        "murmur 自我认知与开发者归属",
        frozenset({"synthetic_identity_verified"}),
        ("identity_", "capability_boundary", "experience_boundary"),
    ),
    ScenarioSpec(
        "creative_roleplay",
        "创意写作与角色扮演",
        frozenset({"doit_creative_writing", "doit_role_play"}),
        ("creative_writing", "role_play", "故事", "诗词"),
    ),
    ScenarioSpec(
        "factual_correction",
        "多轮事实澄清与纠错",
        frozenset({"coig_counterfactual"}),
        ("factual_correction",),
    ),
    ScenarioSpec(
        "humanities_chinese",
        "中文、人文与历史",
        frozenset({"doit_chinese", "doit_history"}),
        ("古诗", "文言", "作者介绍", "历史"),
    ),
)


def build_scenario_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    matched_records = 0
    total_records = 0

    for row in rows:
        total_records += 1
        source = str(row.get("source", "unknown"))
        category = str(row.get("category", "unknown"))
        source_counts[source] += 1
        category_counts[category] += 1
        matched = False
        for scenario in SCENARIOS:
            if scenario.matches(source, category):
                scenario_counts[scenario.key] += 1
                matched = True
        matched_records += int(matched)

    return {
        "total_records": total_records,
        "records_matching_at_least_one_scenario": matched_records,
        "coverage_ratio": round(matched_records / total_records, 6) if total_records else 0.0,
        "scenario_counts_are_overlapping": True,
        "scenarios": [
            {
                "key": scenario.key,
                "label": scenario.label,
                "records": scenario_counts[scenario.key],
            }
            for scenario in SCENARIOS
        ],
        "source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
    }
