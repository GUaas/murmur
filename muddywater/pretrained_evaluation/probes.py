from __future__ import annotations

from muddywater.assessment.types import GenerationProbe, TextProbe


TEXT_PROBES = [
    TextProbe(
        "zh_expository",
        "chinese_expository",
        "语言模型通过上下文预测下一个词元。可靠的评估应区分训练记录、独立复测与主观观察。",
        "zh-Hans",
    ),
    TextProbe(
        "zh_news",
        "chinese_news",
        "气象部门发布暴雨预警后，交通部门加强了重点路段巡查，并提醒市民关注最新信息。",
        "zh-Hans",
    ),
    TextProbe(
        "en_technical",
        "technical_english",
        "A decoder-only transformer predicts each token from its preceding context under a causal attention mask.",
        "en",
    ),
    TextProbe(
        "python",
        "code",
        "def moving_average(values, window):\n    return [sum(values[i:i+window]) / window for i in range(len(values)-window+1)]\n",
        "python",
    ),
    TextProbe(
        "math",
        "mathematics",
        "For real numbers a and b, the inequality a^2 + b^2 >= 2ab follows from (a-b)^2 >= 0.",
        "math",
    ),
    TextProbe(
        "unicode",
        "unicode_robustness",
        "实验完成✅，温度为23.5℃；样本包括繁體字、かな、한글与🙂。",
        "mixed",
    ),
]


GENERATION_PROBES = [
    GenerationProbe(
        "zh_continuation",
        "raw_continuation",
        "雨停以后，江南小镇的石板路上",
    ),
    GenerationProbe(
        "zh_knowledge",
        "factual_continuation",
        "中国的首都是",
        accepted_substrings=("北京",),
    ),
    GenerationProbe(
        "en_continuation",
        "raw_continuation",
        "The experiment was repeated three times. The results showed that",
    ),
    GenerationProbe(
        "python_completion",
        "code",
        "def add(a, b):\n    return",
        check_python_syntax=True,
    ),
]
