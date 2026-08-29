from __future__ import annotations

from .catalog import HfDataset


GSM8K_ZH = HfDataset(
    key="gsm8k_zh",
    repo_id="meta-math/GSM8K_zh",
    revision="4a5009abc37cbb2d3fd1a745f80e5ea1405ba9aa",
    files=("GSM8K_zh.json",),
)

COIG_ALIGNMENT = HfDataset(
    key="coig_alignment",
    repo_id="BAAI/COIG",
    revision="9f25758ec94f82762fb9c09a5c60e908cfb83632",
    files=(
        "counterfactural_correction_multi_round_chat.tar.gz",
        "human_value_alignment_instructions_part1.json",
        "leetcode_instructions.jsonl",
    ),
)

DOIT = HfDataset(
    key="doit",
    repo_id="ChiyuSONG/dynamics-of-instruction-tuning",
    revision="4ae3b55e7fd7966aa59afb7b819558f682e4ef3c",
    files=(
        "curated/full/chinese_full.json",
        "curated/full/code_full.json",
        "curated/full/creative_writing_full.json",
        "curated/full/history_full.json",
        "curated/full/reasoning_full.json",
        "curated/full/role_play_full.json",
        "curated/full/understanding_full.json",
    ),
)

CMRC2018 = HfDataset(
    key="cmrc2018",
    repo_id="hfl/cmrc2018",
    revision="137f2c45a24275fb68f6961c4d357f46288886aa",
    files=("data/train-00000-of-00001.parquet",),
)

ADDITIONAL_HF_DATASETS = (GSM8K_ZH, COIG_ALIGNMENT, DOIT, CMRC2018)

AUGMENTATION_QUOTAS = {
    "gsm8k_zh": 7_400,
    "coig_human_value": 3_000,
    "coig_leetcode": 5_000,
    "coig_counterfactual": 6_000,
    "cmrc2018": 10_000,
    "doit_chinese": 1_400,
    "doit_code": 3_000,
    "doit_creative_writing": 1_000,
    "doit_history": 1_800,
    "doit_reasoning": 6_000,
    "doit_role_play": 1_000,
    "doit_understanding": 4_000,
    "synthetic_math_verified": 8_000,
    "synthetic_safety_verified": 2_000,
    "synthetic_instruction_verified": 2_000,
    "synthetic_identity_verified": 2_500,
}
