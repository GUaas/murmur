from __future__ import annotations

from .catalog import HfDataset


OASST2 = HfDataset(
    key="oasst2",
    repo_id="OpenAssistant/oasst2",
    revision="179dd21fc55192153d94adb0e0ce8f69e222bf75",
    files=("2023-11-05_oasst2_ready.trees.jsonl.gz",),
)

DRCD = HfDataset(
    key="drcd",
    repo_id="ihainan/DRCD-Simplified-Chinese",
    revision="de11764c42349f940e89b0dbfcff16b26a45056f",
    files=("DRCD_train_simplified_chinese.json",),
)

DOLLY_CURATED = HfDataset(
    key="dolly_curated",
    repo_id="argilla/databricks-dolly-15k-curated-en",
    revision="4dcd1dedbe148307a833c931b21ca456a1fc4281",
    files=("data/train-00000-of-00001-15a05aeec7726f9d.parquet",),
)

TULU_INSTRUCTION_FOLLOWING = HfDataset(
    key="tulu_if",
    repo_id="allenai/tulu-3-sft-personas-instruction-following",
    revision="fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
    files=("data/train-00000-of-00001.parquet",),
)

V3_HF_DATASETS = (DRCD, DOLLY_CURATED, TULU_INSTRUCTION_FOLLOWING)

DOLLY_CATEGORY_QUOTAS = {
    "information_extraction": 1_000,
    "summarization": 1_000,
    "closed_qa": 550,
}

V3_QUOTAS = {
    "drcd": 15_000,
    "tulu_instruction_following": 5_000,
    **{f"dolly_{category}": quota for category, quota in DOLLY_CATEGORY_QUOTAS.items()},
}
