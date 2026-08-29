from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HfDataset:
    key: str
    repo_id: str
    revision: str
    files: tuple[str, ...]


SMOLTALK_QUOTAS = {
    "advice-seeking": 4_000,
    "document-qa": 7_000,
    "editing": 5_000,
    "everyday": 6_000,
    "information-seeking": 10_000,
    "planning": 3_000,
    "rewrite": 4_000,
    "safe": 3_000,
    "summary": 4_000,
    "translate": 2_000,
}

SMOLTALK = HfDataset(
    key="smoltalk_chinese",
    repo_id="opencsg/smoltalk-chinese",
    revision="5edf5d0d7b06794755cf78b274a150a4efe1c9d4",
    files=tuple(f"{category}.parquet" for category in SMOLTALK_QUOTAS),
)

COIG = HfDataset(
    key="coig_cqia",
    repo_id="m-a-p/COIG-CQIA",
    revision="8b55868c6168adf86c30e7ca0f782cca1c514297",
    files=("COIG-CQIA-full.jsonl",),
)

OASST2 = HfDataset(
    key="oasst2",
    repo_id="OpenAssistant/oasst2",
    revision="179dd21fc55192153d94adb0e0ce8f69e222bf75",
    files=("data/train-00000-of-00001-88ba0162028a73fc.parquet",),
)

KDCONV = HfDataset(
    key="kdconv",
    repo_id="thu-coai/kdconv",
    revision="460c94a39c1498241b3c7e94a22be25e1489601e",
    files=("film/train.json", "music/train.json", "travel/train.json"),
)

CROSSWOZ = HfDataset(
    key="crosswoz",
    repo_id="GEM/CrossWOZ",
    revision="0c6f57946a15c70c44b28b81ae5fad9558abae01",
    files=("data.zip",),
)

ULTRADATA = HfDataset(
    key="ultradata",
    repo_id="openbmb/UltraData-SFT-2605",
    revision="affda6aca75e7cff78e73f93ad08d4c3b01f097c",
    files=(
        "data/no_think/Chinese-general/Chinese-general_no_think_part-01-of-50.jsonl",
        "data/no_think/Chinese-general/Chinese-general_no_think_part-02-of-50.jsonl",
        "data/no_think/IF/IF_no_think_part-01-of-20.jsonl",
        "data/no_think/IF/IF_no_think_part-02-of-20.jsonl",
        "data/no_think/Knowledge/Knowledge_no_think_part-01-of-80.jsonl",
    ),
)

HF_DATASETS = (SMOLTALK, COIG, OASST2, KDCONV, CROSSWOZ)
DUCONV_URL = "https://bj.bcebos.com/paddlenlp/datasets/DuConv.zip"

SOURCE_QUOTAS = {
    "coig_cqia": 30_000,
    "oasst2_zh": 0,
    "kdconv": 8_000,
    "crosswoz": 5_000,
    "duconv": 10_000,
    "ultradata_chinese_general": 15_000,
    "ultradata_if": 6_000,
    "ultradata_knowledge": 4_000,
    "synthetic_verified": 2_000,
}

# COIG-CQIA is useful but its source distribution is highly skewed. These
# caps stop encyclopedic, medical, legal, and social-media subsets from
# consuming too much of a 203M model's limited capacity.
COIG_CATEGORY_CAPS = {
    "医药问答": 0,
    "法律考研": 0,
    "概念解析": 5_000,
    "知乎问答": 0,
    "故事概要": 1_200,
    "中学考试": 1_000,
    "小红书风格文本": 0,
    "wikihow": 600,
}
