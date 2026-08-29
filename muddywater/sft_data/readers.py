from __future__ import annotations

import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq

from .catalog import SMOLTALK_QUOTAS, ULTRADATA
from .records import SFTRecord, normalize_messages, normalize_text, stable_hex


DUCONV_TOKEN_SPACE = re.compile(
    r"(?:(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fffA-Za-z0-9]))"
    r"|(?:(?<=[A-Za-z0-9]) +(?=[\u3400-\u9fff]))"
    r"| +(?=[，。！？、；：）》】」』,.!?;:])"
    r"|(?<=[，。！？、；：,.!?;:]) +"
    r"|(?<=[（《【「『]) +"
)


def _detokenize_duconv(text: Any) -> str:
    value = normalize_text(text)
    value = re.sub(r"\s*·\s*", "·", value)
    previous = None
    while previous != value:
        previous = value
        value = DUCONV_TOKEN_SPACE.sub("", value)
    return value


def _window_dialogue(
    messages: list[dict[str, str]],
    *,
    max_messages: int = 12,
    stride: int = 8,
) -> Iterator[tuple[int, list[dict[str, str]]]]:
    if len(messages) <= max_messages:
        if messages and messages[-1]["role"] == "assistant":
            yield 0, messages
        return
    for start in range(0, len(messages), stride):
        chunk = messages[start : start + max_messages]
        while chunk and chunk[0]["role"] != "user":
            chunk = chunk[1:]
        while chunk and chunk[-1]["role"] != "assistant":
            chunk = chunk[:-1]
        if len(chunk) >= 2:
            yield start, chunk
        if start + max_messages >= len(messages):
            break


def read_smoltalk(raw_dir: Path, category: str) -> Iterator[SFTRecord]:
    path = raw_dir / "smoltalk_chinese" / f"{category}.parquet"
    parquet = pq.ParquetFile(path)
    requested_columns = ["conversations", "score", "difficulty", "classify", "magpie_model"]
    available_columns = set(parquet.schema_arrow.names)
    columns = [column for column in requested_columns if column in available_columns]
    row_index = 0
    for batch in parquet.iter_batches(batch_size=1024, columns=columns):
        for row in batch.to_pylist():
            row_index += 1
            if "score" in available_columns and int(row.get("score") or 0) < 4:
                continue
            messages = normalize_messages(row.get("conversations") or [])
            if not messages:
                continue
            yield SFTRecord(
                messages=messages,
                source="smoltalk_chinese",
                category=category,
                group_id=f"smoltalk:{category}:{row_index}",
                source_id=f"{category}:{row_index}",
                metadata={
                    "score": row.get("score"),
                    "difficulty": row.get("difficulty"),
                    "classify": row.get("classify"),
                    "generator": row.get("magpie_model"),
                },
            )


def read_coig(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "coig_cqia" / "COIG-CQIA-full.jsonl"
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = normalize_text(row.get("instruction"))
            input_text = normalize_text(row.get("input"))
            output = normalize_text(row.get("output"))
            if input_text:
                instruction = f"{instruction}\n\n{input_text}" if instruction else input_text
            task = row.get("task_type") or {}
            minor = task.get("minor") if isinstance(task, dict) else None
            category = str(minor[0]) if isinstance(minor, list) and minor else "general"
            yield SFTRecord(
                messages=[
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                source="coig_cqia",
                category=category,
                group_id=f"coig:{line_index}",
                source_id=str(line_index),
                metadata={
                    "answer_from": row.get("answer_from"),
                    "human_verified": row.get("human_verified"),
                },
            )


def read_oasst2(raw_dir: Path, max_paths_per_tree: int = 2) -> Iterator[SFTRecord]:
    path = raw_dir / "oasst2" / "data" / "train-00000-of-00001-88ba0162028a73fc.parquet"
    columns = [
        "message_id", "parent_id", "text", "role", "lang", "review_result",
        "deleted", "rank", "review_count", "message_tree_id",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    nodes: dict[str, dict[str, Any]] = {}
    trees: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("lang") != "zh" or row.get("deleted") or row.get("review_result") is False:
            continue
        nodes[row["message_id"]] = row
        trees[row["message_tree_id"]].append(row)

    for tree_id, tree_rows in trees.items():
        candidates: list[tuple[int, int, list[dict[str, str]], str]] = []
        for row in tree_rows:
            if row.get("role") != "assistant":
                continue
            path_rows: list[dict[str, Any]] = []
            current: dict[str, Any] | None = row
            valid = True
            while current is not None:
                path_rows.append(current)
                parent_id = current.get("parent_id")
                if not parent_id:
                    break
                current = nodes.get(parent_id)
                if current is None:
                    valid = False
                    break
            if not valid:
                continue
            path_rows.reverse()
            messages = normalize_messages(
                [{"role": item.get("role"), "content": item.get("text")} for item in path_rows]
            )
            rank = int(row.get("rank")) if row.get("rank") is not None else 1_000_000
            reviews = int(row.get("review_count") or 0)
            candidates.append((-len(messages), rank - reviews, messages, row["message_id"]))
        ordered = sorted(candidates, key=lambda item: (item[0], item[1], item[3]))
        for _, _, messages, message_id in ordered[:max_paths_per_tree]:
            yield SFTRecord(
                messages=messages,
                source="oasst2_zh",
                category="assistant_dialogue",
                group_id=f"oasst2:{tree_id}",
                source_id=message_id,
            )


def read_kdconv(raw_dir: Path) -> Iterator[SFTRecord]:
    for domain in ("film", "music", "travel"):
        rows = json.loads((raw_dir / "kdconv" / domain / "train.json").read_text(encoding="utf-8"))
        for dialogue_index, row in enumerate(rows):
            messages = normalize_messages(
                [
                    {"role": "user" if index % 2 == 0 else "assistant", "content": item.get("message")}
                    for index, item in enumerate(row.get("messages") or [])
                ]
            )
            group_id = f"kdconv:{domain}:{dialogue_index}"
            for start, window in _window_dialogue(messages):
                yield SFTRecord(
                    messages=window,
                    source="kdconv",
                    category=domain,
                    group_id=group_id,
                    source_id=f"{domain}:{dialogue_index}:{start}",
                )


def read_crosswoz(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "crosswoz" / "data.zip"
    with zipfile.ZipFile(path) as archive:
        rows = json.loads(archive.read("train.json").decode("utf-8"))
    for dialogue_id, row in rows.items():
        messages = normalize_messages(row.get("messages") or [])
        group_id = f"crosswoz:{dialogue_id}"
        for start, window in _window_dialogue(messages):
            yield SFTRecord(
                messages=window,
                source="crosswoz",
                category="task_oriented",
                group_id=group_id,
                source_id=f"{dialogue_id}:{start}",
            )


def _locate_duconv_train(archive: zipfile.ZipFile) -> str:
    candidates = [name for name in archive.namelist() if name.replace("\\", "/").endswith("/train.txt")]
    if not candidates:
        raise FileNotFoundError("DuConv.zip does not contain train.txt")
    return candidates[0]


def read_duconv(raw_dir: Path) -> Iterator[SFTRecord]:
    path = raw_dir / "duconv" / "DuConv.zip"
    with zipfile.ZipFile(path) as archive:
        train_name = _locate_duconv_train(archive)
        with archive.open(train_name) as binary:
            for line_index, raw_line in enumerate(binary, start=1):
                row = json.loads(raw_line.decode("utf-8"))
                turns = row.get("conversation") or row.get("history") or []
                messages = normalize_messages(
                    [
                        {
                            "role": "user" if index % 2 == 0 else "assistant",
                            "content": _detokenize_duconv(content),
                        }
                        for index, content in enumerate(turns)
                    ]
                )
                response = normalize_text(row.get("response"))
                if response:
                    messages.append({"role": "assistant", "content": response})
                group_id = f"duconv:{line_index}"
                for start, window in _window_dialogue(messages):
                    yield SFTRecord(
                        messages=window,
                        source="duconv",
                        category="knowledge_dialogue",
                        group_id=group_id,
                        source_id=f"{line_index}:{start}",
                    )


def _messages_from_flexible_row(row: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("messages", "conversations"):
        value = row.get(key)
        if isinstance(value, list):
            return normalize_messages(value)
    instruction = normalize_text(row.get("instruction", row.get("prompt", row.get("query", ""))))
    output = normalize_text(row.get("output", row.get("response", row.get("answer", ""))))
    return normalize_messages(
        [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]
    )


def read_ultradata(raw_dir: Path, domain: str) -> Iterator[SFTRecord]:
    prefix = {
        "chinese_general": "data/no_think/Chinese-general/",
        "if": "data/no_think/IF/",
        "knowledge": "data/no_think/Knowledge/",
    }[domain]
    source_dir = raw_dir / ULTRADATA.key
    paths = sorted(path for path in source_dir.rglob("*.jsonl") if prefix in path.as_posix())
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = _messages_from_flexible_row(row)
                source_id = f"{path.name}:{line_index}"
                yield SFTRecord(
                    messages=messages,
                    source=f"ultradata_{domain}",
                    category=domain,
                    group_id=f"ultradata:{domain}:{source_id}",
                    source_id=source_id,
                )
