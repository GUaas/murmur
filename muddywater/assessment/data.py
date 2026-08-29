from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .types import ChoiceExample


DATASET_SERVER = "https://datasets-server.huggingface.co"


class HFDatasetClient:
    """Small cached client for the Hugging Face dataset-viewer API.

    The evaluator intentionally avoids the heavyweight ``datasets`` dependency.
    Every response is cached verbatim and identified by a stable request hash.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        timeout_seconds: int = 60,
        minimum_network_interval_seconds: float = 1.25,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = int(timeout_seconds)
        self.minimum_network_interval_seconds = float(minimum_network_interval_seconds)
        self._last_network_request_at = 0.0
        self.requests: list[dict[str, Any]] = []

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode(sorted((str(k), str(v)) for k, v in params.items()))
        url = f"{DATASET_SERVER}/{endpoint}?{query}"
        request_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{endpoint}_{request_hash}.json"
        cache_hit = cache_path.exists()
        if cache_hit:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            error: Exception | None = None
            for attempt in range(8):
                try:
                    wait_seconds = self.minimum_network_interval_seconds - (
                        time.monotonic() - self._last_network_request_at
                    )
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                    request = urllib.request.Request(
                        url,
                        headers={"User-Agent": "murmur-comprehensive-evaluator/1.0"},
                    )
                    with urllib.request.urlopen(
                        request,
                        timeout=self.timeout_seconds,
                    ) as response:
                        raw = response.read()
                    self._last_network_request_at = time.monotonic()
                    payload = json.loads(raw.decode("utf-8"))
                    cache_path.write_bytes(raw)
                    break
                except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                    self._last_network_request_at = time.monotonic()
                    error = exc
                    if isinstance(exc, urllib.error.HTTPError) and exc.code not in {
                        408,
                        425,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }:
                        raise RuntimeError(f"Dataset request failed: {url}") from error
                    if attempt == 7:
                        raise RuntimeError(f"Dataset request failed: {url}") from error
                    retry_after = 0
                    if isinstance(exc, urllib.error.HTTPError):
                        try:
                            retry_after = int(exc.headers.get("Retry-After", "0"))
                        except (TypeError, ValueError):
                            retry_after = 0
                    time.sleep(max(retry_after, min(30, 2**attempt)))
            else:  # pragma: no cover - defensive; loop either breaks or raises.
                raise RuntimeError(f"Dataset request failed: {url}") from error
        self.requests.append(
            {
                "endpoint": endpoint,
                "params": params,
                "url": url,
                "request_sha256": request_hash,
                "response_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                "cache_file": str(cache_path),
                "cache_hit": cache_hit,
            }
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected dataset response for {url}")
        return payload

    def split_size(self, dataset: str, config: str, split: str) -> int:
        payload = self._get_json(
            "size",
            {"dataset": dataset, "config": config, "split": split},
        )
        for item in payload.get("size", {}).get("splits", []):
            if item.get("split") == split:
                return int(item["num_rows"])
        raise KeyError(f"No size metadata for {dataset}/{config}/{split}")

    def rows(
        self,
        dataset: str,
        config: str,
        split: str,
        offset: int,
        length: int,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            "rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": int(offset),
                "length": min(100, int(length)),
            },
        )
        return [dict(item) for item in payload.get("rows", [])]

    def splits(self, dataset: str) -> list[dict[str, Any]]:
        payload = self._get_json("splits", {"dataset": dataset})
        return [dict(item) for item in payload.get("splits", [])]

    def all_rows(self, dataset: str, config: str, split: str) -> list[dict[str, Any]]:
        total = self.split_size(dataset, config, split)
        output: list[dict[str, Any]] = []
        for offset in range(0, total, 100):
            output.extend(self.rows(dataset, config, split, offset, min(100, total - offset)))
        return output

    def block_sample_rows(
        self,
        dataset: str,
        config: str,
        split: str,
        sample_count: int,
        block_size: int = 40,
    ) -> list[dict[str, Any]]:
        """Sample deterministic blocks spread across a split.

        This is explicitly a resource-bounded sample, not a claim of a full
        official benchmark run. Spreading blocks avoids a first-N-only bias.
        """

        total = self.split_size(dataset, config, split)
        target = min(int(sample_count), total)
        if target >= total:
            return self.all_rows(dataset, config, split)
        blocks = max(1, math.ceil(target / int(block_size)))
        max_offset = max(0, total - int(block_size))
        offsets = (
            [0]
            if blocks == 1
            else [round(index * max_offset / (blocks - 1)) for index in range(blocks)]
        )
        by_index: dict[int, dict[str, Any]] = {}
        for offset in offsets:
            for item in self.rows(dataset, config, split, offset, block_size):
                by_index[int(item["row_idx"])] = item
        ordered = [by_index[index] for index in sorted(by_index)]
        return _even_sample(ordered, target)


def _even_sample(values: list[Any], count: int) -> list[Any]:
    if count >= len(values):
        return list(values)
    if count <= 0:
        return []
    if count == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]


def _clean_hellaswag(text: str) -> str:
    text = str(text).replace("[title]", " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _answer_index_from_labels(labels: Iterable[Any], answer: Any) -> int:
    normalized = [str(value).strip() for value in labels]
    answer_text = str(answer).strip()
    if answer_text not in normalized:
        raise ValueError(f"Answer {answer_text!r} is absent from labels {normalized!r}")
    return normalized.index(answer_text)


def adapt_hellaswag(items: list[dict[str, Any]]) -> list[ChoiceExample]:
    output = []
    for item in items:
        row = item["row"]
        label = str(row.get("label", "")).strip()
        if not label.isdigit():
            continue
        endings = tuple(" " + _clean_hellaswag(value) for value in row["endings"])
        output.append(
            ChoiceExample(
                task="hellaswag",
                example_id=str(row.get("ind", item["row_idx"])),
                category=str(row.get("activity_label", "unknown")),
                context=_clean_hellaswag(row.get("ctx", "")),
                options=endings,
                answer_index=int(label),
                primary_metric="accuracy_norm",
                metadata={"source_row_idx": int(item["row_idx"])},
            )
        )
    return output


def adapt_arc(items: list[dict[str, Any]], task: str) -> list[ChoiceExample]:
    output = []
    for item in items:
        row = item["row"]
        choices = row["choices"]
        texts = tuple(" " + str(value).strip() for value in choices["text"])
        answer_index = _answer_index_from_labels(choices["label"], row["answerKey"])
        output.append(
            ChoiceExample(
                task=task,
                example_id=str(row.get("id", item["row_idx"])),
                category="science",
                context=f"Question: {str(row['question']).strip()}\nAnswer:",
                options=texts,
                answer_index=answer_index,
                primary_metric="accuracy_norm",
                metadata={"source_row_idx": int(item["row_idx"])},
            )
        )
    return output


def adapt_boolq(items: list[dict[str, Any]]) -> list[ChoiceExample]:
    output = []
    for item in items:
        row = item["row"]
        question = str(row["question"]).strip().rstrip("?") + "?"
        output.append(
            ChoiceExample(
                task="boolq",
                example_id=str(item["row_idx"]),
                category="reading_comprehension",
                context=f"{str(row['passage']).strip()}\nQuestion: {question}\nAnswer:",
                options=(" no", " yes"),
                answer_index=1 if bool(row["answer"]) else 0,
                metadata={"source_row_idx": int(item["row_idx"])},
            )
        )
    return output


def adapt_winogrande(items: list[dict[str, Any]]) -> list[ChoiceExample]:
    output = []
    for item in items:
        row = item["row"]
        sentence = str(row["sentence"])
        if sentence.count("_") != 1:
            continue
        before, after = sentence.split("_", 1)
        options = (
            str(row["option1"]) + after,
            str(row["option2"]) + after,
        )
        output.append(
            ChoiceExample(
                task="winogrande",
                example_id=str(item["row_idx"]),
                category="coreference",
                context=before,
                options=options,
                answer_index=int(str(row["answer"]).strip()) - 1,
                metadata={"source_row_idx": int(item["row_idx"])},
            )
        )
    return output


def _exam_prompt(question: str, choices: Iterable[Any], answer_label: str) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_lines = [f"{letters[index]}. {str(value).strip()}" for index, value in enumerate(choices)]
    return f"{str(question).strip()}\n" + "\n".join(choice_lines) + f"\n{answer_label}"


def adapt_mmlu(items: list[dict[str, Any]], per_subject: int) -> list[ChoiceExample]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["row"].get("subject", "unknown"))].append(item)
    output = []
    for subject in sorted(grouped):
        for item in _even_sample(grouped[subject], min(per_subject, len(grouped[subject]))):
            row = item["row"]
            choices = list(row["choices"])
            header = (
                "The following are multiple choice questions (with answers) about "
                f"{subject.replace('_', ' ')}.\n\n"
            )
            output.append(
                ChoiceExample(
                    task="mmlu_zero_shot",
                    example_id=f"{subject}:{item['row_idx']}",
                    category=subject,
                    context=header + _exam_prompt(row["question"], choices, "Answer:"),
                    options=tuple(f" {letter}" for letter in "ABCD"[: len(choices)]),
                    answer_index=int(row["answer"]),
                    metadata={"source_row_idx": int(item["row_idx"]), "subject": subject},
                )
            )
    return output


def adapt_ceval(items: list[dict[str, Any]], subject: str, per_subject: int) -> list[ChoiceExample]:
    output = []
    for item in _even_sample(items, min(per_subject, len(items))):
        row = item["row"]
        choices = [row[letter] for letter in "ABCD"]
        answer = str(row["answer"]).strip().upper()
        if answer not in "ABCD":
            continue
        output.append(
            ChoiceExample(
                task="ceval_zero_shot",
                example_id=f"{subject}:{item['row_idx']}",
                category=subject,
                context=(
                    "以下是中国考试的单项选择题，请选出其中的正确答案。\n\n"
                    + _exam_prompt(row["question"], choices, "答案：")
                ),
                options=tuple(letter for letter in "ABCD"),
                answer_index="ABCD".index(answer),
                metadata={"source_row_idx": int(item["row_idx"]), "subject": subject},
            )
        )
    return output


def build_benchmark_suite(
    client: HFDatasetClient,
    sample_count: int = 160,
    mmlu_per_subject: int = 3,
    ceval_per_subject: int = 3,
) -> tuple[dict[str, list[ChoiceExample]], dict[str, Any]]:
    """Download/adapt a broad, resource-bounded benchmark suite."""

    suite: dict[str, list[ChoiceExample]] = {}
    suite["hellaswag"] = adapt_hellaswag(
        client.block_sample_rows("Rowan/hellaswag", "default", "validation", sample_count)
    )
    suite["arc_easy"] = adapt_arc(
        client.block_sample_rows(
            "allenai/ai2_arc", "ARC-Easy", "validation", sample_count
        ),
        "arc_easy",
    )
    suite["arc_challenge"] = adapt_arc(
        client.block_sample_rows(
            "allenai/ai2_arc", "ARC-Challenge", "validation", sample_count
        ),
        "arc_challenge",
    )
    suite["boolq"] = adapt_boolq(
        client.block_sample_rows("google/boolq", "default", "validation", sample_count)
    )
    suite["winogrande"] = adapt_winogrande(
        client.block_sample_rows(
            "allenai/winogrande",
            "winogrande_debiased",
            "validation",
            sample_count,
        )
    )

    mmlu_rows = client.all_rows("cais/mmlu", "all", "validation")
    suite["mmlu_zero_shot"] = adapt_mmlu(mmlu_rows, per_subject=mmlu_per_subject)

    ceval_splits = client.splits("ceval/ceval-exam")
    ceval_subjects = sorted(
        {
            str(item["config"])
            for item in ceval_splits
            if str(item.get("split")) == "val"
        }
    )
    ceval_examples: list[ChoiceExample] = []
    for subject in ceval_subjects:
        # Every C-Eval validation subject has fewer than 100 rows. Avoiding a
        # separate size request halves API traffic and prevents rate limiting.
        rows = client.rows("ceval/ceval-exam", subject, "val", 0, 100)
        ceval_examples.extend(adapt_ceval(rows, subject, ceval_per_subject))
    suite["ceval_zero_shot"] = ceval_examples

    provenance = {
        "method": "deterministic spread-block sample; MMLU/C-Eval stratified by subject",
        "not_official_full_run": True,
        "dataset_server": DATASET_SERVER,
        "sources": {
            "hellaswag": "https://huggingface.co/datasets/Rowan/hellaswag",
            "arc": "https://huggingface.co/datasets/allenai/ai2_arc",
            "boolq": "https://huggingface.co/datasets/google/boolq",
            "winogrande": "https://huggingface.co/datasets/allenai/winogrande",
            "mmlu": "https://huggingface.co/datasets/cais/mmlu",
            "ceval": "https://github.com/hkust-nlp/ceval",
        },
        "sample_count_per_general_task": int(sample_count),
        "mmlu_per_subject": int(mmlu_per_subject),
        "ceval_per_subject": int(ceval_per_subject),
        "task_counts": {name: len(examples) for name, examples in suite.items()},
    }
    return suite, provenance
