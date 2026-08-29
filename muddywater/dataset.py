from __future__ import annotations

import glob
import json
import random
import re
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .templates import (
    DEFAULT_SYSTEM_PROMPT,
    format_messages,
    format_messages_with_labels,
    format_sft_example,
    format_sft_example_with_labels,
)
from .packing import (
    build_document_sample_starts,
    build_sample_starts,
    cross_document_targets,
    document_ids_for_positions,
    grouped_hash_validation_mask,
    pack_labeled_token_sequences,
    validate_lm_stride,
)
from .tokenizer import CharacterTokenizer
from .source_target import SourceTargetTemplate
from .utils import as_list


SUPPORTED_EXTENSIONS = {".txt", ".jsonl"}


def discover_data_files(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in as_list(paths):
        path_str = str(raw_path)
        matched = glob.glob(path_str, recursive=True) if any(ch in path_str for ch in "*?[") else []
        candidates = [Path(p) for p in matched] if matched else [Path(raw_path)]

        for path in candidates:
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                        files.append(child)
            elif path.is_file():
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    raise ValueError(f"Unsupported data file extension: {path}")
                files.append(path)
            else:
                raise FileNotFoundError(f"Data path not found: {path}")

    unique_files = sorted({file.resolve(): file for file in files}.values())
    if not unique_files:
        raise FileNotFoundError(f"No data files found in paths: {paths}")
    return unique_files


def _read_txt(path: Path, txt_split: str) -> list[str]:
    content = path.read_text(encoding="utf-8")
    if txt_split == "document":
        return [content.strip()]
    if txt_split == "blankline":
        return [chunk.strip() for chunk in re.split(r"\n\s*\n", content) if chunk.strip()]
    if txt_split == "line":
        return [line.strip() for line in content.splitlines() if line.strip()]
    raise ValueError("txt_split must be one of: line, blankline, document")


def _field_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def _format_instruction_record(
    obj: dict,
    instruction_key: str,
    input_key: str,
    output_key: str,
    chat_template: str,
    system_prompt: str,
    instruction_template: str,
    instruction_template_no_input: str,
    path: Path,
    line_no: int,
) -> str:
    if instruction_key not in obj:
        raise KeyError(f"Missing key '{instruction_key}' at {path}:{line_no}")
    if output_key not in obj:
        raise KeyError(f"Missing key '{output_key}' at {path}:{line_no}")

    instruction = _field_to_text(obj.get(instruction_key)).strip()
    input_text = _field_to_text(obj.get(input_key)).strip()
    output = _field_to_text(obj.get(output_key)).strip()
    return format_sft_example(
        instruction=instruction,
        input_text=input_text,
        output=output,
        chat_template=chat_template,
        system_prompt=system_prompt,
        instruction_template=instruction_template,
        instruction_template_no_input=instruction_template_no_input,
    )


def _format_instruction_record_labeled(
    obj: dict,
    instruction_key: str,
    input_key: str,
    output_key: str,
    chat_template: str,
    system_prompt: str,
    path: Path,
    line_no: int,
) -> tuple[str, str]:
    if instruction_key not in obj:
        raise KeyError(f"Missing key '{instruction_key}' at {path}:{line_no}")
    if output_key not in obj:
        raise KeyError(f"Missing key '{output_key}' at {path}:{line_no}")

    instruction = _field_to_text(obj.get(instruction_key)).strip()
    input_text = _field_to_text(obj.get(input_key)).strip()
    output = _field_to_text(obj.get(output_key)).strip()
    return format_sft_example_with_labels(
        instruction=instruction,
        input_text=input_text,
        output=output,
        chat_template=chat_template,
        system_prompt=system_prompt,
    )


def _read_jsonl(
    path: Path,
    text_key: str,
    jsonl_format: str,
    instruction_key: str,
    input_key: str,
    output_key: str,
    chat_template: str,
    system_prompt: str,
    instruction_template: str,
    instruction_template_no_input: str,
) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

            if jsonl_format == "text":
                if text_key not in obj:
                    raise KeyError(f"Missing key '{text_key}' at {path}:{line_no}")
                text = _field_to_text(obj.get(text_key))
            elif jsonl_format == "instruction":
                text = _format_instruction_record(
                    obj=obj,
                    instruction_key=instruction_key,
                    input_key=input_key,
                    output_key=output_key,
                    chat_template=chat_template,
                    system_prompt=system_prompt,
                    instruction_template=instruction_template,
                    instruction_template_no_input=instruction_template_no_input,
                    path=path,
                    line_no=line_no,
                )
            elif jsonl_format == "messages":
                messages = obj.get("messages")
                if not messages or not isinstance(messages, list):
                    raise ValueError(
                        f"Expected 'messages' key with a list value at {path}:{line_no}"
                    )
                text = format_messages(
                    messages=messages,
                    chat_template=chat_template,
                    system_prompt=system_prompt,
                )
            else:
                raise ValueError("jsonl_format must be one of: text, instruction, messages")
            if text.strip():
                texts.append(text.strip())
    return texts


def load_texts(
    paths: str | Path | Iterable[str | Path],
    jsonl_text_key: str = "text",
    jsonl_format: str = "text",
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
    chat_template: str = "chatml",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    instruction_template: str = "用户：{instruction}\n输入：{input}\n助手：{output}",
    instruction_template_no_input: str = "用户：{instruction}\n助手：{output}",
    txt_split: str = "line",
    min_chars: int = 1,
) -> list[str]:
    texts: list[str] = []
    for path in discover_data_files(paths):
        suffix = path.suffix.lower()
        if suffix == ".txt":
            texts.extend(_read_txt(path, txt_split=txt_split))
        elif suffix == ".jsonl":
            texts.extend(
                _read_jsonl(
                    path,
                    text_key=jsonl_text_key,
                    jsonl_format=jsonl_format,
                    instruction_key=instruction_key,
                    input_key=input_key,
                    output_key=output_key,
                    chat_template=chat_template,
                    system_prompt=system_prompt,
                    instruction_template=instruction_template,
                    instruction_template_no_input=instruction_template_no_input,
                )
            )

    texts = [text for text in texts if len(text) >= min_chars]
    if not texts:
        raise ValueError("No non-empty texts loaded from the configured data paths.")
    return texts


def split_train_val(
    texts: list[str],
    val_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
    split_mode: str = "hash",
) -> tuple[list[str], list[str]]:
    if val_ratio <= 0 or len(texts) < 2:
        return texts, []

    if split_mode == "hash":
        validation_mask = grouped_hash_validation_mask(
            texts,
            val_ratio=val_ratio,
            seed=seed,
        )
        val_indices = {idx for idx, is_validation in enumerate(validation_mask) if is_validation}
        train_texts = [text for idx, text in enumerate(texts) if idx not in val_indices]
        val_texts = [text for idx, text in enumerate(texts) if idx in val_indices]
        return train_texts, val_texts

    if split_mode != "random":
        raise ValueError("split_mode must be one of: hash, random")

    indices = list(range(len(texts)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    val_size = max(1, int(len(texts) * val_ratio))
    val_indices = set(indices[:val_size])
    train_texts = [text for idx, text in enumerate(texts) if idx not in val_indices]
    val_texts = [text for idx, text in enumerate(texts) if idx in val_indices]
    return train_texts, val_texts


def load_labeled_texts(
    paths: str | Path | Iterable[str | Path],
    jsonl_text_key: str = "text",
    jsonl_format: str = "messages",
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
    source_key: str = "source",
    target_key: str = "target",
    target_separator: str = "<|im_start|>",
    source_label: str = "",
    target_label: str | None = None,
    chat_template: str = "chatml",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    instruction_template: str = "用户：{instruction}\n输入：{input}\n助手：{output}",
    instruction_template_no_input: str = "用户：{instruction}\n助手：{output}",
    txt_split: str = "line",
    min_chars: int = 1,
    label_mask_for_non_messages: bool = False,
) -> list[tuple[str, str]]:
    """Like :func:`load_texts`, but returns ``(text, label_mask_str)`` pairs.

    When ``jsonl_format == "source_target"`` the source and separator are
    masked while the target and automatically appended EOS are supervised.
    When ``jsonl_format == "messages"`` the label mask is generated by
    :func:`~muddywater.templates.format_messages_with_labels`. Instruction
    data can use the same assistant-only masking by setting
    ``label_mask_for_non_messages=True``. Other formats default to all
    ``'1'`` (everything is trainable).
    """
    samples: list[tuple[str, str]] = []
    for path in discover_data_files(paths):
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            if jsonl_format == "source_target":
                samples.extend(
                    _read_jsonl_source_target_labeled(
                        path,
                        source_key=source_key,
                        target_key=target_key,
                        target_separator=target_separator,
                        source_label=source_label,
                        target_label=target_label,
                    )
                )
            elif jsonl_format == "messages":
                samples.extend(
                    _read_jsonl_labeled(
                        path,
                        chat_template=chat_template,
                        system_prompt=system_prompt,
                    )
                )
            elif jsonl_format == "instruction" and label_mask_for_non_messages:
                samples.extend(
                    _read_jsonl_instruction_labeled(
                        path,
                        instruction_key=instruction_key,
                        input_key=input_key,
                        output_key=output_key,
                        chat_template=chat_template,
                        system_prompt=system_prompt,
                    )
                )
            else:
                texts = _read_jsonl(
                    path,
                    text_key=jsonl_text_key,
                    jsonl_format=jsonl_format,
                    instruction_key=instruction_key,
                    input_key=input_key,
                    output_key=output_key,
                    chat_template=chat_template,
                    system_prompt=system_prompt,
                    instruction_template=instruction_template,
                    instruction_template_no_input=instruction_template_no_input,
                )
                samples.extend((text, "1" * len(text)) for text in texts)
        elif suffix == ".txt":
            texts = _read_txt(path, txt_split=txt_split)
            samples.extend((text, "1" * len(text)) for text in texts)
        else:
            raise ValueError(f"Unsupported data file extension: {path}")

    filtered_samples: list[tuple[str, str]] = []
    for text, mask in samples:
        if len(text) != len(mask):
            raise ValueError(
                "Labeled sample text/mask length mismatch: "
                f"text={len(text)} mask={len(mask)}"
            )
        if len(text) >= min_chars and text.strip():
            filtered_samples.append((text, mask))
    samples = filtered_samples
    if not samples:
        raise ValueError("No non-empty labeled texts loaded from the configured data paths.")
    return samples


def _read_jsonl_source_target_labeled(
    path: Path,
    source_key: str,
    target_key: str,
    target_separator: str,
    source_label: str = "",
    target_label: str | None = None,
) -> list[tuple[str, str]]:
    """Read compact source/target JSONL with target-only character masks."""

    template = SourceTargetTemplate(
        source_key=source_key,
        target_key=target_key,
        target_separator=target_separator,
        source_label=source_label,
        target_label=target_label,
    )
    samples: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            try:
                samples.append(template.format_record(record))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid source_target row at {path}:{line_no}: {exc}") from exc
    return samples


def _read_jsonl_labeled(
    path: Path,
    chat_template: str,
    system_prompt: str,
) -> list[tuple[str, str]]:
    """Read a JSONL file in OpenAI messages format and return (text, label_mask) pairs."""
    samples: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

            messages = obj.get("messages")
            if not messages or not isinstance(messages, list):
                raise ValueError(f"Expected 'messages' key at {path}:{line_no}")
            text, mask = format_messages_with_labels(
                messages=messages,
                chat_template=chat_template,
                system_prompt=system_prompt,
            )
            if len(text) != len(mask):
                raise ValueError(
                    f"Labeled text/mask length mismatch at {path}:{line_no}: "
                    f"text={len(text)} mask={len(mask)}"
                )
            if text.strip():
                samples.append((text, mask))
    return samples


def _read_jsonl_instruction_labeled(
    path: Path,
    instruction_key: str,
    input_key: str,
    output_key: str,
    chat_template: str,
    system_prompt: str,
) -> list[tuple[str, str]]:
    """Read instruction JSONL and return assistant-only label masks."""
    samples: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

            text, mask = _format_instruction_record_labeled(
                obj=obj,
                instruction_key=instruction_key,
                input_key=input_key,
                output_key=output_key,
                chat_template=chat_template,
                system_prompt=system_prompt,
                path=path,
                line_no=line_no,
            )
            if len(text) != len(mask):
                raise ValueError(
                    f"Labeled text/mask length mismatch at {path}:{line_no}: "
                    f"text={len(text)} mask={len(mask)}"
                )
            if text.strip():
                samples.append((text, mask))
    return samples


class LanguageModelingDataset(Dataset):
    """Fixed-length next-token-prediction samples for decoder-only training."""

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: CharacterTokenizer,
        max_seq_len: int,
        stride: int | None = None,
        add_bos: bool = True,
        ignore_index: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.window_size = self.max_seq_len + 1
        self.stride = validate_lm_stride(self.max_seq_len, stride)
        self.add_bos = bool(add_bos)
        self.ignore_index = int(ignore_index)
        self.samples: list[list[int]] = []

        for text in texts:
            ids = tokenizer.encode(text, add_bos=self.add_bos, add_eos=True)
            self.samples.extend(self._make_samples(ids))

        if not self.samples:
            raise ValueError("LanguageModelingDataset received no usable samples.")

    def _make_samples(self, ids: list[int]) -> list[list[int]]:
        if len(ids) <= self.window_size:
            return [self._pad(ids)]

        samples: list[list[int]] = []
        for start in range(0, len(ids) - 1, self.stride):
            chunk = ids[start : start + self.window_size]
            if len(chunk) < 2:
                break
            samples.append(self._pad(chunk))
            if start + self.window_size >= len(ids):
                break
        return samples

    def _pad(self, ids: list[int]) -> list[int]:
        ids = ids[: self.window_size]
        if len(ids) < self.window_size:
            ids = ids + [self.tokenizer.pad_id] * (self.window_size - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.samples[index], dtype=torch.long)
        input_ids = ids[:-1]
        labels = ids[1:].clone()
        labels[labels == self.tokenizer.pad_id] = self.ignore_index
        attention_mask = (input_ids != self.tokenizer.pad_id).to(torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class LabeledLanguageModelingDataset(Dataset):
    """Fixed-length next-token-prediction dataset with assistant-only loss.

    Each sample is a ``(text, label_mask_str)`` tuple.  ``label_mask_str``
    has the same length as *text*; ``'1'`` characters denote positions
    that should contribute to the loss (i.e. assistant responses) and
    ``'0'`` characters denote positions that should be ignored
    (system / user / chat markers).

    Padded positions always receive ``ignore_index`` as the label, same as
    the base :class:`LanguageModelingDataset`.
    """

    def __init__(
        self,
        labeled_texts: Iterable[tuple[str, str]],
        tokenizer: CharacterTokenizer,
        max_seq_len: int,
        stride: int | None = None,
        add_bos: bool = True,
        label_mask_policy: str = "any",
        ignore_index: int = -100,
        pack_sequences: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.window_size = self.max_seq_len + 1
        self.stride = validate_lm_stride(self.max_seq_len, stride)
        self.add_bos = bool(add_bos)
        self.ignore_index = int(ignore_index)
        self.pack_sequences = bool(pack_sequences)
        self.label_mask_policy = self._normalize_label_mask_policy(label_mask_policy)
        self.samples: list[tuple[list[int], list[int], list[int] | None]] = []
        self.skipped_empty_label_windows = 0
        self.supervised_tokens = 0
        encoded_sequences: list[tuple[list[int], list[int]]] = []

        for text, label_mask_str in labeled_texts:
            ids, mask_ids = self._encode_labeled(text, label_mask_str)
            encoded_sequences.append((ids, mask_ids))

        if self.pack_sequences:
            packed_windows, packing_stats = pack_labeled_token_sequences(
                encoded_sequences,
                max_seq_len=self.max_seq_len,
                pad_id=self.tokenizer.pad_id,
                stride=self.stride,
            )
            self.samples = [
                (window.token_ids, window.label_mask, window.document_ids)
                for window in packed_windows
            ]
            self.skipped_empty_label_windows = int(packing_stats["skipped_empty_windows"])
            self.supervised_tokens = int(packing_stats["supervised_tokens"])
            self.packing_stats = packing_stats
        else:
            for ids, mask_ids in encoded_sequences:
                self.samples.extend(self._make_labeled_samples(ids, mask_ids))
            non_padding_tokens = sum(
                sum(1 for token_id in ids if token_id != self.tokenizer.pad_id)
                for ids, _, _ in self.samples
            )
            capacity = len(self.samples) * self.window_size
            self.packing_stats = {
                "input_sequences": len(encoded_sequences),
                "fragments": len(self.samples) + self.skipped_empty_label_windows,
                "windows": len(self.samples),
                "packed_windows": 0,
                "skipped_empty_windows": self.skipped_empty_label_windows,
                "non_padding_tokens": non_padding_tokens,
                "padding_tokens": capacity - non_padding_tokens,
                "supervised_tokens": self.supervised_tokens,
                "packing_efficiency": (
                    round(non_padding_tokens / capacity, 8) if capacity else 0.0
                ),
            }

        if not self.samples:
            raise ValueError("LabeledLanguageModelingDataset received no usable samples.")

    def _encode_labeled(self, text: str, label_mask_str: str) -> tuple[list[int], list[int]]:
        """Encode text and produce aligned token-level mask ids."""
        if len(text) != len(label_mask_str):
            raise ValueError(
                "Labeled sample text/mask length mismatch before encoding: "
                f"text={len(text)} mask={len(label_mask_str)}"
            )
        ids, spans = self.tokenizer.encode_with_char_spans(
            text,
            add_bos=self.add_bos,
            add_eos=True,
        )
        mask: list[int] = []
        supervise_eos = bool(label_mask_str) and label_mask_str[-1] == "1"
        for token_id, span in zip(ids, spans):
            if span is None:
                mask.append(1 if supervise_eos and token_id == self.tokenizer.eos_id else 0)
                continue
            start, end = span
            if start < 0 or end <= start:
                mask.append(0)
                continue
            mask_slice = label_mask_str[start:end]
            mask.append(1 if self._span_is_trainable(mask_slice) else 0)
        return ids, mask

    @staticmethod
    def _normalize_label_mask_policy(policy: str) -> str:
        normalized = str(policy or "any").strip().lower()
        if normalized not in {"any", "all", "majority"}:
            raise ValueError("label_mask_policy must be one of: any, all, majority")
        return normalized

    def _span_is_trainable(self, mask_slice: str) -> bool:
        if not mask_slice:
            return False
        trainable_count = mask_slice.count("1")
        if self.label_mask_policy == "all":
            return trainable_count == len(mask_slice)
        if self.label_mask_policy == "majority":
            return trainable_count * 2 >= len(mask_slice)
        return trainable_count > 0

    def _make_labeled_samples(
        self, ids: list[int], mask_ids: list[int]
    ) -> list[tuple[list[int], list[int], list[int] | None]]:
        assert len(ids) == len(mask_ids)
        if len(ids) <= self.window_size:
            samples: list[tuple[list[int], list[int], list[int] | None]] = []
            self._append_labeled_sample(samples, ids, mask_ids)
            return samples

        samples: list[tuple[list[int], list[int], list[int] | None]] = []
        for start in range(0, len(ids) - 1, self.stride):
            chunk_ids = ids[start : start + self.window_size]
            chunk_mask = mask_ids[start : start + self.window_size]
            if len(chunk_ids) < 2:
                break
            self._append_labeled_sample(samples, chunk_ids, chunk_mask)
            if start + self.window_size >= len(ids):
                break
        return samples

    def _append_labeled_sample(
        self,
        samples: list[tuple[list[int], list[int], list[int] | None]],
        ids: list[int],
        mask_ids: list[int],
    ) -> None:
        padded_ids, padded_mask = self._pad_labeled(ids, mask_ids)
        supervised = sum(padded_mask[1:])
        if supervised <= 0:
            self.skipped_empty_label_windows += 1
            return
        self.supervised_tokens += supervised
        samples.append((padded_ids, padded_mask, None))

    def _pad_labeled(
        self, ids: list[int], mask_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        ids = ids[: self.window_size]
        mask_ids = mask_ids[: self.window_size]
        if len(ids) < self.window_size:
            pad_len = self.window_size - len(ids)
            ids = ids + [self.tokenizer.pad_id] * pad_len
            mask_ids = mask_ids + [0] * pad_len
        return ids, mask_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids, mask_ids, document_ids = self.samples[index]
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        mask_tensor = torch.tensor(mask_ids, dtype=torch.long)

        input_ids = ids_tensor[:-1]
        labels = ids_tensor[1:].clone()
        label_mask = mask_tensor[1:]

        # Mask non-assistant positions
        labels[label_mask == 0] = self.ignore_index

        attention_mask = (input_ids != self.tokenizer.pad_id).to(torch.long)
        item = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if document_ids is not None:
            item["document_ids"] = torch.tensor(document_ids[:-1], dtype=torch.long)
        return item


class TokenBlockDataset(Dataset):
    """Random-access next-token samples from a contiguous token-id memmap.

    This is intended for real pretraining runs: preprocess JSONL once into
    an EOS-packed token stream, memory-map it, then train on fixed windows
    without padding every short document independently.
    """

    def __init__(
        self,
        token_path: str | Path,
        max_seq_len: int,
        stride: int | None = None,
        tail_min_gap_ratio: float = 0.5,
        dtype: str | np.dtype | None = None,
        expected_vocab_size: int | None = None,
        expected_tokenizer_sha256: str | None = None,
        expected_add_bos: bool | None = None,
        strict_meta: bool = True,
        document_attention: bool = False,
        ignore_cross_document_targets: bool = False,
        single_document_windows: bool = False,
        ignore_index: int = -100,
    ) -> None:
        self.token_path = Path(token_path)
        if not self.token_path.exists():
            raise FileNotFoundError(f"Token cache not found: {self.token_path}")

        self.max_seq_len = int(max_seq_len)
        self.window_size = self.max_seq_len + 1
        self.stride = validate_lm_stride(self.max_seq_len, stride)
        self.tail_min_gap_ratio = float(tail_min_gap_ratio)
        self.meta_path = self.token_path.with_suffix(self.token_path.suffix + ".meta.json")
        self.meta = self._load_meta()
        self.document_attention = bool(document_attention)
        self.ignore_cross_document_targets = bool(ignore_cross_document_targets)
        self.single_document_windows = bool(single_document_windows)
        self.ignore_index = int(ignore_index)
        self._validate_meta(
            expected_vocab_size=expected_vocab_size,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            expected_add_bos=expected_add_bos,
            strict_meta=strict_meta,
        )
        self.dtype = np.dtype(dtype or self.meta.get("dtype", "uint32"))
        self.tokens = np.memmap(self.token_path, dtype=self.dtype, mode="r")
        self.num_tokens = int(self.meta.get("num_tokens", self.tokens.shape[0]))
        if self.num_tokens != int(self.tokens.shape[0]):
            raise ValueError(
                f"Token cache metadata mismatch for {self.token_path}: "
                f"meta num_tokens={self.num_tokens}, file tokens={self.tokens.shape[0]}"
            )
        if self.num_tokens < self.window_size:
            raise ValueError(
                f"Token cache {self.token_path} has {self.num_tokens} tokens, "
                f"need at least {self.window_size}."
            )
        self.doc_starts = self._load_doc_starts() if self._needs_doc_starts else None
        if self.single_document_windows:
            if self.doc_starts is None:
                raise ValueError("single_document_windows requires document boundary metadata")
            self.sample_starts = build_document_sample_starts(
                doc_starts=self.doc_starts,
                num_tokens=self.num_tokens,
                window_size=self.window_size,
                stride=self.stride,
                tail_min_gap_ratio=self.tail_min_gap_ratio,
            )
            if self.sample_starts.size == 0:
                raise ValueError(
                    f"Token cache {self.token_path} has no document with at least "
                    f"{self.window_size} tokens for single-document windows."
                )
        else:
            self.sample_starts = build_sample_starts(
                num_tokens=self.num_tokens,
                window_size=self.window_size,
                stride=self.stride,
                tail_min_gap_ratio=self.tail_min_gap_ratio,
            )
        self.num_samples = int(self.sample_starts.size)

    @property
    def _needs_doc_starts(self) -> bool:
        return (
            self.document_attention
            or self.ignore_cross_document_targets
            or self.single_document_windows
        )

    def _load_meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        with self.meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _doc_starts_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        meta_value = self.meta.get("doc_starts_file") or self.meta.get("document_starts_file")
        if meta_value:
            meta_path = Path(str(meta_value))
            candidates.append(meta_path if meta_path.is_absolute() else self.token_path.parent / meta_path)
        candidates.append(self.token_path.with_suffix(self.token_path.suffix + ".doc_starts.npy"))
        return candidates

    def _load_doc_starts(self) -> np.ndarray:
        for path in self._doc_starts_candidates():
            if path.exists():
                doc_starts = np.load(path)
                return self._validate_doc_starts(doc_starts, path)
        searched = ", ".join(str(path) for path in self._doc_starts_candidates())
        raise ValueError(
            f"Document boundary metadata is required for {self.token_path}. "
            f"Looked for: {searched}. Rebuild the cache with scripts/prepare_pretrain_cache.py."
        )

    def _validate_doc_starts(self, doc_starts: np.ndarray, path: Path) -> np.ndarray:
        starts = np.asarray(doc_starts, dtype=np.uint64)
        if starts.ndim != 1 or starts.size == 0:
            raise ValueError(f"Document starts must be a non-empty 1D array: {path}")
        if int(starts[0]) != 0:
            raise ValueError(f"First document start must be 0 in {path}")
        if int(starts[-1]) >= self.num_tokens:
            raise ValueError(
                f"Last document start {int(starts[-1])} is outside token cache "
                f"with {self.num_tokens} tokens: {path}"
            )
        if np.any(starts[1:] <= starts[:-1]):
            raise ValueError(f"Document starts must be strictly increasing: {path}")
        return starts

    def _validate_meta(
        self,
        expected_vocab_size: int | None,
        expected_tokenizer_sha256: str | None,
        expected_add_bos: bool | None,
        strict_meta: bool,
    ) -> None:
        if not strict_meta:
            return
        if not self.meta:
            raise ValueError(
                f"Token cache metadata is required for {self.token_path}. "
                "Rebuild it with scripts/prepare_pretrain_cache.py."
            )

        if expected_vocab_size is not None:
            meta_vocab_size = self.meta.get("vocab_size")
            if meta_vocab_size is None:
                raise ValueError(
                    f"Token cache metadata missing vocab_size for {self.token_path}. "
                    "Rebuild the cache so tokenizer compatibility can be checked."
                )
            if int(meta_vocab_size) != int(expected_vocab_size):
                raise ValueError(
                    f"Token cache vocab mismatch for {self.token_path}: "
                    f"cache={meta_vocab_size}, current_tokenizer={expected_vocab_size}."
                )

        if expected_tokenizer_sha256 is not None:
            meta_sha = self.meta.get("tokenizer_sha256")
            if not meta_sha:
                raise ValueError(
                    f"Token cache metadata missing tokenizer_sha256 for {self.token_path}. "
                    "Rebuild the cache with the current prepare_pretrain_cache.py."
                )
            if str(meta_sha).lower() != expected_tokenizer_sha256.lower():
                raise ValueError(
                    f"Token cache tokenizer mismatch for {self.token_path}. "
                    "The cache was built with a different tokenizer file."
                )

        if expected_add_bos is not None:
            meta_add_bos = self.meta.get("add_bos")
            if meta_add_bos is None:
                raise ValueError(
                    f"Token cache metadata missing add_bos for {self.token_path}. "
                    "Rebuild the cache with the current prepare_pretrain_cache.py."
                )
            if bool(meta_add_bos) != bool(expected_add_bos):
                raise ValueError(
                    f"Token cache BOS setting mismatch for {self.token_path}: "
                    f"cache={bool(meta_add_bos)}, config={bool(expected_add_bos)}."
                )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        start = int(self.sample_starts[index])
        window = np.asarray(self.tokens[start : start + self.window_size], dtype=np.int64)
        ids = torch.from_numpy(window.copy()).to(torch.long)
        input_ids = ids[:-1]
        labels = ids[1:].clone()
        item = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if self.doc_starts is not None:
            positions = np.arange(start, start + self.window_size, dtype=np.uint64)
            doc_ids = document_ids_for_positions(self.doc_starts, positions)
            if self.ignore_cross_document_targets:
                labels[cross_document_targets(doc_ids)] = self.ignore_index
            if self.document_attention:
                input_doc_ids = doc_ids[:-1]
                item["document_ids"] = torch.from_numpy(input_doc_ids.astype(np.int64, copy=False))
                item["document_mask_needed"] = torch.tensor(
                    bool(np.any(input_doc_ids != input_doc_ids[0])),
                    dtype=torch.bool,
                )
        return item


class ShardedTokenBlockDataset(Dataset):
    """Concatenate multiple token cache shards behind one Dataset interface."""

    def __init__(
        self,
        token_paths: Iterable[str | Path],
        max_seq_len: int,
        stride: int | None = None,
        tail_min_gap_ratio: float = 0.5,
        dtype: str | np.dtype | None = None,
        expected_vocab_size: int | None = None,
        expected_tokenizer_sha256: str | None = None,
        expected_add_bos: bool | None = None,
        strict_meta: bool = True,
        document_attention: bool = False,
        ignore_cross_document_targets: bool = False,
        single_document_windows: bool = False,
        ignore_index: int = -100,
    ) -> None:
        paths = [Path(path) for path in token_paths]
        if not paths:
            raise ValueError("ShardedTokenBlockDataset requires at least one token shard.")

        self.shards = [
            TokenBlockDataset(
                token_path=path,
                max_seq_len=max_seq_len,
                stride=stride,
                tail_min_gap_ratio=tail_min_gap_ratio,
                dtype=dtype,
                expected_vocab_size=expected_vocab_size,
                expected_tokenizer_sha256=expected_tokenizer_sha256,
                expected_add_bos=expected_add_bos,
                strict_meta=strict_meta,
                document_attention=document_attention,
                ignore_cross_document_targets=ignore_cross_document_targets,
                single_document_windows=single_document_windows,
                ignore_index=ignore_index,
            )
            for path in paths
        ]
        self.max_seq_len = int(max_seq_len)
        self.stride = validate_lm_stride(self.max_seq_len, stride)
        self.tail_min_gap_ratio = float(tail_min_gap_ratio)
        self.document_attention = bool(document_attention)
        self.ignore_cross_document_targets = bool(ignore_cross_document_targets)
        self.single_document_windows = bool(single_document_windows)
        self.ignore_index = int(ignore_index)
        self.cumulative_sizes: list[int] = []
        running = 0
        for shard in self.shards:
            running += len(shard)
            self.cumulative_sizes.append(running)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = bisect_right(self.cumulative_sizes, index)
        previous = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
        return self.shards[shard_idx][index - previous]


def token_cache_dtype(vocab_size: int) -> np.dtype:
    return np.dtype("uint16" if vocab_size <= np.iinfo(np.uint16).max else "uint32")
