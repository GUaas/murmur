from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from muddywater.config import load_config
from muddywater.config_validation import validate_pretrain_config
from muddywater.dataset import load_labeled_texts
from muddywater.paths import resolve_config_path, resolve_config_paths_in_data
from muddywater.tokenizer import CharacterTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_values(paths: list[str], source_key: str) -> set[str]:
    values: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping) or source_key not in record:
                    raise ValueError(f"Missing source field at {path}:{line_no}")
                values.add(str(record[source_key]).strip())
    return values


def _labeled_samples(data: Mapping[str, Any], paths: list[str]) -> list[tuple[str, str]]:
    return load_labeled_texts(
        paths,
        jsonl_format=str(data.get("jsonl_format", "source_target")),
        source_key=str(data.get("source_key", "source")),
        target_key=str(data.get("target_key", "target")),
        target_separator=str(data.get("target_separator", "<|im_start|>")),
        source_label=str(data.get("source_label", "")),
        target_label=data.get("target_label"),
        min_chars=int(data.get("min_chars", 1)),
    )


def validate_setup(config_path: Path, *, require_checkpoint: bool = True) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    validate_pretrain_config(config)
    data = resolve_config_paths_in_data(dict(config["data"]), config_path)
    tokenizer_config = dict(config["tokenizer"])
    tokenizer_path = resolve_config_path(tokenizer_config["path"], config_path=config_path)
    tokenizer = CharacterTokenizer.load(tokenizer_path)

    expected_tokenizer_sha = str(tokenizer_config.get("sha256", "")).lower()
    actual_tokenizer_sha = _sha256(tokenizer_path)
    if expected_tokenizer_sha and actual_tokenizer_sha != expected_tokenizer_sha:
        raise ValueError(
            f"Tokenizer SHA-256 mismatch: expected={expected_tokenizer_sha} "
            f"actual={actual_tokenizer_sha}"
        )

    labels = [str(data.get("source_label", "")), str(data.get("target_label", ""))]
    label_ids: dict[str, int] = {}
    for label in labels:
        if not label:
            raise ValueError("Both source_label and target_label must be configured")
        token_id = tokenizer.token_to_id.get(label)
        if token_id is None or int(token_id) == tokenizer.unk_id:
            raise ValueError(f"Reserved label is not in the tokenizer vocabulary: {label!r}")
        encoded_probe = tokenizer.encode(
            f"probe{label}probe",
            add_bos=False,
            add_eos=False,
            truncation=False,
        )
        if encoded_probe.count(int(token_id)) != 1:
            raise ValueError(
                f"Reserved label is not preserved atomically by the tokenizer: "
                f"{label!r} -> {encoded_probe}"
            )
        label_ids[label] = int(token_id)
    if len(set(label_ids.values())) != len(label_ids):
        raise ValueError("Source and target labels map to the same tokenizer ID")

    train_paths = list(data["train_paths"])
    validation_paths = list(data.get("val_paths", []))
    train_samples = _labeled_samples(data, train_paths)
    validation_samples = _labeled_samples(data, validation_paths)
    max_seq_len = int(config["model"]["max_seq_len"])
    encoded_lengths = [
        len(tokenizer.encode(text, add_bos=True, add_eos=True, truncation=False))
        for text, _ in train_samples + validation_samples
    ]
    over_limit = sum(length > max_seq_len + 1 for length in encoded_lengths)
    if over_limit:
        raise ValueError(
            f"{over_limit} examples exceed the configured dataset window of {max_seq_len + 1} tokens"
        )

    source_key = str(data.get("source_key", "source"))
    train_sources = _load_source_values(train_paths, source_key)
    validation_sources = _load_source_values(validation_paths, source_key)
    source_overlap = train_sources & validation_sources
    if source_overlap:
        raise ValueError(f"Train/validation source leakage detected: {len(source_overlap)} sources")

    checkpoint_path = resolve_config_path(
        config["training"]["init_from"], config_path=config_path
    )
    if require_checkpoint and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")

    return {
        "status": "ready",
        "config": str(config_path),
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": actual_tokenizer_sha,
            "vocab_size": tokenizer.vocab_size,
            "label_ids": label_ids,
        },
        "data": {
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "source_overlap": 0,
            "max_encoded_tokens": max(encoded_lengths),
            "configured_window_tokens": max_seq_len + 1,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "present": checkpoint_path.is_file(),
            "bytes": checkpoint_path.stat().st_size if checkpoint_path.is_file() else 0,
        },
    }
