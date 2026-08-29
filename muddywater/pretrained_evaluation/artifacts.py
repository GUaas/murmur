from __future__ import annotations

import inspect
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from muddywater.model import GPTConfig, GPTLanguageModel
from muddywater.tokenizer import CharacterTokenizer
from muddywater.utils import file_sha256


@dataclass
class LoadedArtifacts:
    model: GPTLanguageModel
    tokenizer: CharacterTokenizer
    checkpoint_path: Path
    tokenizer_path: Path
    checkpoint_metadata: dict[str, Any]
    artifact_audit: dict[str, Any]


def _safe_load(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    signature = inspect.signature(torch.load)
    if "weights_only" in signature.parameters:
        kwargs["weights_only"] = True
    if "mmap" in signature.parameters:
        kwargs["mmap"] = True
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary: {path}")
    return payload


def _state_dict(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model_state", payload)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint contains no model state dictionary")
    non_tensors = [name for name, value in state.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise ValueError(f"Model state contains non-tensors: {non_tensors[:5]}")
    return state


def audit_state_tensors(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    dtype_elements: Counter[str] = Counter()
    nonfinite: dict[str, int] = {}
    all_zero: list[str] = []
    largest_absolute_value = 0.0
    largest_absolute_tensor = None
    total_elements = 0

    for name, tensor in state.items():
        count = int(tensor.numel())
        total_elements += count
        dtype_elements[str(tensor.dtype)] += count
        if tensor.is_floating_point() or tensor.is_complex():
            bad = int((~torch.isfinite(tensor)).sum().item())
            if bad:
                nonfinite[name] = bad
        if count:
            absolute_max = float(tensor.abs().max().item())
            if absolute_max == 0.0:
                all_zero.append(name)
            if absolute_max > largest_absolute_value:
                largest_absolute_value = absolute_max
                largest_absolute_tensor = name

    embedding_key = "transformer.wte.weight"
    head_key = "lm_head.weight"
    tied_serialized_equal = (
        embedding_key in state
        and head_key in state
        and torch.equal(state[embedding_key], state[head_key])
    )
    return {
        "tensor_count": len(state),
        "serialized_elements": total_elements,
        "dtype_elements": dict(dtype_elements),
        "all_finite": not nonfinite,
        "nonfinite_by_tensor": nonfinite,
        "all_zero_tensors": all_zero,
        "largest_absolute_value": largest_absolute_value,
        "largest_absolute_tensor": largest_absolute_tensor,
        "serialized_embedding_equals_lm_head": tied_serialized_equal,
    }


def load_artifacts(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: torch.device,
) -> LoadedArtifacts:
    checkpoint_path = Path(checkpoint_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    payload = _safe_load(checkpoint_path)
    state = _state_dict(payload)
    state_audit = audit_state_tensors(state)
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary")
    model_config = config.get("model", {})
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError("Checkpoint is missing config.model architecture metadata")

    tokenizer = CharacterTokenizer.load(tokenizer_path)
    tokenizer_sha256 = file_sha256(tokenizer_path)
    expected_tokenizer_sha256 = str(config.get("tokenizer", {}).get("sha256", ""))
    tokenizer_hash_matches = bool(expected_tokenizer_sha256) and (
        tokenizer_sha256.lower() == expected_tokenizer_sha256.lower()
    )
    if expected_tokenizer_sha256 and not tokenizer_hash_matches:
        raise ValueError(
            "Tokenizer SHA-256 does not match checkpoint metadata: "
            f"expected={expected_tokenizer_sha256}, actual={tokenizer_sha256}"
        )
    expected_vocab = int(model_config.get("vocab_size", tokenizer.vocab_size))
    if expected_vocab != int(tokenizer.vocab_size):
        raise ValueError(
            f"Tokenizer vocabulary mismatch: checkpoint={expected_vocab}, "
            f"tokenizer={tokenizer.vocab_size}"
        )

    resolved_model_config = dict(model_config)
    resolved_model_config["vocab_size"] = int(tokenizer.vocab_size)
    model = GPTLanguageModel(GPTConfig.from_dict(resolved_model_config))
    incompatible = model.load_state_dict(state, strict=True)
    model.eval().to(device)

    unique_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    best_val_loss = payload.get("best_val_loss")
    recorded_ppl = (
        math.exp(min(float(best_val_loss), 50.0)) if best_val_loss is not None else None
    )
    metadata = {key: value for key, value in payload.items() if key != "model_state"}
    audit = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
            "format": payload.get("format"),
            "step": payload.get("step"),
            "epoch": payload.get("epoch"),
            "recorded_best_val_loss": best_val_loss,
            "recorded_best_val_perplexity": recorded_ppl,
            "recorded_metric_is_independently_verified": False,
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "bytes": tokenizer_path.stat().st_size,
            "sha256": tokenizer_sha256,
            "expected_sha256": expected_tokenizer_sha256 or None,
            "sha256_matches_checkpoint": tokenizer_hash_matches,
            "vocab_size": int(tokenizer.vocab_size),
        },
        "state": state_audit,
        "model": {
            "config": vars(model.config),
            "unique_parameters": unique_parameters,
            "parameter_bytes_current_dtype": parameter_bytes,
            "strict_load_passed": not incompatible.missing_keys
            and not incompatible.unexpected_keys,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "embedding_and_lm_head_are_tied": (
                model.transformer["wte"].weight.data_ptr() == model.lm_head.weight.data_ptr()
            ),
        },
    }
    return LoadedArtifacts(
        model=model,
        tokenizer=tokenizer,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        checkpoint_metadata=metadata,
        artifact_audit=audit,
    )
