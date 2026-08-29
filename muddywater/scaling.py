from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

from .model_extras import layer_window_size, padded_vocab_size


@dataclass(frozen=True)
class ScalingResult:
    enabled: bool
    depth: int | None
    model_updates: dict[str, Any]
    training_updates: dict[str, Any]
    derived: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _round_power_of_two(value: float) -> int:
    value = max(1.0, float(value))
    return 2 ** round(math.log2(value))


def derive_depth_model_config(model_config: dict[str, Any]) -> dict[str, int]:
    depth = int(model_config.get("depth", model_config.get("n_layers", 12)))
    aspect_ratio = int(model_config.get("aspect_ratio", 64))
    head_dim = int(model_config.get("head_dim", 128))
    if depth <= 0:
        raise ValueError("model.depth must be positive when model.auto_scale=true")
    if aspect_ratio <= 0 or head_dim <= 0:
        raise ValueError("model.aspect_ratio and model.head_dim must be positive")

    base_dim = depth * aspect_ratio
    n_embd = math.ceil(base_dim / head_dim) * head_dim
    n_heads = max(1, n_embd // head_dim)
    n_kv_heads = model_config.get("n_kv_heads")
    if n_kv_heads is None:
        kv_ratio = int(model_config.get("kv_head_ratio", 1) or 1)
        n_kv_heads = max(1, n_heads // max(1, kv_ratio))
    n_kv_heads = int(n_kv_heads)
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"auto-scaled n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
        )
    return {
        "n_layers": depth,
        "n_embd": n_embd,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
    }


def _mlp_hidden_size(model_config: dict[str, Any], n_embd: int) -> int:
    mlp_type = str(model_config.get("mlp_type", "gelu")).lower()
    if mlp_type == "swiglu":
        hidden = int(8 * n_embd / 3)
        multiple_of = int(model_config.get("mlp_hidden_multiple_of", 1) or 1)
        if multiple_of > 1:
            hidden = math.ceil(hidden / multiple_of) * multiple_of
        return hidden
    return 4 * n_embd


def estimate_scaling_matmul_params(model_config: dict[str, Any], vocab_size: int) -> int:
    n_layers = int(model_config["n_layers"])
    n_heads = int(model_config["n_heads"])
    n_kv_heads = int(model_config.get("n_kv_heads") or n_heads)
    n_embd = int(model_config["n_embd"])
    head_dim = n_embd // n_heads
    kv_dim = n_kv_heads * head_dim
    padded_vocab = padded_vocab_size(
        int(vocab_size),
        int(model_config.get("pad_vocab_size_to", 1) or 1),
    )

    attn_params = n_embd * (n_embd + 2 * kv_dim) + n_embd * n_embd
    hidden = _mlp_hidden_size(model_config, n_embd)
    mlp_type = str(model_config.get("mlp_type", "gelu")).lower()
    if mlp_type == "swiglu":
        mlp_params = n_embd * (2 * hidden) + hidden * n_embd
    else:
        mlp_params = n_embd * hidden + hidden * n_embd
    lm_head_params = n_embd * padded_vocab
    return int(n_layers * (attn_params + mlp_params) + lm_head_params)


def estimate_attention_flops_per_token_from_config(model_config: dict[str, Any]) -> int:
    n_layers = int(model_config["n_layers"])
    n_heads = int(model_config["n_heads"])
    n_embd = int(model_config["n_embd"])
    max_seq_len = int(model_config.get("max_seq_len", 0) or 0)
    if n_layers <= 0 or n_heads <= 0 or n_embd <= 0 or max_seq_len <= 0:
        return 0
    head_dim = n_embd // n_heads
    pattern = model_config.get("attention_window_pattern", "L")
    total = 0
    for layer_idx in range(n_layers):
        window = layer_window_size(pattern, layer_idx, n_layers, max_seq_len)
        effective_seq = max_seq_len if window is None else min(window, max_seq_len)
        total += 12 * n_heads * head_dim * effective_seq
    return int(total)


def apply_auto_scaling(
    config: dict[str, Any],
    vocab_size: int,
    world_size: int = 1,
) -> tuple[dict[str, Any], ScalingResult]:
    """Apply nanochat-style depth and schedule scaling to a config copy."""

    updated = copy.deepcopy(config)
    model_config = dict(updated.get("model", {}))
    training_config = dict(updated.get("training", {}))
    enabled = _is_enabled(model_config.get("auto_scale")) or _is_enabled(
        training_config.get("auto_scale")
    )
    if not enabled:
        return updated, ScalingResult(False, None, {}, {}, {})

    model_updates = derive_depth_model_config(model_config)
    model_config.update(model_updates)
    model_config["vocab_size"] = int(vocab_size)
    updated["model"] = model_config

    scaling_params = estimate_scaling_matmul_params(model_config, vocab_size=vocab_size)
    target_ratio = float(training_config.get("target_param_data_ratio", 0) or 0)
    target_tokens = None
    training_updates: dict[str, Any] = {}
    if target_ratio > 0:
        target_tokens = max(1, int(target_ratio * scaling_params))
        if training_config.get("max_train_tokens") in {None, ""}:
            training_updates["max_train_tokens"] = target_tokens

    auto_batch = _is_enabled(training_config.get("auto_batch_tokens"))
    if auto_batch and target_tokens is not None:
        reference_source = dict(model_config)
        reference_source.pop("n_kv_heads", None)
        reference_source["depth"] = 12
        reference_model = dict(model_config)
        reference_model.update(derive_depth_model_config(reference_source))
        reference_tokens = target_ratio * estimate_scaling_matmul_params(
            reference_model,
            vocab_size=vocab_size,
        )
        reference_batch_tokens = int(training_config.get("reference_batch_tokens", 2**19))
        predicted_batch_tokens = reference_batch_tokens * (
            target_tokens / max(1.0, reference_tokens)
        ) ** 0.383
        total_batch_tokens = _round_power_of_two(predicted_batch_tokens)
        micro_tokens = (
            int(training_config.get("batch_size", 1))
            * int(model_config.get("max_seq_len", 1))
            * max(1, int(world_size))
        )
        grad_accum_steps = max(1, round(total_batch_tokens / max(1, micro_tokens)))
        actual_batch_tokens = grad_accum_steps * max(1, micro_tokens)
        training_updates["grad_accum_steps"] = grad_accum_steps
        training_updates["total_batch_tokens"] = actual_batch_tokens

        if _is_enabled(training_config.get("scale_learning_rate", True)):
            batch_lr_scale = (actual_batch_tokens / reference_batch_tokens) ** 0.5
            for key in ("learning_rate", "matrix_lr", "embedding_lr", "lm_head_lr", "scalar_lr"):
                if training_config.get(key) not in {None, ""}:
                    training_updates[key] = float(training_config[key]) * batch_lr_scale
            training_updates["batch_lr_scale"] = batch_lr_scale

        if _is_enabled(training_config.get("scale_weight_decay", True)):
            base_weight_decay = float(training_config.get("weight_decay", 0.0) or 0.0)
            weight_decay_scale = (actual_batch_tokens / reference_batch_tokens) ** 0.5
            weight_decay_scale *= reference_tokens / max(1, target_tokens)
            training_updates["weight_decay"] = base_weight_decay * weight_decay_scale
            training_updates["weight_decay_scale"] = weight_decay_scale

    training_config.update(training_updates)
    updated["training"] = training_config
    derived = {
        "scaling_matmul_params": scaling_params,
        "target_param_data_ratio": target_ratio if target_ratio > 0 else None,
        "target_train_tokens": target_tokens,
        "attention_flops_per_token": estimate_attention_flops_per_token_from_config(
            model_config
        ),
    }
    return updated, ScalingResult(
        enabled=True,
        depth=int(model_config["n_layers"]),
        model_updates=model_updates,
        training_updates=training_updates,
        derived=derived,
    )
