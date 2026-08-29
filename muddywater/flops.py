from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .model_extras import layer_window_size


@dataclass(frozen=True)
class ParameterSummary:
    total_params: int
    trainable_params: int
    embedding_params: int
    non_embedding_params: int
    matmul_params: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FlopEstimate:
    flops_per_token: int
    matmul_flops_per_token: int
    attention_flops_per_token: int
    total_training_flops: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _unique_parameter_count(parameters) -> int:
    seen: set[int] = set()
    total = 0
    for param in parameters:
        identity = id(param)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(param.numel())
    return total


def _is_embedding_parameter(name: str) -> bool:
    return any(token in name for token in ("transformer.wte", "transformer.wpe"))


def summarize_parameters(model: torch.nn.Module) -> ParameterSummary:
    """Summarize unique trainable parameters and matmul weights for GPT-style models."""

    named_params = list(model.named_parameters())
    total_params = _unique_parameter_count(param for _, param in named_params)
    trainable_params = _unique_parameter_count(param for _, param in named_params if param.requires_grad)
    embedding_params = _unique_parameter_count(
        param for name, param in named_params if _is_embedding_parameter(name)
    )
    matmul_params = sum(
        int(module.weight.numel())
        for module in model.modules()
        if isinstance(module, nn.Linear)
    )
    return ParameterSummary(
        total_params=total_params,
        trainable_params=trainable_params,
        embedding_params=embedding_params,
        non_embedding_params=max(0, total_params - embedding_params),
        matmul_params=matmul_params,
    )


def estimate_attention_flops_per_token(model: torch.nn.Module) -> int:
    """Estimate causal-attention train FLOPs/token from the model config."""

    config = getattr(model, "config", None)
    if config is None:
        return 0
    n_layers = int(getattr(config, "n_layers", 0) or getattr(config, "n_layer", 0) or 0)
    n_heads = int(getattr(config, "n_heads", 0) or getattr(config, "n_head", 0) or 0)
    n_embd = int(getattr(config, "n_embd", 0) or 0)
    max_seq_len = int(getattr(config, "max_seq_len", 0) or getattr(config, "sequence_len", 0) or 0)
    if n_layers <= 0 or n_heads <= 0 or n_embd <= 0 or max_seq_len <= 0:
        return 0
    head_dim = n_embd // n_heads
    pattern = getattr(config, "attention_window_pattern", "L")
    total = 0
    for layer_idx in range(n_layers):
        window = layer_window_size(pattern, layer_idx, n_layers, max_seq_len)
        effective_seq = max_seq_len if window is None else min(window, max_seq_len)
        total += 12 * n_heads * head_dim * effective_seq
    return int(total)


def estimate_training_flops(model: torch.nn.Module, token_count: int) -> FlopEstimate:
    """Estimate training FLOPs for already-consumed tokens."""

    summary = summarize_parameters(model)
    matmul_flops = int(6 * summary.matmul_params)
    attention_flops = estimate_attention_flops_per_token(model)
    flops_per_token = int(matmul_flops + attention_flops)
    return FlopEstimate(
        flops_per_token=flops_per_token,
        matmul_flops_per_token=matmul_flops,
        attention_flops_per_token=attention_flops,
        total_training_flops=int(flops_per_token * max(0, int(token_count))),
    )
