from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class LanguageModelLoss:
    """Token-summed language-model losses and their normalization count."""

    loss: torch.Tensor
    ce_loss: torch.Tensor
    loss_sum: torch.Tensor
    ce_loss_sum: torch.Tensor
    token_count: torch.Tensor
    z_loss_sum: torch.Tensor | None


def apply_logit_softcap(
    logits: torch.Tensor,
    softcap: float | None,
) -> torch.Tensor:
    """Apply a smooth logit bound in fp32 when a positive cap is configured."""

    if softcap is None or float(softcap) <= 0:
        return logits
    cap = float(softcap)
    return cap * torch.tanh(logits.float() / cap)


def language_model_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
    z_loss_weight: float = 0.0,
) -> LanguageModelLoss:
    """Compute token-summed CE and optional z-loss from materialized logits."""

    flat_labels = labels.reshape(-1)
    flat_logits = logits.reshape(-1, logits.size(-1))
    token_count = (flat_labels != int(ignore_index)).sum()
    ce_loss_sum = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=int(ignore_index),
        reduction="sum",
    )
    z_loss_sum = _z_loss_sum(
        flat_logits,
        flat_labels,
        ignore_index=int(ignore_index),
        z_loss_weight=float(z_loss_weight),
    )
    loss_sum = ce_loss_sum if z_loss_sum is None else ce_loss_sum + z_loss_sum.type_as(ce_loss_sum)
    denominator = token_count.clamp_min(1)
    return LanguageModelLoss(
        loss=loss_sum / denominator,
        ce_loss=ce_loss_sum / denominator,
        loss_sum=loss_sum,
        ce_loss_sum=ce_loss_sum,
        token_count=token_count,
        z_loss_sum=z_loss_sum,
    )


def chunked_language_model_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    *,
    vocab_size: int,
    ignore_index: int,
    logit_softcap: float | None,
    z_loss_weight: float = 0.0,
    chunk_size: int = 256,
) -> LanguageModelLoss:
    """Project hidden states and compute LM loss without materializing all logits.

    Chunking over tokens bounds peak memory at roughly
    ``chunk_size * vocab_size`` logits.  The returned objective is identical to
    the full-logit formulation up to floating-point summation order.
    """

    if hidden.shape[:-1] != labels.shape:
        raise ValueError(
            "hidden leading dimensions must match labels; "
            f"got {tuple(hidden.shape[:-1])} and {tuple(labels.shape)}"
        )
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_labels = labels.reshape(-1)
    tokens_per_chunk = int(chunk_size)
    if tokens_per_chunk <= 0:
        tokens_per_chunk = max(1, int(flat_hidden.size(0)))

    ce_parts: list[torch.Tensor] = []
    z_parts: list[torch.Tensor] = []
    token_count = (flat_labels != int(ignore_index)).sum()
    for start in range(0, int(flat_hidden.size(0)), tokens_per_chunk):
        end = min(start + tokens_per_chunk, int(flat_hidden.size(0)))
        chunk_labels = flat_labels[start:end]
        chunk_args = (
            lm_head,
            flat_hidden[start:end],
            chunk_labels,
            int(vocab_size),
            int(ignore_index),
            logit_softcap,
            float(z_loss_weight),
        )
        if torch.is_grad_enabled():
            chunk_ce, chunk_z = checkpoint(
                _projected_chunk_loss_sums,
                *chunk_args,
                use_reentrant=False,
            )
        else:
            chunk_ce, chunk_z = _projected_chunk_loss_sums(*chunk_args)
        ce_parts.append(chunk_ce)
        if z_loss_weight > 0:
            z_parts.append(chunk_z)

    if not ce_parts:
        raise ValueError("Cannot compute language-model loss for an empty token sequence")
    ce_loss_sum = torch.stack(ce_parts).sum()
    z_loss_sum = torch.stack(z_parts).sum() if z_parts else None
    loss_sum = ce_loss_sum if z_loss_sum is None else ce_loss_sum + z_loss_sum.type_as(ce_loss_sum)
    denominator = token_count.clamp_min(1)
    return LanguageModelLoss(
        loss=loss_sum / denominator,
        ce_loss=ce_loss_sum / denominator,
        loss_sum=loss_sum,
        ce_loss_sum=ce_loss_sum,
        token_count=token_count,
        z_loss_sum=z_loss_sum,
    )


def _projected_chunk_loss_sums(
    lm_head: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
    logit_softcap: float | None,
    z_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one token chunk; safe to recompute during checkpoint backward."""

    logits = lm_head(hidden)
    if logits.size(-1) != int(vocab_size):
        logits = logits[..., : int(vocab_size)]
    logits = apply_logit_softcap(logits, logit_softcap)
    ce_loss_sum = F.cross_entropy(
        logits,
        labels,
        ignore_index=int(ignore_index),
        reduction="sum",
    )
    z_loss_sum = _z_loss_sum(
        logits,
        labels,
        ignore_index=int(ignore_index),
        z_loss_weight=float(z_loss_weight),
    )
    if z_loss_sum is None:
        z_loss_sum = logits.new_zeros((), dtype=torch.float32)
    return ce_loss_sum, z_loss_sum


def _z_loss_sum(
    flat_logits: torch.Tensor,
    flat_labels: torch.Tensor,
    *,
    ignore_index: int,
    z_loss_weight: float,
) -> torch.Tensor | None:
    if z_loss_weight <= 0:
        return None
    valid = flat_labels != int(ignore_index)
    log_z = torch.logsumexp(flat_logits.float(), dim=-1)
    return float(z_loss_weight) * log_z[valid].pow(2).sum()
