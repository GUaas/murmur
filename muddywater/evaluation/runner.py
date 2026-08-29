from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader

from muddywater.distributed import all_reduce_sum, unwrap_model

from .batches import move_eval_batch
from .metrics import EvaluationMetrics


AutocastFactory = Callable[[], object]


def _output_loss_sum(
    outputs: dict[str, object],
    labels: torch.Tensor,
    ignore_index: int,
) -> tuple[float, int]:
    output_loss_sum = outputs.get("ce_loss_sum", outputs.get("loss_sum"))
    output_token_count = outputs.get("loss_token_count")
    if isinstance(output_loss_sum, torch.Tensor) and isinstance(output_token_count, torch.Tensor):
        valid_tokens = int(output_token_count.item())
        if valid_tokens <= 0:
            return 0.0, 0
        return float(output_loss_sum.item()), valid_tokens

    loss = outputs.get("loss")
    if not isinstance(loss, torch.Tensor):
        return 0.0, 0
    valid_tokens = int((labels != ignore_index).sum().item())
    if valid_tokens <= 0:
        return 0.0, 0
    return float(loss.item()) * valid_tokens, valid_tokens


def _target_byte_count(
    labels: torch.Tensor,
    ignore_index: int,
    tokenizer: Any | None,
) -> int | None:
    if tokenizer is None:
        return None
    valid = labels.detach()[labels != ignore_index]
    if valid.numel() == 0:
        return 0
    token_ids = [int(token_id) for token_id in valid.cpu().tolist()]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return len(text.encode("utf-8"))


@torch.no_grad()
def evaluate_language_model(
    model: torch.nn.Module,
    data_loader: DataLoader | None,
    device: torch.device,
    ignore_index: int = -100,
    max_batches: int | None = None,
    autocast_factory: AutocastFactory | None = None,
    reduce_distributed: bool = True,
    tokenizer: Any | None = None,
) -> EvaluationMetrics:
    """Evaluate a decoder-only LM with token-weighted cross entropy."""

    if data_loader is None:
        return EvaluationMetrics.empty()

    autocast_factory = autocast_factory or nullcontext
    eval_model = unwrap_model(model)
    was_training = model.training
    eval_model.eval()

    loss_sum = 0.0
    token_count = 0
    byte_count = 0
    has_byte_count = tokenizer is not None
    batch_count = 0
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        moved = move_eval_batch(batch, device=device)
        input_ids = moved["input_ids"]
        labels = moved["labels"]
        attention_mask = moved["attention_mask"]
        document_ids = moved["document_ids"]
        if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("Evaluation batches must contain tensor input_ids and labels.")

        with autocast_factory():
            outputs = eval_model(
                input_ids,
                labels=labels,
                attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                document_ids=document_ids if isinstance(document_ids, torch.Tensor) else None,
                ignore_index=ignore_index,
                z_loss_weight=0.0,
                return_logits=False,
            )

        batch_loss_sum, valid_tokens = _output_loss_sum(
            outputs=outputs,
            labels=labels,
            ignore_index=ignore_index,
        )
        if valid_tokens > 0:
            loss_sum += batch_loss_sum
            token_count += valid_tokens
            batch_bytes = _target_byte_count(
                labels=labels,
                ignore_index=ignore_index,
                tokenizer=tokenizer,
            )
            if batch_bytes is not None:
                byte_count += batch_bytes
            batch_count += 1

    if reduce_distributed:
        totals = torch.tensor(
            [loss_sum, float(token_count), float(byte_count), float(batch_count)],
            dtype=torch.float64,
            device=device,
        )
        totals = all_reduce_sum(totals)
        loss_sum = float(totals[0].item())
        token_count = int(totals[1].item())
        byte_count = int(totals[2].item())
        batch_count = int(totals[3].item())

    if was_training:
        model.train()
    return EvaluationMetrics.from_totals(
        loss_sum=loss_sum,
        token_count=token_count,
        byte_count=byte_count if has_byte_count else None,
        batch_count=batch_count,
    )
