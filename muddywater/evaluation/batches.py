from __future__ import annotations

import torch


def batch_needs_document_mask(batch: dict[str, torch.Tensor]) -> bool:
    document_ids = batch.get("document_ids")
    document_mask_needed = batch.get("document_mask_needed")
    if isinstance(document_mask_needed, torch.Tensor):
        return bool(document_mask_needed.any().item())
    return document_ids is not None


def move_eval_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    """Move a language-model eval batch to a device."""

    moved: dict[str, torch.Tensor | None] = {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }
    attention_mask = batch.get("attention_mask")
    moved["attention_mask"] = (
        attention_mask.to(device, non_blocking=True)
        if attention_mask is not None
        else None
    )
    document_ids = batch.get("document_ids")
    moved["document_ids"] = (
        document_ids.to(device, non_blocking=True)
        if batch_needs_document_mask(batch) and document_ids is not None
        else None
    )
    return moved
