from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import default_collate


SEQUENCE_KEYS = {"input_ids", "labels", "document_ids"}


def empirical_unigram_ce(labels: torch.Tensor, ignore_index: int = -100) -> float:
    """Return the empirical unigram cross entropy of the valid labels."""
    flat_labels = labels.detach().reshape(-1)
    valid_labels = flat_labels[flat_labels != int(ignore_index)]
    if valid_labels.numel() == 0:
        raise ValueError("Cannot estimate unigram CE from a batch with no valid labels.")
    _, counts = torch.unique(valid_labels.to(device="cpu"), return_counts=True)
    probabilities = counts.to(torch.float64) / int(valid_labels.numel())
    return float(-(probabilities * probabilities.log()).sum().item())


def estimate_dataset_unigram_ce(
    dataset,
    max_samples: int | None = None,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Estimate label unigram CE from deterministic, evenly spaced indices.

    This reads dataset items directly instead of using a shuffled loader. It is
    reproducible, covers ordered/sharded datasets better than a prefix sample,
    and does not perturb training sampler state.
    """
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive when provided.")
    samples_checked = len(dataset)
    if max_samples is not None:
        samples_checked = min(samples_checked, int(max_samples))
    if samples_checked <= 0:
        raise ValueError("Cannot estimate unigram CE from an empty dataset.")

    if samples_checked == len(dataset):
        sample_indices = range(len(dataset))
        sampling_strategy = "full_dataset"
    else:
        sample_indices = [
            (index * len(dataset)) // samples_checked for index in range(samples_checked)
        ]
        sampling_strategy = "evenly_spaced"

    token_counts: dict[int, int] = {}
    valid_tokens = 0
    for index in sample_indices:
        labels = dataset[index]["labels"].detach().reshape(-1)
        labels = labels[labels != int(ignore_index)].to(device="cpu")
        if labels.numel() == 0:
            continue
        values, counts = torch.unique(labels, return_counts=True)
        valid_tokens += int(labels.numel())
        for value, count in zip(values.tolist(), counts.tolist()):
            token_id = int(value)
            token_counts[token_id] = token_counts.get(token_id, 0) + int(count)

    if valid_tokens <= 0:
        raise ValueError("Selected dataset samples contain no valid labels.")
    probabilities = torch.tensor(
        list(token_counts.values()),
        dtype=torch.float64,
    ) / valid_tokens
    ce = float(-(probabilities * probabilities.log()).sum().item())
    return {
        "ce": ce,
        "perplexity": math.exp(ce),
        "valid_tokens": valid_tokens,
        "unique_labels": len(token_counts),
        "samples_checked": samples_checked,
        "is_full_dataset": samples_checked == len(dataset),
        "sampling_strategy": sampling_strategy,
    }


def trim_batch_sequence(
    batch: Mapping[str, Any],
    max_seq_len: int | None,
) -> dict[str, Any]:
    """Clone a collated LM batch and optionally keep a prefix of its sequence."""
    if max_seq_len is not None and int(max_seq_len) <= 0:
        raise ValueError("max_seq_len must be positive when provided.")
    limit = None if max_seq_len is None else int(max_seq_len)
    trimmed: dict[str, Any] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            trimmed[key] = copy.deepcopy(value)
            continue
        tensor = value.detach().clone()
        if limit is None:
            trimmed[key] = tensor
        elif key in SEQUENCE_KEYS and tensor.dim() >= 2:
            trimmed[key] = tensor[:, :limit]
        elif key == "attention_mask" and tensor.dim() == 2:
            trimmed[key] = tensor[:, :limit]
        elif key == "attention_mask" and tensor.dim() in {3, 4}:
            trimmed[key] = tensor[..., :limit, :limit]
        else:
            trimmed[key] = tensor

    input_ids = trimmed.get("input_ids")
    labels = trimmed.get("labels")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("Batch must contain tensor input_ids and labels.")
    if input_ids.shape != labels.shape:
        raise ValueError(
            "input_ids and labels must have matching shapes; "
            f"got {tuple(input_ids.shape)} and {tuple(labels.shape)}."
        )
    return trimmed


def fixed_batch_from_dataset(
    dataset,
    batch_size: int = 1,
    start_index: int = 0,
    max_seq_len: int | None = None,
) -> dict[str, Any]:
    """Collate fixed, consecutive real dataset items for an overfit probe."""
    batch_size = int(batch_size)
    start_index = int(start_index)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    stop_index = start_index + batch_size
    if stop_index > len(dataset):
        raise ValueError(
            f"Requested dataset indices [{start_index}, {stop_index}) but dataset "
            f"contains only {len(dataset)} samples."
        )
    batch = default_collate([dataset[index] for index in range(start_index, stop_index)])
    return trim_batch_sequence(batch, max_seq_len=max_seq_len)


def gradient_l2_norm(model: torch.nn.Module) -> float:
    """Return the global L2 norm of all currently populated gradients."""
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().to(dtype=torch.float32)
        squared_norm += float(torch.sum(gradient * gradient).item())
    return math.sqrt(squared_norm)


def rng_fork_devices(device: torch.device) -> list[int]:
    """Return every CUDA device whose RNG may be touched by ``manual_seed``."""

    if device.type != "cuda":
        return []
    return list(range(torch.cuda.device_count()))


def seed_probe_rng(seed: int, device: torch.device) -> None:
    """Seed the probe while the caller owns a matching ``fork_rng`` context."""

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward_ce(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    ignore_index: int,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "labels": batch["labels"],
        "ignore_index": int(ignore_index),
        "z_loss_weight": 0.0,
        "return_logits": False,
    }
    for key in ("attention_mask", "document_ids"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            kwargs[key] = value
    outputs = model(batch["input_ids"], **kwargs)
    if not isinstance(outputs, Mapping):
        raise TypeError("Overfit diagnostic expects the model forward to return a mapping.")
    ce_loss = outputs.get("ce_loss")
    if ce_loss is None:
        ce_loss = outputs.get("loss")
    if not isinstance(ce_loss, torch.Tensor) or ce_loss.numel() != 1:
        raise ValueError("Model output must contain a scalar ce_loss or loss tensor.")
    if not bool(torch.isfinite(ce_loss).item()):
        raise FloatingPointError(f"Non-finite CE encountered: {float(ce_loss.detach().item())}")
    return ce_loss


def _crossed_threshold(initial: float, values: list[float], threshold: float) -> bool:
    return initial >= threshold and any(value < threshold for value in values[1:])


def run_fixed_batch_overfit(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
    ignore_index: int = -100,
    reference_unigram_ce: float | None = None,
    target_ce: float | None = None,
    grad_clip: float | None = None,
    device: str | torch.device | None = None,
    seed: int = 42,
    clone_model: bool = True,
) -> dict[str, Any]:
    """Overfit one immutable batch and return a JSON-serializable diagnosis.

    By default the model is deep-copied, so the caller's parameters, optimizer
    state, and train/eval mode are not changed. No checkpoints or report files
    are written by this function.
    """
    steps = int(steps)
    learning_rate = float(learning_rate)
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if grad_clip is not None and float(grad_clip) <= 0:
        raise ValueError("grad_clip must be positive when provided.")

    batch = trim_batch_sequence(batch, max_seq_len=None)
    valid_tokens = int((batch["labels"] != int(ignore_index)).sum().item())
    if valid_tokens <= 0:
        raise ValueError("Fixed batch contains no valid labels.")
    batch_unigram_ce = empirical_unigram_ce(batch["labels"], ignore_index=ignore_index)
    unigram_ce = (
        batch_unigram_ce if reference_unigram_ce is None else float(reference_unigram_ce)
    )
    if not math.isfinite(unigram_ce):
        raise ValueError("reference_unigram_ce must be finite when provided.")
    effective_target_ce = unigram_ce if target_ce is None else float(target_ce)
    if not math.isfinite(effective_target_ce):
        raise ValueError("target_ce must be finite when provided.")

    working_model = copy.deepcopy(model) if clone_model else model
    target_device = torch.device(device) if device is not None else _model_device(working_model)
    working_model.to(target_device)
    moved_batch = _move_batch(batch, target_device)
    original_training_mode = working_model.training
    # Exercise the real training-only forward path, including gradient checkpointing.
    # The RNG fork and seed keep configured dropout reproducible without disabling it.
    working_model.train()
    dropout_probabilities = sorted(
        {
            float(module.p)
            for module in working_model.modules()
            if isinstance(module, torch.nn.Dropout)
        }
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in working_model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(weight_decay),
    )

    ce_history: list[float] = []
    gradient_norm_history: list[float] = []
    fork_devices = rng_fork_devices(target_device)
    try:
        with torch.random.fork_rng(devices=fork_devices):
            seed_probe_rng(seed, target_device)

            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                ce_loss = _forward_ce(working_model, moved_batch, ignore_index=ignore_index)
                ce_history.append(float(ce_loss.detach().item()))
                ce_loss.backward()
                gradient_norm_history.append(gradient_l2_norm(working_model))
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(working_model.parameters(), float(grad_clip))
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            final_ce_loss = _forward_ce(working_model, moved_batch, ignore_index=ignore_index)
            final_ce = float(final_ce_loss.detach().item())
            final_ce_loss.backward()
            final_gradient_norm = gradient_l2_norm(working_model)
            optimizer.zero_grad(set_to_none=True)
    finally:
        working_model.train(original_training_mode)

    ce_history.append(final_ce)
    initial_ce = ce_history[0]
    minimum_ce = min(ce_history)
    target_source = "explicit" if target_ce is not None else "reference_unigram"
    return {
        "scope": "model_forward_backward_adamw_sanity",
        "formal_trainer_exercised": False,
        "not_exercised": [
            "distributed_data_parallel",
            "gradient_accumulation",
            "mixed_precision_and_grad_scaler",
            "trainer_scheduler_and_token_quota",
            "configured_non_adamw_optimizer",
        ],
        "steps": steps,
        "optimizer": "adamw",
        "learning_rate": learning_rate,
        "weight_decay": float(weight_decay),
        "betas": [float(betas[0]), float(betas[1])],
        "grad_clip": None if grad_clip is None else float(grad_clip),
        "seed": int(seed),
        "device": str(target_device),
        "model_was_cloned": bool(clone_model),
        "writes_files": False,
        "training_mode_enabled": True,
        "dropout_disabled": False,
        "dropout_probabilities": dropout_probabilities,
        "batch_size": int(batch["input_ids"].size(0)),
        "sequence_length": int(batch["input_ids"].size(1)),
        "valid_tokens": valid_tokens,
        "initial_ce": initial_ce,
        "final_ce": final_ce,
        "minimum_ce": minimum_ce,
        "absolute_ce_drop": initial_ce - final_ce,
        "relative_ce_drop": (initial_ce - final_ce) / max(abs(initial_ce), 1e-12),
        "ce_history": ce_history,
        "batch_unigram_ce": batch_unigram_ce,
        "reference_unigram_ce": unigram_ce,
        "below_reference_unigram_initial": initial_ce < unigram_ce,
        "below_reference_unigram_final": final_ce < unigram_ce,
        "crossed_reference_unigram": _crossed_threshold(
            initial_ce,
            ce_history,
            unigram_ce,
        ),
        "target_ce": effective_target_ce,
        "target_source": target_source,
        "below_target_final": final_ce < effective_target_ce,
        "crossed_target": _crossed_threshold(
            initial_ce,
            ce_history,
            effective_target_ce,
        ),
        "passed": final_ce < effective_target_ce,
        "initial_gradient_norm": gradient_norm_history[0],
        "final_gradient_norm": final_gradient_norm,
        "minimum_step_gradient_norm": min(gradient_norm_history),
        "maximum_step_gradient_norm": max(gradient_norm_history),
        "gradient_norm_history": gradient_norm_history,
    }
