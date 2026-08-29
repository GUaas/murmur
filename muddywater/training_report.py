from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .distributed import unwrap_model
from .flops import estimate_training_flops, summarize_parameters
from .metrics import safe_rate
from .utils import atomic_write_text


def build_training_summary(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    elapsed_seconds: float,
    global_step: int,
    seen_train_tokens: int,
    best_val_loss: float,
    final_metrics: dict[str, Any] | None,
    device: torch.device,
    precision_name: str,
    world_size: int,
    total_steps: int,
    warmup_steps: int,
    tokens_per_step_estimate: int,
    session_start_global_step: int = 0,
    session_start_seen_train_tokens: int = 0,
    processed_train_tokens: int | None = None,
    session_start_processed_train_tokens: int = 0,
    attempted_optimizer_steps: int | None = None,
    overflow_skip_count: int = 0,
    nonfinite_grad_skip_count: int = 0,
    schedule_total_steps: int | None = None,
    schedule_total_tokens: int | None = None,
    status: str = "completed",
    termination_reason: str = "max_epochs_reached",
    target_reached: bool = True,
    loss_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable summary for one training invocation."""

    status = str(status).strip().lower()
    if status not in {"completed", "paused", "failed"}:
        raise ValueError(
            "training summary status must be 'completed', 'paused', or 'failed'"
        )
    if status == "completed" and not target_reached:
        raise ValueError("a completed training summary must have target_reached=true")
    if status == "paused" and target_reached:
        raise ValueError("a paused training summary must have target_reached=false")

    unwrapped_model = unwrap_model(model)
    param_summary = summarize_parameters(unwrapped_model)
    elapsed_seconds = float(elapsed_seconds)
    processed_train_tokens = int(
        seen_train_tokens if processed_train_tokens is None else processed_train_tokens
    )
    session_steps = max(0, int(global_step) - int(session_start_global_step))
    session_supervised_tokens = max(
        0, int(seen_train_tokens) - int(session_start_seen_train_tokens)
    )
    session_processed_tokens = max(
        0,
        int(processed_train_tokens) - int(session_start_processed_train_tokens),
    )
    cumulative_flops = estimate_training_flops(
        unwrapped_model,
        token_count=processed_train_tokens,
    )
    session_flops = estimate_training_flops(
        unwrapped_model,
        token_count=session_processed_tokens,
    )
    summary: dict[str, Any] = {
        "status": status,
        "termination_reason": str(termination_reason),
        "target_reached": bool(target_reached),
        "global_step": int(global_step),
        "supervised_train_tokens": int(seen_train_tokens),
        "processed_train_tokens": int(processed_train_tokens),
        # Backward-compatible alias with its semantics made explicit.
        "seen_train_tokens": int(seen_train_tokens),
        "seen_train_tokens_semantics": "supervised_non_ignore_labels_on_applied_steps",
        "elapsed_seconds": elapsed_seconds,
        "throughput": {
            "processed_tokens_per_second": safe_rate(
                session_processed_tokens, elapsed_seconds
            ),
            "supervised_tokens_per_second": safe_rate(
                session_supervised_tokens, elapsed_seconds
            ),
            "tokens_per_second": safe_rate(session_supervised_tokens, elapsed_seconds),
            "tokens_per_second_semantics": "supervised_tokens_per_second",
            "steps_per_second": safe_rate(session_steps, elapsed_seconds),
        },
        "session": {
            "start_global_step": int(session_start_global_step),
            "start_seen_train_tokens": int(session_start_seen_train_tokens),
            "start_processed_train_tokens": int(session_start_processed_train_tokens),
            "global_steps": session_steps,
            "supervised_train_tokens": session_supervised_tokens,
            "processed_train_tokens": session_processed_tokens,
            "seen_train_tokens": session_supervised_tokens,
            "elapsed_seconds": elapsed_seconds,
            "flops": session_flops.to_dict(),
        },
        "cumulative": {
            "global_step": int(global_step),
            "supervised_train_tokens": int(seen_train_tokens),
            "processed_train_tokens": int(processed_train_tokens),
            "seen_train_tokens": int(seen_train_tokens),
            "flops": cumulative_flops.to_dict(),
        },
        "schedule": {
            "training_limit_total_steps": int(total_steps),
            "total_steps": int(
                total_steps if schedule_total_steps is None else schedule_total_steps
            ),
            "total_tokens": (
                None if schedule_total_tokens is None else int(schedule_total_tokens)
            ),
            "extension_policy": "preserve_original_horizon",
            "warmup_steps": int(warmup_steps),
            "tokens_per_step_estimate": int(tokens_per_step_estimate),
        },
        "stability": {
            "attempted_optimizer_steps": int(
                global_step if attempted_optimizer_steps is None else attempted_optimizer_steps
            ),
            "applied_optimizer_steps": int(global_step),
            "amp_overflow_skips": int(overflow_skip_count),
            "nonfinite_gradient_skips": int(nonfinite_grad_skip_count),
        },
        "runtime": {
            "device": str(device),
            "precision": precision_name,
            "world_size": int(world_size),
        },
        "model": {
            "config": dict(config.get("model", {})),
            "parameters": param_summary.to_dict(),
        },
        # Backward-compatible alias for cumulative FLOPs.
        "flops": cumulative_flops.to_dict(),
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "final_metrics": final_metrics,
        "loss_guard": loss_guard,
    }
    return summary


def write_training_summary(
    output_dir: str | Path,
    summary: dict[str, Any],
    *,
    overwrite: bool = True,
) -> Path:
    """Write training_summary.json and return its path."""

    path = Path(output_dir) / "training_summary.json"
    content = json.dumps(summary, ensure_ascii=False, indent=2)
    atomic_write_text(path, content, overwrite=overwrite)
    return path
