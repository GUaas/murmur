from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any

import torch


_RUNTIME_OPTIMIZER_GROUP_KEYS = frozenset(
    {
        "initial_lr",
        "initial_weight_decay",
        "weight_decay",
        "parameter_role",
        "use_muon",
        "muon_momentum",
        "muon_ns_steps",
        "muon_update_scale",
        "muon_nesterov",
        "muon_orthogonalization",
        "muon_row_equilibration",
        "muon_renormalize",
    }
)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model
    while True:
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
            continue
        return unwrapped


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    batch_in_epoch: int = 0,
    best_val_loss: float | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_version": 2,
        "model_state": _unwrap_model(model).state_dict(),
        "epoch": epoch,
        "step": step,
        "batch_in_epoch": batch_in_epoch,
        "best_val_loss": best_val_loss,
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        torch.save(payload, temp_path)
        temp_path.replace(output_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise
    return output_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(checkpoint_path, **load_kwargs)
    restore_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        strict=strict,
    )
    return checkpoint


def restore_checkpoint(
    checkpoint: dict[str, Any],
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    strict: bool = True,
) -> None:
    """Restore an already-loaded checkpoint payload into runtime objects."""

    if model is not None:
        state = checkpoint.get("model_state", checkpoint)
        _unwrap_model(model).load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer_state" in checkpoint:
        runtime_group_values = [
            {
                key: group[key]
                for key in _RUNTIME_OPTIMIZER_GROUP_KEYS
                if key in group
            }
            for group in optimizer.param_groups
        ]
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if len(runtime_group_values) != len(optimizer.param_groups):
            raise ValueError(
                "Optimizer parameter-group count changed across full resume; "
                "use init_from instead."
            )
        # Optimizer checkpoints contain dynamic LR/WD values.  The strictly
        # validated current config remains authoritative for group roles and
        # base hyperparameters; the trainer reapplies the persisted schedule
        # before the next update.
        for group, runtime_values in zip(optimizer.param_groups, runtime_group_values):
            group.update(runtime_values)
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
