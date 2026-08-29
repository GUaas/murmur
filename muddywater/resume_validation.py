from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


# Gradient checkpointing changes memory/computation strategy, not model behavior.
# Training schedule fields live outside ``model`` and are intentionally ignored.
RESUME_MUTABLE_MODEL_FIELDS = frozenset({"gradient_checkpointing", "loss_chunk_size"})
RESUME_MUTABLE_TRAINING_FIELDS = frozenset({"max_train_tokens"})

RESUME_TRAINING_DEFAULTS: dict[str, Any] = {
    "optimizer": "adamw",
    "learning_rate": 3e-4,
    "min_lr": None,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "batch_size": 8,
    "grad_accum_steps": 1,
    "precision": "auto",
    "amp": True,
    "ignore_index": -100,
    "z_loss_weight": 0.0,
    "warmup_steps": 0,
    "warmup_ratio": None,
    "warmup_tokens": None,
    "lr_decay": True,
    "grad_clip": 1.0,
    "strict_load": True,
    "max_train_tokens": None,
    "embedding_lr": None,
    "value_embedding_lr": None,
    "lm_head_lr": None,
    "matrix_lr": None,
    "scalar_lr": None,
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "muon_update_scale": 1.0,
    "muon_nesterov": True,
    "muon_orthogonalization": "polar_express",
    "muon_row_equilibration": True,
    "muon_renormalize": True,
    "muon_momentum_schedule": None,
    "muon_weight_decay_schedule": None,
    "muon_momentum_warmup_steps": 400,
    "muon_momentum_start": 0.85,
    "muon_momentum_peak": 0.97,
    "muon_momentum_final": 0.90,
    "warmdown_ratio": 0.65,
}


def _config_mapping(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return dict(asdict(config))
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"Unsupported model config type: {type(config).__name__}")


def _normalized(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_normalized(item) for item in value)
    return value


def _training_value(config: Mapping[str, Any], field: str) -> Any:
    if field == "min_lr":
        value = config.get(field)
        if value is None:
            value = float(config.get("learning_rate", RESUME_TRAINING_DEFAULTS["learning_rate"])) * 0.1
        return float(value)
    if field in {"muon_momentum_schedule", "muon_weight_decay_schedule"} and field not in config:
        optimizer = str(config.get("optimizer", "adamw")).strip().lower()
        return optimizer in {"muon", "muon_adamw", "hybrid_muon"}
    value = config.get(field, RESUME_TRAINING_DEFAULTS[field])
    if field in {"optimizer", "precision", "muon_orthogonalization"} and value is not None:
        return str(value).strip().lower()
    return _normalized(value)


def _resolve_checkpoint_model_config(raw_config: Mapping[str, Any], current_model: Any) -> dict[str, Any]:
    current_config = getattr(current_model, "config", None)
    if current_config is None:
        raise ValueError("Current model does not expose a config for resume validation.")
    config_type = type(current_config)
    from_dict = getattr(config_type, "from_dict", None)
    if callable(from_dict):
        try:
            return _config_mapping(from_dict(dict(raw_config)))
        except Exception as exc:
            raise ValueError(f"Could not resolve checkpoint model config: {exc}") from exc
    return dict(raw_config)


def validate_resume_model_config(checkpoint: Mapping[str, Any], current_model: Any) -> None:
    """Reject resumes that would silently change model behavior.

    The checkpoint's raw model mapping is resolved through the current config
    class so older checkpoints that omitted then-defaulted fields remain usable.
    Only model fields are compared; training limits and schedules may change.
    """

    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError(
            "Resume checkpoint has no saved config; use init_from for a weights-only "
            "load or resume from a checkpoint created by this trainer."
        )
    raw_model_config = checkpoint_config.get("model")
    if not isinstance(raw_model_config, Mapping):
        raise ValueError(
            "Resume checkpoint has no saved model config; refusing an unsafe resume. "
            "Use init_from for a weights-only load instead."
        )

    current_config = _config_mapping(getattr(current_model, "config", None))
    checkpoint_model_config = _resolve_checkpoint_model_config(raw_model_config, current_model)
    compared_fields = sorted(
        (set(current_config) | set(checkpoint_model_config)) - RESUME_MUTABLE_MODEL_FIELDS
    )
    mismatches: list[str] = []
    for field in compared_fields:
        checkpoint_value = checkpoint_model_config.get(field, "<missing>")
        current_value = current_config.get(field, "<missing>")
        if _normalized(checkpoint_value) != _normalized(current_value):
            mismatches.append(
                f"{field}: checkpoint={checkpoint_value!r}, current={current_value!r}"
            )

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Resume checkpoint model config is incompatible with the current model: "
            f"{details}. Training limits such as max_epochs/max_steps may change, "
            "but model behavior fields must remain identical."
        )


def validate_resume_training_state(
    checkpoint: Mapping[str, Any],
    current_training_config: Mapping[str, Any],
    *,
    world_size: int,
    precision_name: str,
    seed: int = 42,
    current_run_identity: Mapping[str, Any] | None = None,
) -> None:
    """Validate optimizer state, update semantics, precision, and topology."""

    if "model_state" not in checkpoint:
        raise ValueError("Resume checkpoint has no model_state; use init_from for weights-only loading.")
    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError(
            "Resume checkpoint has no optimizer_state; use init_from for a weights-only load."
        )
    if int(checkpoint.get("step", 0) or 0) > 0 and not optimizer_state.get("state"):
        raise ValueError(
            "Resume checkpoint optimizer_state is empty after step 0; use init_from instead."
        )
    effective_precision = str(precision_name).strip().lower()
    if effective_precision == "fp16" and not checkpoint.get("scaler_state"):
        raise ValueError(
            "FP16 resume checkpoint has no scaler_state; use init_from or a full checkpoint."
        )

    checkpoint_config = checkpoint.get("config")
    checkpoint_training = (
        checkpoint_config.get("training") if isinstance(checkpoint_config, Mapping) else None
    )
    if not isinstance(checkpoint_training, Mapping):
        raise ValueError("Resume checkpoint has no saved training config; refusing unsafe resume.")
    if not bool(current_training_config.get("strict_load", True)):
        raise ValueError("Full resume requires training.strict_load=true; use init_from otherwise.")
    mismatches: list[str] = []
    for field in RESUME_TRAINING_DEFAULTS:
        if field in RESUME_MUTABLE_TRAINING_FIELDS:
            continue
        checkpoint_value = _training_value(checkpoint_training, field)
        current_value = _training_value(current_training_config, field)
        if checkpoint_value != current_value:
            mismatches.append(
                f"{field}: checkpoint={checkpoint_value!r}, current={current_value!r}"
            )

    extra = checkpoint.get("extra")
    runtime = extra.get("resume_runtime") if isinstance(extra, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError(
            "Resume checkpoint lacks resume_runtime topology metadata; use init_from or "
            "a checkpoint created by the updated trainer."
        )
    checkpoint_world_size = int(runtime.get("world_size", 0) or 0)
    checkpoint_precision = str(runtime.get("precision", "")).strip().lower()
    checkpoint_seed = int(checkpoint_config.get("seed", 42))
    if checkpoint_world_size != int(world_size):
        mismatches.append(
            f"world_size: checkpoint={checkpoint_world_size}, current={int(world_size)}"
        )
    if checkpoint_precision != effective_precision:
        mismatches.append(
            f"effective_precision: checkpoint={checkpoint_precision!r}, "
            f"current={effective_precision!r}"
        )
    if checkpoint_seed != int(seed):
        mismatches.append(f"seed: checkpoint={checkpoint_seed}, current={int(seed)}")
    rng_states = extra.get("rng_states_by_rank") if isinstance(extra, Mapping) else None
    if not isinstance(rng_states, list) or len(rng_states) != checkpoint_world_size:
        mismatches.append(
            "rng_states_by_rank: checkpoint does not contain one RNG state per saved rank"
        )

    if current_run_identity is not None:
        saved_identity = extra.get("run_identity") if isinstance(extra, Mapping) else None
        if not isinstance(saved_identity, Mapping):
            mismatches.append(
                "run_identity: checkpoint has no tokenizer/data identity; use init_from "
                "for this legacy or weights-only checkpoint"
            )
        else:
            _append_run_identity_mismatches(
                mismatches,
                checkpoint_identity=saved_identity,
                current_identity=current_run_identity,
            )

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Resume checkpoint training state is incompatible with the current run: "
            f"{details}. max_epochs/max_steps/max_train_tokens and "
            "logging/evaluation/save settings may "
            "change, but optimizer, batch, precision, loss, warmup, and topology settings "
            "must remain identical."
        )


def _append_run_identity_mismatches(
    mismatches: list[str],
    *,
    checkpoint_identity: Mapping[str, Any],
    current_identity: Mapping[str, Any],
) -> None:
    checkpoint_tokenizer = checkpoint_identity.get("tokenizer")
    current_tokenizer = current_identity.get("tokenizer")
    checkpoint_sha = (
        str(checkpoint_tokenizer.get("sha256") or "").strip().lower()
        if isinstance(checkpoint_tokenizer, Mapping)
        else ""
    )
    current_sha = (
        str(current_tokenizer.get("sha256") or "").strip().lower()
        if isinstance(current_tokenizer, Mapping)
        else ""
    )
    if checkpoint_sha != current_sha:
        mismatches.append(
            "tokenizer_sha256: "
            f"checkpoint={checkpoint_sha or '<missing>'!r}, "
            f"current={current_sha or '<missing>'!r}; changing token IDs requires init_from"
        )

    checkpoint_data = checkpoint_identity.get("data")
    current_data = current_identity.get("data")
    if not isinstance(checkpoint_data, Mapping) or not isinstance(current_data, Mapping):
        mismatches.append("data_identity: missing checkpoint or current data identity")
        return

    checkpoint_length = (
        checkpoint_data.get("dataset", {}).get("length")
        if isinstance(checkpoint_data.get("dataset"), Mapping)
        else None
    )
    current_length = (
        current_data.get("dataset", {}).get("length")
        if isinstance(current_data.get("dataset"), Mapping)
        else None
    )
    if checkpoint_length != current_length:
        mismatches.append(
            "dataset_length: "
            f"checkpoint={checkpoint_length!r}, current={current_length!r}"
        )

    if _canonical_data_identity(checkpoint_data) != _canonical_data_identity(current_data):
        mismatches.append(
            "data_fingerprint: training data/cache identity changed; use init_from "
            "instead of full resume"
        )


def _canonical_data_identity(value: Any) -> Any:
    """Normalize identity metadata before a resume comparison.

    A resource with a cryptographic content hash is unchanged when only its
    filesystem timestamp changes (for example after a deterministic corpus
    generator is rerun). Large unhashed cache shards still retain mtime in the
    comparison, while their manifests and metadata remain content-addressed.
    Stored aggregate fingerprints are derived fields and are recomputed by
    structural comparison here for backward compatibility.
    """

    if isinstance(value, Mapping):
        has_content_hash = bool(str(value.get("sha256") or "").strip())
        return {
            str(key): _canonical_data_identity(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key != "fingerprint" and not (key == "mtime_ns" and has_content_hash)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_data_identity(item) for item in value]
    return value
