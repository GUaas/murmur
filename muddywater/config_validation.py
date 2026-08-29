from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROOT_KEYS = frozenset(
    {
        "seed",
        "device",
        "strict_config",
        "tokenizer",
        "model",
        "data",
        "training",
        "generation",
        "checkpoint",
        "auto_scale",
        "__config_path__",
    }
)

TOKENIZER_KEYS = frozenset(
    {
        "path",
        "model_type",
        "vocab_size",
        "byte_fallback",
        "split_digits",
        "character_coverage",
        "max_sentencepiece_length",
        "max_sentence_length",
        "hard_vocab_limit",
        "normalization_rule_name",
        "remove_extra_whitespaces",
        "user_defined_symbols",
        "sample_size",
        "diagnostic_size",
        "diagnostic_paths",
        "progress",
        "num_threads",
        "pre_tokenizer",
        "min_frequency",
        "sha256",
        "tokenizer_sha256",
    }
)

DATA_KEYS = frozenset(
    {
        "train_paths",
        "val_paths",
        "jsonl_format",
        "jsonl_text_key",
        "instruction_key",
        "input_key",
        "output_key",
        "source_key",
        "target_key",
        "target_separator",
        "source_label",
        "target_label",
        "chat_template",
        "system_prompt",
        "instruction_template",
        "instruction_template_no_input",
        "txt_split",
        "min_chars",
        "validation_split",
        "shuffle_split",
        "split_mode",
        "train_on_inputs",
        "token_cache_dir",
        "train_token_file",
        "train_token_files",
        "val_token_file",
        "val_token_files",
        "add_bos",
        "stride",
        "tail_min_gap_ratio",
        "strict_tokenizer_match",
        "document_boundary_policy",
        "document_attention",
        "ignore_cross_document_targets",
        "diagnostic_samples",
        "label_mask_policy",
        "pack_sequences",
    }
)

TRAINING_KEYS = frozenset(
    {
        "output_dir",
        "optimizer",
        "batch_size",
        "grad_accum_steps",
        "eval_batch_size",
        "num_workers",
        "eval_num_workers",
        "max_epochs",
        "max_steps",
        "max_train_tokens",
        "session_max_seconds",
        "learning_rate",
        "min_lr",
        "weight_decay",
        "betas",
        "warmup_steps",
        "warmup_ratio",
        "warmup_tokens",
        "lr_decay",
        "grad_clip",
        "precision",
        "amp",
        "compile",
        "log_interval",
        "log_rank_files",
        "eval_interval",
        "eval_batches",
        "eval_shuffle",
        "eval_resample_each_eval",
        "eval_sampler_epoch",
        "save_interval",
        "save_best",
        "final_eval",
        "initial_eval",
        "ignore_index",
        "z_loss_weight",
        "resume_from",
        "init_from",
        "strict_load",
        "write_diagnostics",
        "write_run_manifest",
        "write_training_summary",
        "diagnostic_samples",
        "dense_mask_warning_bytes",
        "embedding_lr",
        "value_embedding_lr",
        "lm_head_lr",
        "matrix_lr",
        "scalar_lr",
        "muon_momentum",
        "muon_ns_steps",
        "muon_update_scale",
        "muon_nesterov",
        "muon_orthogonalization",
        "muon_row_equilibration",
        "muon_renormalize",
        "muon_momentum_schedule",
        "muon_weight_decay_schedule",
        "muon_momentum_warmup_steps",
        "muon_momentum_start",
        "muon_momentum_peak",
        "muon_momentum_final",
        "warmdown_ratio",
        "auto_scale",
        "auto_batch_tokens",
        "reference_batch_tokens",
        "target_param_data_ratio",
        "scale_learning_rate",
        "scale_weight_decay",
        "loss_guard",
    }
)

LOSS_GUARD_KEYS = frozenset(
    {
        "enabled",
        "min_steps",
        "baseline_window",
        "recent_window",
        "min_train_ce_drop",
        "max_evals_without_improvement",
        "min_val_improvement",
    }
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(str(key) for key in mapping if str(key) not in allowed)
    if unknown:
        raise ValueError(
            f"Unknown {path} field(s): {', '.join(repr(key) for key in unknown)}. "
            "Strict configs reject misspelled or unsupported options."
        )


def _positive_int(mapping: Mapping[str, Any], key: str, path: str, *, optional: bool = False) -> None:
    value = mapping.get(key)
    if value is None and optional:
        return
    if value is None or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{path}.{key} must be a positive integer")


def _non_negative_int(mapping: Mapping[str, Any], key: str, path: str) -> None:
    value = mapping.get(key)
    if value is None:
        return
    if isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"{path}.{key} must be a non-negative integer")


def _positive_float(mapping: Mapping[str, Any], key: str, path: str) -> None:
    value = mapping.get(key)
    if value is None or isinstance(value, bool) or float(value) <= 0:
        raise ValueError(f"{path}.{key} must be positive")


def _validate_training(training: Mapping[str, Any]) -> None:
    _reject_unknown(training, TRAINING_KEYS, "training")
    for key in ("batch_size", "grad_accum_steps", "max_epochs", "log_interval"):
        _positive_int(training, key, "training")
    for key in ("eval_batch_size", "eval_batches", "max_steps", "max_train_tokens"):
        _positive_int(training, key, "training", optional=True)
    for key in ("num_workers", "eval_num_workers"):
        _non_negative_int(training, key, "training")
    for key in ("eval_interval", "save_interval"):
        _positive_int(training, key, "training")
    _positive_float(training, "learning_rate", "training")
    if training.get("session_max_seconds") is not None:
        _positive_float(training, "session_max_seconds", "training")

    min_lr = training.get("min_lr")
    if min_lr is not None:
        if float(min_lr) < 0 or float(min_lr) > float(training["learning_rate"]):
            raise ValueError("training.min_lr must be between 0 and training.learning_rate")
    if float(training.get("weight_decay", 0.0)) < 0:
        raise ValueError("training.weight_decay must be non-negative")
    betas = training.get("betas", (0.9, 0.95))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("training.betas must contain exactly two values")
    if any(float(beta) < 0 or float(beta) >= 1 for beta in betas):
        raise ValueError("training.betas values must be in [0, 1)")

    active_limits = [
        key for key in ("max_steps", "max_train_tokens") if training.get(key) is not None
    ]
    if len(active_limits) > 1:
        raise ValueError("Set only one of training.max_steps or training.max_train_tokens")
    active_warmups = [
        key
        for key in ("warmup_steps", "warmup_ratio", "warmup_tokens")
        if training.get(key) not in {None, 0, 0.0}
    ]
    if len(active_warmups) > 1:
        raise ValueError(
            "Set only one of training.warmup_steps, training.warmup_ratio, or "
            "training.warmup_tokens"
        )
    warmup_ratio = training.get("warmup_ratio")
    if warmup_ratio is not None and not 0 <= float(warmup_ratio) < 1:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    if str(training.get("precision", "auto")).lower() not in {
        "auto",
        "fp32",
        "float32",
        "32",
        "fp16",
        "float16",
        "16",
        "bf16",
        "bfloat16",
    }:
        raise ValueError("training.precision must be auto, fp32, fp16, or bf16")

    guard = training.get("loss_guard")
    if guard is not None:
        guard = _mapping(guard, "training.loss_guard")
        _reject_unknown(guard, LOSS_GUARD_KEYS, "training.loss_guard")
        for key in ("min_steps", "baseline_window", "recent_window"):
            _positive_int(guard, key, "training.loss_guard")
        _positive_int(
            guard,
            "max_evals_without_improvement",
            "training.loss_guard",
            optional=True,
        )
        for key in ("min_train_ce_drop", "min_val_improvement"):
            value = guard.get(key)
            if value is not None and float(value) < 0:
                raise ValueError(f"training.loss_guard.{key} must be non-negative")


def validate_pretrain_config(config: Mapping[str, Any]) -> None:
    """Fail closed for configs that explicitly opt into strict validation."""

    strict = config.get("strict_config", False)
    if not isinstance(strict, bool):
        raise ValueError("strict_config must be a boolean")
    if not strict:
        return

    _reject_unknown(config, ROOT_KEYS, "root config")
    tokenizer = _mapping(config.get("tokenizer"), "tokenizer")
    model = _mapping(config.get("model"), "model")
    data = _mapping(config.get("data"), "data")
    training = _mapping(config.get("training"), "training")
    del model  # GPTConfig performs the authoritative model-field validation.

    _reject_unknown(tokenizer, TOKENIZER_KEYS, "tokenizer")
    _reject_unknown(data, DATA_KEYS, "data")
    _validate_training(training)

    for key in ("vocab_size", "sample_size", "diagnostic_size", "num_threads"):
        _positive_int(tokenizer, key, "tokenizer")
    _positive_int(data, "min_chars", "data")
    _positive_int(data, "stride", "data")
    validation_split = data.get("validation_split")
    if validation_split is not None and not 0 <= float(validation_split) < 1:
        raise ValueError("data.validation_split must be in [0, 1)")
    if not data.get("train_paths"):
        raise ValueError("Strict pretraining configs require data.train_paths for provenance")
    jsonl_format = str(data.get("jsonl_format", "text")).strip().lower()
    if jsonl_format not in {"text", "messages", "instruction", "source_target"}:
        raise ValueError(
            "data.jsonl_format must be one of: text, messages, instruction, source_target"
        )
    if jsonl_format == "source_target" and bool(data.get("train_on_inputs", False)):
        raise ValueError("source_target requires data.train_on_inputs=false")
    uses_raw_target_only_data = (
        jsonl_format in {"messages", "instruction", "source_target"}
        and not bool(data.get("train_on_inputs", False))
    )
    if not data.get("token_cache_dir") and not uses_raw_target_only_data:
        raise ValueError(
            "Strict pretraining configs require data.token_cache_dir so large corpora "
            "cannot be loaded into RAM accidentally"
        )
    if (
        data.get("token_cache_dir")
        and not data.get("train_token_files")
        and not data.get("train_token_file")
    ):
        raise ValueError("Strict pretraining configs require train token-cache files")
