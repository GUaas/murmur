from __future__ import annotations

import gc
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .flops import estimate_training_flops, summarize_parameters
from .model import GPTConfig, GPTLanguageModel
from .optim import build_grad_scaler, resolve_precision


@dataclass(frozen=True)
class TrainingBenchmarkSettings:
    batch_size: int
    grad_accum_steps: int
    warmup_steps: int = 2
    measured_steps: int = 8
    target_tokens_per_second: float = 60_000.0
    production_margin_tokens_per_second: float = 65_000.0
    planned_train_tokens: int = 20_000_000_000
    compile_mode: str = "config"

    def validate(self) -> None:
        for name in ("batch_size", "grad_accum_steps", "measured_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if float(self.target_tokens_per_second) <= 0:
            raise ValueError("target_tokens_per_second must be positive")
        if float(self.production_margin_tokens_per_second) < float(
            self.target_tokens_per_second
        ):
            raise ValueError(
                "production_margin_tokens_per_second must be at least the target"
            )
        if int(self.planned_train_tokens) <= 0:
            raise ValueError("planned_train_tokens must be positive")
        if normalize_compile_mode(self.compile_mode) not in {"config", "on", "off"}:
            raise AssertionError("Unreachable compile mode validation")


def normalize_compile_mode(value: str) -> str:
    normalized = str(value or "config").strip().lower()
    aliases = {
        "true": "on",
        "yes": "on",
        "1": "on",
        "false": "off",
        "no": "off",
        "0": "off",
        "auto": "config",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"config", "on", "off"}:
        raise ValueError("compile_mode must be one of: config, on, off")
    return normalized


def resolve_compile_enabled(
    compile_mode: str,
    training_config: Mapping[str, Any],
) -> bool:
    normalized = normalize_compile_mode(compile_mode)
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    return bool(training_config.get("compile", False))


def model_config_from_experiment(config: Mapping[str, Any]) -> GPTConfig:
    tokenizer_config = config.get("tokenizer", {})
    model_config = dict(config.get("model", {}))
    vocab_size = tokenizer_config.get("vocab_size")
    if vocab_size is None:
        raise ValueError(
            "Synthetic benchmarking requires tokenizer.vocab_size in the config"
        )
    model_config["vocab_size"] = int(vocab_size)
    return GPTConfig.from_dict(model_config)


def variant_name(config_path: str | Path) -> str:
    stem = Path(config_path).stem.lower()
    if "deep" in stem:
        return "deep"
    if "wide" in stem:
        return "wide"
    return Path(config_path).stem


def choose_recommended_variant(
    results: list[Mapping[str, Any]],
    target_tokens_per_second: float,
) -> str | None:
    successful = [
        result
        for result in results
        if result.get("status") == "ok"
        and float(result.get("tokens_per_second", 0.0)) >= target_tokens_per_second
    ]
    deep = [result for result in successful if result.get("variant") == "deep"]
    if deep:
        return "deep"
    wide = [result for result in successful if result.get("variant") == "wide"]
    if wide:
        return "wide"
    return None


def _autocast_context(precision, device: torch.device):
    if not precision.amp_enabled:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=precision.amp_dtype,
        enabled=True,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _build_optimizer(
    model: GPTLanguageModel,
    training_config: Mapping[str, Any],
    device: torch.device,
) -> torch.optim.Optimizer:
    betas = training_config.get("betas", (0.9, 0.95))
    return model.configure_optimizers(
        weight_decay=float(training_config.get("weight_decay", 0.1)),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        betas=(float(betas[0]), float(betas[1])),
        device_type=device.type,
        optimizer_type=str(training_config.get("optimizer", "adamw")),
        muon_momentum=float(training_config.get("muon_momentum", 0.95)),
        muon_ns_steps=int(training_config.get("muon_ns_steps", 5)),
        muon_update_scale=float(training_config.get("muon_update_scale", 1.0)),
        muon_nesterov=bool(training_config.get("muon_nesterov", True)),
        muon_orthogonalization=str(
            training_config.get("muon_orthogonalization", "polar_express")
        ),
        muon_row_equilibration=bool(
            training_config.get("muon_row_equilibration", True)
        ),
        muon_renormalize=bool(training_config.get("muon_renormalize", True)),
        embedding_learning_rate=training_config.get("embedding_lr"),
        value_embedding_learning_rate=training_config.get("value_embedding_lr"),
        lm_head_learning_rate=training_config.get("lm_head_lr"),
        matrix_learning_rate=training_config.get("matrix_lr"),
        scalar_learning_rate=training_config.get("scalar_lr"),
    )


def _run_optimizer_step(
    *,
    runtime_model,
    optimizer: torch.optim.Optimizer,
    scaler,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    grad_accum_steps: int,
    precision,
    device: torch.device,
    grad_clip: float,
    ignore_index: int,
    z_loss_weight: float,
) -> float:
    last_loss = float("nan")
    for _ in range(int(grad_accum_steps)):
        with _autocast_context(precision, device):
            outputs = runtime_model(
                input_ids,
                labels=labels,
                ignore_index=ignore_index,
                z_loss_weight=z_loss_weight,
            )
            loss = outputs["loss"]
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError("Model did not return a scalar training loss")
            scaled_loss = loss / int(grad_accum_steps)
        scaler.scale(scaled_loss).backward()
        last_loss = float(loss.detach().float().item())

    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            grad_clip,
        )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return last_loss


def run_training_benchmark(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    settings: TrainingBenchmarkSettings,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Measure steady-state synthetic pretraining throughput for one architecture."""

    settings.validate()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The AMD/NVIDIA throughput benchmark requires a CUDA/ROCm GPU")

    training_config = dict(config.get("training", {}))
    precision = resolve_precision(training_config, device.type)
    model_config = model_config_from_experiment(config)
    compile_enabled = resolve_compile_enabled(settings.compile_mode, training_config)
    if compile_enabled and not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile was requested but is unavailable")

    _clear_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = GPTLanguageModel(model_config).to(device)
    model.train()
    parameter_summary = summarize_parameters(model)
    flop_estimate = estimate_training_flops(model, token_count=1)
    optimizer = _build_optimizer(model, training_config, device)
    runtime_model = torch.compile(model) if compile_enabled else model
    scaler = build_grad_scaler(precision)

    input_ids = torch.randint(
        0,
        model_config.vocab_size,
        (settings.batch_size, model_config.max_seq_len),
        dtype=torch.long,
        device=device,
    )
    labels = torch.randint(
        0,
        model_config.vocab_size,
        (settings.batch_size, model_config.max_seq_len),
        dtype=torch.long,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    step_kwargs = {
        "runtime_model": runtime_model,
        "optimizer": optimizer,
        "scaler": scaler,
        "input_ids": input_ids,
        "labels": labels,
        "grad_accum_steps": settings.grad_accum_steps,
        "precision": precision,
        "device": device,
        "grad_clip": float(training_config.get("grad_clip", 0.0)),
        "ignore_index": int(training_config.get("ignore_index", -100)),
        "z_loss_weight": float(training_config.get("z_loss_weight", 0.0)),
    }

    for _ in range(settings.warmup_steps):
        _run_optimizer_step(**step_kwargs)
    _synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    step_seconds: list[float] = []
    last_loss = float("nan")
    for _ in range(settings.measured_steps):
        started = time.perf_counter()
        last_loss = _run_optimizer_step(**step_kwargs)
        _synchronize(device)
        step_seconds.append(time.perf_counter() - started)

    elapsed = sum(step_seconds)
    tokens_per_optimizer_step = (
        settings.batch_size
        * model_config.max_seq_len
        * settings.grad_accum_steps
    )
    measured_tokens = tokens_per_optimizer_step * settings.measured_steps
    tokens_per_second = measured_tokens / elapsed
    planned_hours = settings.planned_train_tokens / tokens_per_second / 3600
    device_properties = torch.cuda.get_device_properties(device)
    result = {
        "status": "ok",
        "variant": variant_name(config_path),
        "config": str(Path(config_path)),
        "architecture": {
            "n_layers": model_config.n_layers,
            "n_embd": model_config.n_embd,
            "n_heads": model_config.n_heads,
            "n_kv_heads": model_config.n_kv_heads,
            "head_dim": model_config.n_embd // model_config.n_heads,
            "max_seq_len": model_config.max_seq_len,
            "vocab_size": model_config.vocab_size,
        },
        "parameters": parameter_summary.to_dict(),
        "settings": asdict(settings),
        "compile_enabled": compile_enabled,
        "precision": precision.name,
        "optimizer": str(training_config.get("optimizer", "adamw")),
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "measured_tokens": measured_tokens,
        "elapsed_seconds": elapsed,
        "mean_step_seconds": elapsed / len(step_seconds),
        "min_step_seconds": min(step_seconds),
        "max_step_seconds": max(step_seconds),
        "tokens_per_second": tokens_per_second,
        "meets_60k_target": tokens_per_second >= settings.target_tokens_per_second,
        "has_production_margin": (
            tokens_per_second >= settings.production_margin_tokens_per_second
        ),
        "estimated_hours_for_planned_tokens": planned_hours,
        "last_loss": last_loss,
        "estimated_training_flops_per_token": flop_estimate.flops_per_token,
        "estimated_training_tflops": (
            flop_estimate.flops_per_token * tokens_per_second / 1e12
        ),
        "memory": {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "runtime": {
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": int(device_properties.total_memory),
            "torch_version": torch.__version__,
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_version": getattr(torch.version, "cuda", None),
        },
    }
    if not math.isfinite(last_loss):
        raise RuntimeError(f"Non-finite loss during throughput benchmark: {last_loss}")
    return result


def release_benchmark_memory() -> None:
    """Release Python and accelerator caches between architecture variants."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
