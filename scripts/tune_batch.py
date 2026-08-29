from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config
from muddywater.document_boundaries import resolve_document_boundary_settings
from muddywater.model import GPTConfig, GPTLanguageModel
from muddywater.optim import resolve_precision
from muddywater.paths import resolve_config_path, resolve_path
from muddywater.scaling import apply_auto_scaling
from muddywater.tokenizer import CharacterTokenizer
from muddywater.utils import enable_torch_backends, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the largest synthetic micro-batch for a pretraining config."
    )
    parser.add_argument("--config", default="configs/experiment_loss_descent_10k.yaml")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--start-batch-size", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--target-global-batch-tokens",
        type=int,
        default=None,
        help="Defaults to batch_size * grad_accum_steps * max_seq_len * world_size from config.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_device(raw_device: str | None) -> torch.device:
    requested = str(raw_device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def build_model_from_config(
    config: dict[str, Any],
    config_path: Path | None,
    world_size: int,
) -> tuple[GPTLanguageModel, CharacterTokenizer, dict[str, Any]]:
    tokenizer_path = resolve_config_path(
        config.get("tokenizer", {}).get("path", "outputs/tokenizer/bpe_merged_24k.json"),
        config_path=config_path,
    )
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    config, _ = apply_auto_scaling(
        config,
        vocab_size=tokenizer.vocab_size,
        world_size=world_size,
    )
    model_config = dict(config.get("model", {}))
    boundary_settings = resolve_document_boundary_settings(config.get("data", {}))
    if "document_attention_backend" not in model_config:
        model_config["document_attention_backend"] = (
            "varlen" if boundary_settings.policy == "strict_varlen" else "dense"
        )
    model_config["vocab_size"] = tokenizer.vocab_size
    return GPTLanguageModel(GPTConfig.from_dict(model_config)), tokenizer, config


def autocast_context(training_config: dict[str, Any], device: torch.device):
    precision = resolve_precision(training_config, device.type)
    if not precision.amp_enabled:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=precision.amp_dtype,
        enabled=True,
    )


def probe_batch_size(
    model: GPTLanguageModel,
    tokenizer: CharacterTokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    clear_cuda_cache(device)
    model.zero_grad(set_to_none=True)
    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    labels = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    started_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    with autocast_context(training_config, device):
        outputs = model(
            input_ids,
            labels=labels,
            ignore_index=int(training_config.get("ignore_index", -100)),
            z_loss_weight=float(training_config.get("z_loss_weight", 0.0)),
        )
        loss = outputs["loss"]
    if not isinstance(loss, torch.Tensor):
        raise RuntimeError("Model did not return a loss during batch probing.")
    loss.backward()
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "batch_size": batch_size,
        "loss": float(loss.detach().float().item()),
        "cuda_memory_allocated_bytes": int(started_allocated),
        "cuda_peak_memory_bytes": int(peak_allocated),
    }


def probe_largest_batch(
    model: GPTLanguageModel,
    tokenizer: CharacterTokenizer,
    seq_len: int,
    device: torch.device,
    training_config: dict[str, Any],
    start_batch_size: int,
    max_batch_size: int,
) -> tuple[int, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    best = 0
    batch_size = max(1, int(start_batch_size))
    while batch_size <= int(max_batch_size):
        try:
            metrics = probe_batch_size(
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                training_config=training_config,
            )
            metrics["status"] = "ok"
            attempts.append(metrics)
            best = batch_size
            batch_size *= 2
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "oom",
                    "error": str(exc),
                }
            )
            clear_cuda_cache(device)
            break
    return best, attempts


def build_recommendation(
    config: dict[str, Any],
    best_batch_size: int,
    seq_len: int,
    world_size: int,
    target_global_batch_tokens: int | None,
) -> dict[str, Any]:
    training = config.get("training", {})
    configured_batch = int(training.get("batch_size", 1) or 1)
    configured_accum = int(training.get("grad_accum_steps", 1) or 1)
    target_tokens = target_global_batch_tokens or (
        configured_batch * configured_accum * seq_len * max(1, int(world_size))
    )
    usable_batch = max(1, int(best_batch_size))
    grad_accum_steps = max(
        1,
        math.ceil(int(target_tokens) / (usable_batch * seq_len * max(1, int(world_size)))),
    )
    return {
        "batch_size": usable_batch,
        "grad_accum_steps": grad_accum_steps,
        "global_batch_tokens": usable_batch * grad_accum_steps * seq_len * max(1, int(world_size)),
        "target_global_batch_tokens": int(target_tokens),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path or args.config)
    config["__config_path__"] = str(config_path or args.config)
    set_seed(int(config.get("seed", 42)))
    enable_torch_backends()
    device = resolve_device(config.get("device", "auto"))

    model, tokenizer, config = build_model_from_config(
        config,
        config_path=config_path,
        world_size=max(1, int(args.world_size)),
    )
    model.to(device)
    model.train()
    training_config = config.get("training", {})
    seq_len = int(config.get("model", {}).get("max_seq_len", 512))
    best_batch_size, attempts = probe_largest_batch(
        model=model,
        tokenizer=tokenizer,
        seq_len=seq_len,
        device=device,
        training_config=training_config,
        start_batch_size=args.start_batch_size,
        max_batch_size=args.max_batch_size,
    )
    if best_batch_size <= 0:
        raise RuntimeError("No batch size succeeded; lower max_seq_len or inspect the model config.")
    recommendation = build_recommendation(
        config=config,
        best_batch_size=best_batch_size,
        seq_len=seq_len,
        world_size=max(1, int(args.world_size)),
        target_global_batch_tokens=args.target_global_batch_tokens,
    )
    report = {
        "config": str(args.config),
        "device": str(device),
        "seq_len": seq_len,
        "world_size": max(1, int(args.world_size)),
        "attempts": attempts,
        "recommendation": recommendation,
        "yaml_patch": {
            "training": {
                "batch_size": recommendation["batch_size"],
                "grad_accum_steps": recommendation["grad_accum_steps"],
            }
        },
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
