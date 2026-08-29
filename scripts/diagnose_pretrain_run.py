from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config
from muddywater.diagnostics import (
    UNICODE_PROBES,
    algorithm_warnings,
    build_pretrain_diagnostics,
    cache_summary,
    resolve_token_paths,
    tokenizer_summary,
    warmup_summary,
)
from muddywater.paths import resolve_config_path, resolve_config_paths_in_data, resolve_path
from muddywater.utils import atomic_write_text


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose tokenizer/cache/training schedule health.")
    parser.add_argument("--config", default="configs/experiment_loss_descent_10k.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path; relative paths are resolved from the project root.",
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=0,
        help=(
            "Opt in to a fixed-real-batch overfit probe for this many optimizer steps. "
            "Zero disables the probe and performs no model training."
        ),
    )
    parser.add_argument("--overfit-batch-size", type=int, default=1)
    parser.add_argument("--overfit-start-index", type=int, default=0)
    parser.add_argument(
        "--overfit-seq-len",
        type=int,
        default=128,
        help="Use this prefix length from each real dataset sample.",
    )
    parser.add_argument(
        "--overfit-unigram-samples",
        type=int,
        default=128,
        help="Number of deterministic train samples used for the reference unigram CE.",
    )
    parser.add_argument(
        "--overfit-learning-rate",
        type=float,
        default=None,
        help="AdamW learning rate; defaults to training.learning_rate from the config.",
    )
    parser.add_argument("--overfit-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--overfit-threshold",
        type=float,
        default=None,
        help="Optional CE target; defaults to the estimated train-label unigram CE.",
    )
    parser.add_argument("--overfit-grad-clip", type=float, default=None)
    parser.add_argument("--overfit-device", default="auto")
    parser.add_argument(
        "--overfit-seed",
        type=int,
        default=None,
        help="Defaults to the top-level config seed.",
    )
    return parser.parse_args(argv)


def _resolve_probe_device(value: str):
    import torch

    normalized = str(value or "auto").strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the overfit probe but is unavailable.")
    return device


def run_overfit_probe(
    config: dict,
    config_path: str | Path,
    *,
    steps: int,
    batch_size: int,
    start_index: int,
    sequence_length: int,
    unigram_samples: int,
    learning_rate: float | None,
    weight_decay: float,
    target_ce: float | None,
    grad_clip: float | None,
    device: str,
    seed: int | None,
) -> dict:
    """Build a fresh config model and overfit one fixed real-data batch in memory."""
    import torch

    from muddywater.document_boundaries import resolve_document_boundary_settings
    from muddywater.model import GPTConfig, GPTLanguageModel
    from muddywater.overfit_diagnostic import (
        estimate_dataset_unigram_ce,
        fixed_batch_from_dataset,
        rng_fork_devices,
        run_fixed_batch_overfit,
        seed_probe_rng,
    )
    from muddywater.scaling import apply_auto_scaling
    from muddywater.tokenizer import CharacterTokenizer
    from scripts.pretrain import build_datasets

    if int(steps) <= 0:
        raise ValueError("steps must be positive when running the overfit probe.")
    resolved_config = copy.deepcopy(config)
    resolved_config["data"] = resolve_config_paths_in_data(
        resolved_config.get("data", {}),
        config_path=config_path,
    )
    tokenizer_path = resolve_config_path(
        resolved_config.get("tokenizer", {}).get(
            "path", "outputs/tokenizer/bpe_merged_24k.json"
        ),
        config_path=config_path,
    )
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    resolved_config, scaling_result = apply_auto_scaling(
        resolved_config,
        vocab_size=tokenizer.vocab_size,
        world_size=1,
    )
    model_config = dict(resolved_config.get("model", {}))
    boundary_settings = resolve_document_boundary_settings(resolved_config["data"])
    model_config.setdefault(
        "document_attention_backend",
        "varlen" if boundary_settings.policy == "strict_varlen" else "dense",
    )
    model_config["vocab_size"] = tokenizer.vocab_size
    resolved_config["model"] = model_config
    max_seq_len = int(model_config.get("max_seq_len", 512))
    if int(sequence_length) > max_seq_len:
        raise ValueError(
            f"overfit sequence length {sequence_length} exceeds model max_seq_len {max_seq_len}."
        )

    train_dataset, _ = build_datasets(
        resolved_config["data"],
        resolved_config,
        tokenizer,
        max_seq_len=max_seq_len,
        tokenizer_path=tokenizer_path,
    )
    ignore_index = int(resolved_config.get("training", {}).get("ignore_index", -100))
    unigram = estimate_dataset_unigram_ce(
        train_dataset,
        max_samples=int(unigram_samples),
        ignore_index=ignore_index,
    )
    fixed_batch = fixed_batch_from_dataset(
        train_dataset,
        batch_size=int(batch_size),
        start_index=int(start_index),
        max_seq_len=int(sequence_length),
    )

    probe_seed = int(resolved_config.get("seed", 42) if seed is None else seed)
    target_device = _resolve_probe_device(device)
    with torch.random.fork_rng(devices=rng_fork_devices(target_device)):
        seed_probe_rng(probe_seed, target_device)
        model = GPTLanguageModel(GPTConfig.from_dict(model_config))

    training_config = resolved_config.get("training", {})
    probe_learning_rate = float(
        training_config.get("learning_rate", 3e-4)
        if learning_rate is None
        else learning_rate
    )
    betas = training_config.get("betas", [0.9, 0.95])
    result = run_fixed_batch_overfit(
        model,
        fixed_batch,
        steps=int(steps),
        learning_rate=probe_learning_rate,
        weight_decay=float(weight_decay),
        betas=(float(betas[0]), float(betas[1])),
        ignore_index=ignore_index,
        reference_unigram_ce=float(unigram["ce"]),
        target_ce=target_ce,
        grad_clip=grad_clip,
        device=target_device,
        seed=probe_seed,
        clone_model=False,
    )
    result.update(
        {
            "initialization": "fresh_from_config",
            "checkpoint_loaded": False,
            "formal_run_output_touched": False,
            "dataset_type": train_dataset.__class__.__name__,
            "dataset_size": len(train_dataset),
            "dataset_start_index": int(start_index),
            "reference_unigram": unigram,
            "tokenizer_path": str(tokenizer_path),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "auto_scaling_enabled": bool(scaling_result.enabled),
        }
    )
    return result


def main() -> None:
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8")
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path or args.config)
    config["__config_path__"] = str(config_path or args.config)
    report = {
        "config": str(config_path or Path(args.config)),
        **build_pretrain_diagnostics(config, max_samples=args.max_samples),
    }
    if args.overfit_steps > 0:
        report["overfit_probe"] = run_overfit_probe(
            config,
            config_path or args.config,
            steps=args.overfit_steps,
            batch_size=args.overfit_batch_size,
            start_index=args.overfit_start_index,
            sequence_length=args.overfit_seq_len,
            unigram_samples=args.overfit_unigram_samples,
            learning_rate=args.overfit_learning_rate,
            weight_decay=args.overfit_weight_decay,
            target_ce=args.overfit_threshold,
            grad_clip=args.overfit_grad_clip,
            device=args.overfit_device,
            seed=args.overfit_seed,
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        atomic_write_text(output_path, rendered, overwrite=True)
        print(f"Diagnostic report saved to: {output_path}", flush=True)
    print(rendered)


if __name__ == "__main__":
    main()
