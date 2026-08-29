from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.checkpoint import load_checkpoint
from muddywater.config import load_config
from muddywater.evaluation import evaluate_language_model
from muddywater.model import GPTConfig, GPTLanguageModel
from muddywater.paths import (
    resolve_config_path,
    resolve_config_paths_in_data,
    resolve_output_path,
    resolve_path,
)
from muddywater.tokenizer import CharacterTokenizer
from muddywater.utils import count_parameters, enable_torch_backends, file_sha256, resolve_device, set_seed
from scripts.generate import resolve_checkpoint_path, validate_tokenizer_compatibility
from scripts.pretrain import build_datasets, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Murmur checkpoint on train/val data.")
    parser.add_argument("--config", default="configs/experiment_loss_descent_10k.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def default_checkpoint(config: dict[str, Any]) -> Path:
    training_config = config.get("training", {})
    output_dir = Path(training_config.get("output_dir", "outputs/run"))
    return output_dir / "best.pt"


def resolve_eval_checkpoint(raw_path: str | Path | None, config: dict[str, Any]) -> Path:
    checkpoint_path = Path(raw_path) if raw_path else default_checkpoint(config)
    return resolve_checkpoint_path(checkpoint_path)


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    tokenizer: CharacterTokenizer,
    fallback_config: dict[str, Any],
) -> GPTLanguageModel:
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}
    model_config = checkpoint_config.get("model", {})
    if not isinstance(model_config, dict):
        model_config = fallback_config.get("model", {})
    model_config = dict(model_config or {})
    model_config["vocab_size"] = tokenizer.vocab_size
    model = GPTLanguageModel(GPTConfig.from_dict(model_config))
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    return model


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path or args.config)
    set_seed(int(config.get("seed", 42)))
    enable_torch_backends()

    data_config = resolve_config_paths_in_data(config.get("data", {}), config_path or args.config)
    config["data"] = data_config
    training_config = dict(config.get("training", {}))
    training_config["output_dir"] = str(
        resolve_output_path(training_config.get("output_dir", "outputs/run"), base=ROOT)
    )
    config["training"] = training_config
    device = resolve_device(config.get("device", "auto"))
    tokenizer_path = resolve_config_path(
        config.get("tokenizer", {}).get("path", "outputs/tokenizer/bpe_merged_24k.json"),
        config_path=config_path,
    )
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    checkpoint_path = resolve_eval_checkpoint(args.checkpoint, config)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}
    validate_tokenizer_compatibility(
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        checkpoint_config=checkpoint_config,
        strict=bool(config.get("generation", {}).get("strict_tokenizer_match", True)),
    )

    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        fallback_config=config,
    ).to(device)
    model_config = getattr(model, "config")
    train_dataset, val_dataset = build_datasets(
        data_config=data_config,
        config=config,
        tokenizer=tokenizer,
        max_seq_len=int(model_config.max_seq_len),
        tokenizer_path=tokenizer_path,
    )

    batch_size = int(training_config.get("eval_batch_size", training_config.get("batch_size", 8)))
    num_workers = int(training_config.get("eval_num_workers", training_config.get("num_workers", 0)))
    dataset = val_dataset if args.split == "val" else train_dataset
    if dataset is None:
        raise ValueError(f"No {args.split} dataset is available for evaluation.")
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        device=device,
        seed=int(config.get("seed", 42)),
        distributed=None,
        for_training=False,
        max_batches=args.max_batches,
    )
    metrics = evaluate_language_model(
        model=model,
        data_loader=loader,
        device=device,
        ignore_index=int(training_config.get("ignore_index", -100)),
        max_batches=args.max_batches,
        tokenizer=tokenizer,
    )
    report = {
        "config": str(Path(args.config)),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "max_batches": args.max_batches,
        "metrics": metrics.to_dict(),
        "model": {
            "parameters": count_parameters(model),
            "max_seq_len": int(model_config.max_seq_len),
            "vocab_size": int(tokenizer.vocab_size),
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": file_sha256(tokenizer_path),
        },
        "torch": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
    }

    if args.output:
        write_report(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
