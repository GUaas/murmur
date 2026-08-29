from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from muddywater.dataset import ShardedTokenBlockDataset, TokenBlockDataset
from muddywater.document_boundaries import resolve_document_boundary_settings
from muddywater.evaluation import evaluate_language_model
from muddywater.utils import file_sha256


def _validation_shards(cache_dir: Path) -> list[Path]:
    shards = sorted(cache_dir.glob("val_*.bin"))
    if not shards:
        candidate = cache_dir / "val.bin"
        if candidate.exists():
            shards = [candidate]
    if not shards:
        raise FileNotFoundError(
            f"No val_*.bin or val.bin validation shards found in {cache_dir}"
        )
    return shards


def evaluate_heldout_cache(
    *,
    model: Any,
    tokenizer: Any,
    tokenizer_path: str | Path,
    checkpoint_config: dict[str, Any],
    cache_dir: str | Path,
    device: torch.device,
    batch_size: int = 1,
    max_batches: int | None = None,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir).resolve()
    if not cache_dir.is_dir():
        raise NotADirectoryError(f"Validation cache directory not found: {cache_dir}")
    data_config = checkpoint_config.get("data", {})
    if not isinstance(data_config, dict):
        data_config = {}
    boundary = resolve_document_boundary_settings(data_config)
    shards = _validation_shards(cache_dir)
    common = {
        "max_seq_len": int(model.config.max_seq_len),
        "stride": data_config.get("stride", int(model.config.max_seq_len)),
        "expected_vocab_size": int(tokenizer.vocab_size),
        "expected_tokenizer_sha256": file_sha256(tokenizer_path),
        "expected_add_bos": bool(data_config.get("add_bos", True)),
        "strict_meta": bool(data_config.get("strict_tokenizer_match", True)),
        "document_attention": boundary.document_attention,
        "ignore_cross_document_targets": boundary.ignore_cross_document_targets,
        "single_document_windows": boundary.single_document_windows,
        "ignore_index": -100,
    }
    dataset = (
        TokenBlockDataset(shards[0], **common)
        if len(shards) == 1
        else ShardedTokenBlockDataset(shards, **common)
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_language_model(
        model=model,
        data_loader=loader,
        device=device,
        ignore_index=-100,
        max_batches=max_batches,
        tokenizer=tokenizer,
    )
    manifest_path = cache_dir / "manifest.json"
    return {
        "metrics": metrics.to_dict(),
        "cache_dir": str(cache_dir),
        "shards": [str(path) for path in shards],
        "available_batches": len(loader),
        "requested_max_batches": max_batches,
        "full_validation_split": max_batches is None or max_batches >= len(loader),
        "manifest": {
            "path": str(manifest_path),
            "exists": manifest_path.exists(),
            "sha256": file_sha256(manifest_path) if manifest_path.exists() else None,
        },
        "document_boundary_policy": boundary.policy,
    }
