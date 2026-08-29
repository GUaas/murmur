from __future__ import annotations

import glob
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache_stats import (
    estimate_dense_attention_mask_bytes,
    human_bytes,
    summarize_dataset_document_coverage,
    summarize_sample_coverage,
)
from .dataset import ShardedTokenBlockDataset, TokenBlockDataset
from .document_boundaries import resolve_document_boundary_settings
from .packing import cross_document_targets, document_ids_for_positions
from .paths import resolve_config_path
from .tokenizer import CharacterTokenizer
from .utils import as_list, file_sha256


UNICODE_PROBES = ["\U00020bb7", "\U0001f642", "\u03a9", "\u200d"]


def _present(value: Any) -> bool:
    return value is not None and value != ""


def resolve_token_paths(cache_dir: Path, configured, fallback: str) -> list[Path]:
    patterns = as_list(configured) or [fallback]
    paths: list[Path] = []
    for raw_pattern in patterns:
        pattern = Path(str(raw_pattern))
        if not pattern.is_absolute():
            pattern = cache_dir / pattern
        pattern_str = str(pattern)
        if any(ch in pattern_str for ch in "*?["):
            paths.extend(Path(path) for path in sorted(glob.glob(pattern_str)) if Path(path).is_file())
        elif pattern.exists() and pattern.is_file():
            paths.append(pattern)
    return sorted({path.resolve(): path for path in paths}.values())


def warmup_summary(config: dict[str, Any], world_size: int = 1) -> dict[str, Any]:
    training = config.get("training", {})
    model = config.get("model", {})
    batch_size = int(training.get("batch_size", 1))
    grad_accum_steps = int(training.get("grad_accum_steps", 1))
    seq_len = int(model.get("max_seq_len", 0) or 0)
    tokens_per_step = batch_size * grad_accum_steps * seq_len * max(1, int(world_size))
    active = [
        key
        for key in ("warmup_tokens", "warmup_ratio", "warmup_steps")
        if _present(training.get(key)) and (key != "warmup_steps" or int(training.get(key) or 0) > 0)
    ]
    return {
        "tokens_per_step_estimate": tokens_per_step,
        "world_size": max(1, int(world_size)),
        "active_warmup_knobs": active,
        "has_warmup_conflict": len(active) > 1,
    }


def tokenizer_summary(config: dict[str, Any]) -> dict[str, Any]:
    tokenizer_path = Path(
        resolve_config_path(
            config.get("tokenizer", {}).get("path", "outputs/tokenizer/bpe_merged_24k.json"),
            config_path=config.get("__config_path__"),
        )
    )
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    unk_id = getattr(tokenizer, "unk_id", None)
    probes = []
    total_unk = 0
    for text in UNICODE_PROBES:
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        unk_tokens = sum(1 for token_id in ids if unk_id is not None and int(token_id) == int(unk_id))
        total_unk += unk_tokens
        probes.append({"text": text, "ids": ids, "unk_tokens": unk_tokens})

    diagnostics_path = tokenizer_path.with_suffix(tokenizer_path.suffix + ".diagnostics.json")
    diagnostics = None
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    return {
        "path": str(tokenizer_path),
        "sha256": file_sha256(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "unicode_probe_unk_tokens": total_unk,
        "unicode_probes": probes,
        "diagnostics_path": str(diagnostics_path) if diagnostics_path.exists() else None,
        "diagnostics": diagnostics,
    }


def _dataset_shard_for_index(dataset, index: int):
    shards = getattr(dataset, "shards", None)
    if not shards:
        return dataset, int(index)
    shard_idx = bisect_right(dataset.cumulative_sizes, int(index))
    previous = 0 if shard_idx == 0 else dataset.cumulative_sizes[shard_idx - 1]
    return shards[shard_idx], int(index) - previous


def _cache_boundary_stats(dataset, max_samples: int, ignore_index: int) -> dict[str, Any]:
    ignored_labels = 0
    total_labels = 0
    cross_document_windows = 0
    cross_document_targets_seen = 0
    samples_with_boundaries = 0
    samples_checked = min(len(dataset), max(0, int(max_samples)))

    for idx in range(samples_checked):
        item = dataset[idx]
        labels = item["labels"]
        total_labels += int(labels.numel())
        ignored_labels += int((labels == ignore_index).sum().item())

        shard, local_index = _dataset_shard_for_index(dataset, idx)
        doc_starts = getattr(shard, "doc_starts", None)
        if doc_starts is None:
            continue
        sample_starts = getattr(shard, "sample_starts", None)
        if sample_starts is None:
            continue
        start = int(sample_starts[local_index])
        positions = np.arange(start, start + int(shard.window_size), dtype=np.uint64)
        doc_ids = document_ids_for_positions(doc_starts, positions)
        cross_targets = cross_document_targets(doc_ids)
        samples_with_boundaries += 1
        cross_document_targets_seen += int(cross_targets.sum().item())
        if bool(np.any(doc_ids != doc_ids[0])):
            cross_document_windows += 1

    ignored_ratio = ignored_labels / total_labels if total_labels else math.nan
    cross_window_ratio = (
        cross_document_windows / samples_with_boundaries if samples_with_boundaries else math.nan
    )
    cross_target_ratio = (
        cross_document_targets_seen / total_labels if total_labels else math.nan
    )
    return {
        "samples_checked": samples_checked,
        "samples_with_boundary_metadata": samples_with_boundaries,
        "ignored_label_ratio_in_checked_samples": ignored_ratio,
        "effective_token_usage_in_checked_samples": (
            1.0 - ignored_ratio if math.isfinite(ignored_ratio) else math.nan
        ),
        "cross_document_windows_in_checked_samples": cross_document_windows,
        "cross_document_window_ratio_in_checked_samples": cross_window_ratio,
        "cross_document_targets_in_checked_samples": cross_document_targets_seen,
        "cross_document_target_ratio_in_checked_samples": cross_target_ratio,
    }


def cache_summary(config: dict[str, Any], max_samples: int) -> dict[str, Any] | None:
    data = config.get("data", {})
    cache_dir = data.get("token_cache_dir")
    if not cache_dir:
        return None

    tokenizer_path = Path(
        resolve_config_path(
            config.get("tokenizer", {}).get("path", "outputs/tokenizer/bpe_merged_24k.json"),
            config_path=config.get("__config_path__"),
        )
    )
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    token_paths = resolve_token_paths(
        Path(resolve_config_path(cache_dir, config_path=config.get("__config_path__"))),
        data.get("train_token_files"),
        data.get("train_token_file", "train.bin"),
    )
    if not token_paths:
        return {"error": f"train token cache not found in {cache_dir}"}

    boundary_settings = resolve_document_boundary_settings(data)
    ignore_index = int(config.get("training", {}).get("ignore_index", -100))
    dataset_kwargs = {
        "max_seq_len": int(config.get("model", {}).get("max_seq_len", 512)),
        "stride": data.get("stride"),
        "tail_min_gap_ratio": float(data.get("tail_min_gap_ratio", 0.5)),
        "expected_vocab_size": tokenizer.vocab_size,
        "expected_tokenizer_sha256": file_sha256(tokenizer_path)
        if bool(data.get("strict_tokenizer_match", True))
        else None,
        "expected_add_bos": bool(data.get("add_bos", True)),
        "strict_meta": bool(data.get("strict_tokenizer_match", True)),
        "document_attention": boundary_settings.document_attention,
        "ignore_cross_document_targets": boundary_settings.ignore_cross_document_targets,
        "single_document_windows": boundary_settings.single_document_windows,
        "ignore_index": ignore_index,
    }
    try:
        dataset = (
            TokenBlockDataset(token_paths[0], **dataset_kwargs)
            if len(token_paths) == 1
            else ShardedTokenBlockDataset(token_paths, **dataset_kwargs)
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "paths": [str(path) for path in token_paths],
            "document_boundary_policy": boundary_settings.policy,
            "document_boundary_strategy": boundary_settings.strategy,
        }

    shards = getattr(dataset, "shards", [dataset])
    boundary_stats = _cache_boundary_stats(
        dataset,
        max_samples=max_samples,
        ignore_index=ignore_index,
    )
    document_coverage = summarize_dataset_document_coverage(dataset)
    sample_coverage = summarize_sample_coverage(dataset)
    return {
        "paths": [str(path) for path in token_paths],
        "num_tokens": sum(int(shard.num_tokens) for shard in shards),
        "num_samples": len(dataset),
        **boundary_stats,
        "document_coverage": document_coverage,
        "sample_coverage": sample_coverage,
        "document_attention": bool(dataset_kwargs["document_attention"]),
        "ignore_cross_document_targets": bool(dataset_kwargs["ignore_cross_document_targets"]),
        "single_document_windows": bool(dataset_kwargs["single_document_windows"]),
        "document_boundary_policy": boundary_settings.policy,
        "document_boundary_strategy": boundary_settings.strategy,
    }


def algorithm_warnings(
    config: dict[str, Any],
    tokenizer: dict[str, Any],
    cache: dict[str, Any] | None,
) -> list[str]:
    warnings: list[str] = []
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    boundary_settings = resolve_document_boundary_settings(data)

    vocab_size = int(tokenizer.get("vocab_size", 0) or 0)
    if 0 < vocab_size < 32000:
        warnings.append(
            "tokenizer.vocab_size is below 32K; Chinese/code mixed corpora often waste context with very small BPE vocabularies."
        )

    max_seq_len = int(model.get("max_seq_len", 0) or 0)
    if 0 < max_seq_len < 4096:
        warnings.append(
            "model.max_seq_len is below 4096; this is a conservative short-context baseline, not a modern long-context recipe."
        )

    if bool(model.get("input_rmsnorm", False)) and bool(model.get("tie_weights", True)):
        warnings.append(
            "model.input_rmsnorm=true with tied input/output embeddings can create a dominant "
            "current-token copy logit and starve contextual branches; keep input_rmsnorm=false "
            "or explicitly untie the output head."
        )

    if boundary_settings.policy == "strict":
        dense_mask_bytes = estimate_dense_attention_mask_bytes(
            batch_size=int(training.get("batch_size", 1) or 1),
            seq_len=max_seq_len,
        )
        warning_threshold = int(
            training.get("dense_mask_warning_bytes") or 256 * 1024 * 1024
        )
        warnings.append(
            "document_boundary_strategy=strict_dense builds dense [B,T,T] masks; "
            f"estimated per-rank mask memory is {human_bytes(dense_mask_bytes)}."
        )
        if dense_mask_bytes >= warning_threshold:
            warnings.append(
                "strict_dense mask memory exceeds the configured warning threshold "
                f"({human_bytes(warning_threshold)}); switch to label_only_fast or "
                "strict_varlen when throughput or memory matters."
            )
    elif boundary_settings.policy == "strict_varlen":
        warnings.append(
            "document_boundary_strategy=strict_varlen keeps strict document isolation without "
            "building dense [B,T,T] masks; current fallback segments documents with SDPA."
        )
    elif boundary_settings.policy == "label_only":
        warnings.append(
            "document_boundary_strategy=label_only_fast masks cross-document labels but still lets packed tokens attend across documents."
        )
    elif boundary_settings.single_document_windows:
        warnings.append(
            "document_boundary_strategy=single_doc_windows keeps causal attention strict but skips documents shorter than one full training window."
        )
        if cache is not None:
            coverage = (cache.get("document_coverage") or {}).get("single_doc_token_coverage")
            if isinstance(coverage, (int, float)) and coverage < 0.8:
                warnings.append(
                    "single_doc_windows would use less than 80% of cached tokens because many documents are shorter than one full window."
                )

    if bool(data.get("ignore_cross_document_targets", True)) and cache is not None:
        ignored_ratio = cache.get("ignored_label_ratio_in_checked_samples")
        if isinstance(ignored_ratio, (int, float)) and ignored_ratio > 0.05:
            warnings.append(
                "More than 5% of checked labels are ignored at document boundaries; consider longer documents, source-aware packing, or single_doc_windows."
            )

    optimizer = str(training.get("optimizer", "adamw")).lower()
    if optimizer == "adamw":
        warnings.append(
            "training.optimizer=adamw is stable but conservative; muon_adamw is available for controlled efficiency experiments."
        )

    config_name = str(config.get("training", {}).get("output_dir", "")) + " " + str(config)
    if "sota" in config_name.lower():
        warnings.append("This config still contains 'sota' naming; reserve that word for benchmark-proven recipes.")

    if training.get("eval_batches") is None:
        warnings.append(
            "training.eval_batches is unset; full validation may be slow and less comparable across checkpoints on large corpora."
        )

    return warnings


def build_pretrain_diagnostics(
    config: dict[str, Any],
    max_samples: int = 128,
    world_size: int = 1,
    include_torch: bool = True,
) -> dict[str, Any]:
    tokenizer = tokenizer_summary(config)
    cache = cache_summary(config, max_samples=max_samples)
    report = {
        "tokenizer": tokenizer,
        "cache": cache,
        "warmup": warmup_summary(config, world_size=world_size),
        "memory_estimates": memory_estimates(config),
        "algorithm_warnings": algorithm_warnings(config, tokenizer=tokenizer, cache=cache),
    }
    if include_torch:
        report["torch"] = {"cuda_available": torch.cuda.is_available()}
    return report


def memory_estimates(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    model = config.get("model", {})
    batch_size = int(training.get("batch_size", 1) or 1)
    seq_len = int(model.get("max_seq_len", 0) or 0)
    dense_mask_bytes = estimate_dense_attention_mask_bytes(batch_size=batch_size, seq_len=seq_len)
    return {
        "dense_attention_mask_bytes_per_rank": dense_mask_bytes,
        "dense_attention_mask_human_per_rank": human_bytes(dense_mask_bytes),
        "batch_size": batch_size,
        "max_seq_len": seq_len,
    }


def write_pretrain_diagnostics(
    config: dict[str, Any],
    output_dir: str | Path,
    max_samples: int = 128,
    world_size: int = 1,
) -> tuple[Path, dict[str, Any]]:
    report = build_pretrain_diagnostics(
        config,
        max_samples=max_samples,
        world_size=world_size,
        include_torch=True,
    )
    path = Path(output_dir) / "diagnostics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, report
