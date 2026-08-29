from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config
from muddywater.config_validation import validate_pretrain_config
from muddywater.checkpoint import load_checkpoint
from muddywater.dataset import (
    LabeledLanguageModelingDataset,
    LanguageModelingDataset,
    ShardedTokenBlockDataset,
    TokenBlockDataset,
    load_labeled_texts,
    load_texts,
    split_train_val,
)
from muddywater.diagnostics import write_pretrain_diagnostics
from muddywater.document_boundaries import (
    resolve_document_attention,
    resolve_document_boundary_settings,
)
from muddywater.model import GPTConfig, GPTLanguageModel
from muddywater.optim import resolve_precision
from muddywater.distributed import DistributedContext, barrier, cleanup_distributed, init_distributed
from muddywater.packing import route_to_validation, stable_hash_fraction
from muddywater.paths import (
    resolve_config_path,
    resolve_config_paths_in_data,
    resolve_output_path,
    resolve_path,
)
from muddywater.run_manifest import (
    build_training_identity,
    update_run_manifest,
    write_initial_run_manifest,
)
from muddywater.resume_validation import validate_resume_model_config, validate_resume_training_state
from muddywater.sampling import (
    DeterministicEpochSampler,
    NonPaddingDistributedSampler,
    ResumeOffsetSampler,
    seed_worker,
)
from muddywater.scaling import apply_auto_scaling
from muddywater.templates import DEFAULT_SYSTEM_PROMPT
from muddywater.tokenizer import CharacterTokenizer
from muddywater.trainer import Trainer
from muddywater.utils import (
    archive_resume_metadata,
    as_list,
    enable_torch_backends,
    file_sha256,
    get_num_workers,
    prepare_training_output_dir,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain MuddyWaterAI-Murmur from scratch.")
    parser.add_argument("--config", type=str, default="configs/experiment_loss_descent_10k.yaml")
    return parser.parse_args()


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
    distributed: DistributedContext | None = None,
    for_training: bool = True,
    max_batches: int | None = None,
) -> DataLoader:
    max_samples_per_rank = (
        None
        if for_training or max_batches is None
        else max(0, int(max_batches)) * int(batch_size)
    )
    if distributed is not None and distributed.enabled:
        if for_training:
            sampler = DistributedSampler(
                dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=False,
            )
        else:
            sampler = NonPaddingDistributedSampler(
                dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=shuffle,
                seed=seed,
                max_samples_per_rank=max_samples_per_rank,
            )
    else:
        if for_training:
            sampler = DeterministicEpochSampler(dataset, seed=seed, shuffle=True) if shuffle else None
        elif shuffle or max_samples_per_rank is not None:
            sampler = NonPaddingDistributedSampler(
                dataset,
                num_replicas=1,
                rank=0,
                shuffle=shuffle,
                seed=seed,
                max_samples_per_rank=max_samples_per_rank,
            )
        else:
            sampler = None
    if for_training:
        sampler = ResumeOffsetSampler(
            sampler if sampler is not None else SequentialSampler(dataset)
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False if sampler is not None else shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "sampler": sampler,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def resolve_token_paths(cache_dir: Path, configured, fallback: str) -> list[Path]:
    patterns = as_list(configured) or [fallback]
    paths: list[Path] = []
    for raw_pattern in patterns:
        pattern = Path(str(raw_pattern))
        if not pattern.is_absolute():
            pattern = cache_dir / pattern
        pattern_str = str(pattern)
        if any(ch in pattern_str for ch in "*?["):
            matches = sorted(Path(path) for path in glob.glob(pattern_str))
            paths.extend(path for path in matches if path.is_file())
        elif pattern.exists() and pattern.is_file():
            paths.append(pattern)
    return sorted({path.resolve(): path for path in paths}.values())


def token_cache_has_full_window(path: Path, max_seq_len: int) -> bool:
    """Return false only when metadata proves the cache is too short."""
    if path.stat().st_size <= 0:
        return False
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        num_tokens = int(meta.get("num_tokens", 0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return True
    return num_tokens >= int(max_seq_len) + 1


def build_token_dataset(
    token_paths: list[Path],
    max_seq_len: int,
    stride,
    tokenizer,
    tokenizer_digest: str | None,
    add_bos: bool,
    strict_meta: bool,
    data_config: dict,
    config: dict,
):
    boundary_settings = resolve_document_boundary_settings(data_config)
    common = {
        "max_seq_len": max_seq_len,
        "stride": stride,
        "tail_min_gap_ratio": float(data_config.get("tail_min_gap_ratio", 0.5)),
        "expected_vocab_size": tokenizer.vocab_size,
        "expected_tokenizer_sha256": tokenizer_digest,
        "expected_add_bos": add_bos,
        "strict_meta": strict_meta,
        "document_attention": boundary_settings.document_attention,
        "ignore_cross_document_targets": boundary_settings.ignore_cross_document_targets,
        "single_document_windows": boundary_settings.single_document_windows,
        "ignore_index": int(config.get("training", {}).get("ignore_index", -100)),
    }
    if len(token_paths) == 1:
        return TokenBlockDataset(token_paths[0], **common)
    return ShardedTokenBlockDataset(token_paths, **common)


def load_plain_texts(data_config: dict, config: dict) -> tuple[list[str], list[str]]:
    train_texts = load_texts(
        data_config.get("train_paths"),
        jsonl_text_key=data_config.get("jsonl_text_key", "text"),
        jsonl_format=data_config.get("jsonl_format", "text"),
        instruction_key=data_config.get("instruction_key", "instruction"),
        input_key=data_config.get("input_key", "input"),
        output_key=data_config.get("output_key", "output"),
        chat_template=data_config.get("chat_template", "chatml"),
        system_prompt=data_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        instruction_template=data_config.get(
            "instruction_template",
            "用户：{instruction}\n输入：{input}\n助手：{output}",
        ),
        instruction_template_no_input=data_config.get(
            "instruction_template_no_input",
            "用户：{instruction}\n助手：{output}",
        ),
        txt_split=data_config.get("txt_split", "line"),
        min_chars=int(data_config.get("min_chars", 1)),
    )

    val_paths = as_list(data_config.get("val_paths"))
    if val_paths:
        val_texts = load_texts(
            val_paths,
            jsonl_text_key=data_config.get("jsonl_text_key", "text"),
            jsonl_format=data_config.get("jsonl_format", "text"),
            instruction_key=data_config.get("instruction_key", "instruction"),
            input_key=data_config.get("input_key", "input"),
            output_key=data_config.get("output_key", "output"),
            chat_template=data_config.get("chat_template", "chatml"),
            system_prompt=data_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            instruction_template=data_config.get(
                "instruction_template",
                "用户：{instruction}\n输入：{input}\n助手：{output}",
            ),
            instruction_template_no_input=data_config.get(
                "instruction_template_no_input",
                "用户：{instruction}\n助手：{output}",
            ),
            txt_split=data_config.get("txt_split", "line"),
            min_chars=int(data_config.get("min_chars", 1)),
        )
    else:
        train_texts, val_texts = split_train_val(
            train_texts,
            val_ratio=float(data_config.get("validation_split", 0.05)),
            seed=int(config.get("seed", 42)),
            shuffle=bool(data_config.get("shuffle_split", True)),
            split_mode=data_config.get("split_mode", "hash"),
        )
    return train_texts, val_texts


def split_labeled_train_val(
    samples: list[tuple[str, str]],
    val_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
    split_mode: str = "hash",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if val_ratio <= 0 or len(samples) < 2:
        return samples, []

    if split_mode == "hash":
        val_indices = {
            idx
            for idx, (text, _) in enumerate(samples)
            if route_to_validation(text, val_ratio, seed, split_mode="hash")
        }
        if not val_indices:
            # The normal hash route already keeps byte-identical texts in one
            # split. Preserve that invariant in the small-dataset fallback:
            # moving just one duplicate would leak the same example into both
            # train and validation.
            fallback_text = min(
                {text for text, _ in samples},
                key=lambda text: (stable_hash_fraction(text, seed=seed), text),
            )
            val_indices = {
                idx for idx, (text, _) in enumerate(samples) if text == fallback_text
            }
        train_samples = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
        val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
        if not train_samples:
            return samples, []
        return train_samples, val_samples

    if split_mode != "random":
        raise ValueError("split_mode must be one of: hash, random")

    indices = list(range(len(samples)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    val_size = max(1, int(len(samples) * val_ratio))
    val_indices = set(indices[:val_size])
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train_samples, val_samples


def should_use_labeled_dataset(data_config: dict) -> bool:
    jsonl_format = str(data_config.get("jsonl_format", "text")).lower()
    train_on_inputs = bool(data_config.get("train_on_inputs", jsonl_format == "text"))
    return jsonl_format in {"messages", "instruction", "source_target"} and not train_on_inputs


def load_labeled_train_val(
    data_config: dict,
    config: dict,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    jsonl_format = str(data_config.get("jsonl_format", "messages")).lower()
    train_samples = load_labeled_texts(
        data_config.get("train_paths"),
        jsonl_text_key=data_config.get("jsonl_text_key", "text"),
        jsonl_format=jsonl_format,
        instruction_key=data_config.get("instruction_key", "instruction"),
        input_key=data_config.get("input_key", "input"),
        output_key=data_config.get("output_key", "output"),
        source_key=data_config.get("source_key", "source"),
        target_key=data_config.get("target_key", "target"),
        target_separator=data_config.get("target_separator", "<|im_start|>"),
        source_label=data_config.get("source_label", ""),
        target_label=data_config.get("target_label"),
        chat_template=data_config.get("chat_template", "chatml"),
        system_prompt=data_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        instruction_template=data_config.get(
            "instruction_template",
            "用户：{instruction}\n输入：{input}\n助手：{output}",
        ),
        instruction_template_no_input=data_config.get(
            "instruction_template_no_input",
            "用户：{instruction}\n助手：{output}",
        ),
        txt_split=data_config.get("txt_split", "line"),
        min_chars=int(data_config.get("min_chars", 1)),
        label_mask_for_non_messages=jsonl_format == "instruction",
    )

    val_paths = as_list(data_config.get("val_paths"))
    if val_paths:
        val_samples = load_labeled_texts(
            val_paths,
            jsonl_text_key=data_config.get("jsonl_text_key", "text"),
            jsonl_format=jsonl_format,
            instruction_key=data_config.get("instruction_key", "instruction"),
            input_key=data_config.get("input_key", "input"),
            output_key=data_config.get("output_key", "output"),
            source_key=data_config.get("source_key", "source"),
            target_key=data_config.get("target_key", "target"),
            target_separator=data_config.get("target_separator", "<|im_start|>"),
            source_label=data_config.get("source_label", ""),
            target_label=data_config.get("target_label"),
            chat_template=data_config.get("chat_template", "chatml"),
            system_prompt=data_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            instruction_template=data_config.get(
                "instruction_template",
                "用户：{instruction}\n输入：{input}\n助手：{output}",
            ),
            instruction_template_no_input=data_config.get(
                "instruction_template_no_input",
                "用户：{instruction}\n助手：{output}",
            ),
            txt_split=data_config.get("txt_split", "line"),
            min_chars=int(data_config.get("min_chars", 1)),
            label_mask_for_non_messages=jsonl_format == "instruction",
        )
    else:
        train_samples, val_samples = split_labeled_train_val(
            train_samples,
            val_ratio=float(data_config.get("validation_split", 0.05)),
            seed=int(config.get("seed", 42)),
            shuffle=bool(data_config.get("shuffle_split", True)),
            split_mode=data_config.get("split_mode", "hash"),
        )
    return train_samples, val_samples


def build_datasets(
    data_config: dict,
    config: dict,
    tokenizer: CharacterTokenizer,
    max_seq_len: int,
    tokenizer_path: str | Path,
):
    stride = data_config.get("stride")
    add_bos = bool(data_config.get("add_bos", True))
    ignore_index = int(config.get("training", {}).get("ignore_index", -100))
    token_cache_dir = data_config.get("token_cache_dir")
    use_labeled_dataset = should_use_labeled_dataset(data_config)
    if token_cache_dir:
        if use_labeled_dataset:
            raise ValueError(
                "Target-only loss for messages/instruction/source_target data requires raw data loading. "
                "Unset data.token_cache_dir, or set data.train_on_inputs=true for full-transcript LM."
            )
        cache_dir = Path(token_cache_dir)
        strict_meta = bool(data_config.get("strict_tokenizer_match", True))
        tokenizer_digest = file_sha256(tokenizer_path) if strict_meta else None
        train_token_paths = resolve_token_paths(
            cache_dir,
            data_config.get("train_token_files"),
            data_config.get("train_token_file", "train.bin"),
        )
        if not train_token_paths:
            raise FileNotFoundError(f"No train token cache files found in {cache_dir}")
        train_dataset = build_token_dataset(
            train_token_paths,
            max_seq_len=max_seq_len,
            stride=stride,
            tokenizer=tokenizer,
            tokenizer_digest=tokenizer_digest,
            add_bos=add_bos,
            strict_meta=strict_meta,
            data_config=data_config,
            config=config,
        )
        val_token_paths = [
            path
            for path in resolve_token_paths(
                cache_dir,
                data_config.get("val_token_files"),
                data_config.get("val_token_file", "val.bin"),
            )
            if token_cache_has_full_window(path, max_seq_len=max_seq_len)
        ]
        val_dataset = (
            build_token_dataset(
                val_token_paths,
                max_seq_len=max_seq_len,
                stride=stride,
                tokenizer=tokenizer,
                tokenizer_digest=tokenizer_digest,
                add_bos=add_bos,
                strict_meta=strict_meta,
                data_config=data_config,
                config=config,
            )
            if val_token_paths
            else None
        )
        return train_dataset, val_dataset

    if use_labeled_dataset:
        train_samples, val_samples = load_labeled_train_val(data_config, config)
        train_dataset = LabeledLanguageModelingDataset(
            train_samples,
            tokenizer,
            max_seq_len=max_seq_len,
            stride=stride,
            add_bos=add_bos,
            label_mask_policy=data_config.get("label_mask_policy", "any"),
            ignore_index=ignore_index,
            pack_sequences=bool(data_config.get("pack_sequences", False)),
        )
        val_dataset = (
            LabeledLanguageModelingDataset(
                val_samples,
                tokenizer,
                max_seq_len=max_seq_len,
                stride=stride,
                add_bos=add_bos,
                label_mask_policy=data_config.get("label_mask_policy", "any"),
                ignore_index=ignore_index,
                pack_sequences=bool(data_config.get("pack_sequences", False)),
            )
            if val_samples
            else None
        )
        return train_dataset, val_dataset

    train_texts, val_texts = load_plain_texts(data_config, config)
    train_dataset = LanguageModelingDataset(
        train_texts,
        tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
        add_bos=add_bos,
        ignore_index=ignore_index,
    )
    val_dataset = (
        LanguageModelingDataset(
            val_texts,
            tokenizer,
            max_seq_len=max_seq_len,
            stride=stride,
            add_bos=add_bos,
            ignore_index=ignore_index,
        )
        if val_texts
        else None
    )
    return train_dataset, val_dataset


def run_training(config_file: str | Path) -> None:
    """Run the shared language-model training engine."""

    config_path = resolve_path(config_file)
    config = load_config(config_path or config_file)
    validate_pretrain_config(config)
    config["__config_path__"] = str(config_path or config_file)
    distributed = init_distributed(config.get("device", "auto"))
    set_seed(int(config.get("seed", 42)))
    enable_torch_backends()

    device = distributed.device
    data_config = resolve_config_paths_in_data(config.get("data", {}), config_path or config_file)
    config["data"] = data_config
    training_config = dict(config.get("training", {}))
    training_config["output_dir"] = str(
        resolve_output_path(training_config.get("output_dir", "outputs/run"), base=ROOT)
    )
    for checkpoint_key in ("resume_from", "init_from"):
        checkpoint_value = training_config.get(checkpoint_key)
        if checkpoint_value:
            training_config[checkpoint_key] = str(
                resolve_config_path(checkpoint_value, config_path=config_path)
            )
    config["training"] = training_config
    output_dir = Path(training_config["output_dir"])
    output_preparation = prepare_training_output_dir(
        output_dir,
        resume_from=training_config.get("resume_from"),
    )
    barrier(distributed)
    tokenizer_path = resolve_config_path(
        config.get("tokenizer", {}).get("path", "outputs/tokenizer/bpe_merged_24k.json"),
        config_path=config_path,
    )
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    tokenizer_config = dict(config.get("tokenizer", {}))
    tokenizer_config["path"] = str(tokenizer_path)
    tokenizer_config["sha256"] = file_sha256(tokenizer_path)
    config["tokenizer"] = tokenizer_config

    config, scaling_result = apply_auto_scaling(
        config,
        vocab_size=tokenizer.vocab_size,
        world_size=distributed.world_size,
    )
    if scaling_result.enabled:
        config["auto_scale"] = scaling_result.to_dict()
        training_config = dict(config.get("training", {}))

    model_config_dict = dict(config.get("model", {}))
    boundary_settings = resolve_document_boundary_settings(data_config)
    if "document_attention_backend" not in model_config_dict:
        model_config_dict["document_attention_backend"] = (
            "varlen" if boundary_settings.policy == "strict_varlen" else "dense"
        )
    model_config_dict["vocab_size"] = tokenizer.vocab_size
    config["model"] = model_config_dict
    model = GPTLanguageModel(GPTConfig.from_dict(model_config_dict))

    resume_checkpoint = None
    if output_preparation.mode == "resume":
        assert output_preparation.resume_from is not None
        resume_checkpoint = load_checkpoint(output_preparation.resume_from, map_location="cpu")
        validate_resume_model_config(resume_checkpoint, model)
        resume_precision = resolve_precision(training_config, device.type)
        validate_resume_training_state(
            resume_checkpoint,
            training_config,
            world_size=distributed.world_size,
            precision_name=resume_precision.name,
            seed=int(config.get("seed", 42)),
        )
    barrier(distributed)

    diagnostics_path = None
    manifest_path = None
    try:
        max_seq_len = int(model_config_dict.get("max_seq_len", 512))
        train_dataset, val_dataset = build_datasets(
            data_config,
            config,
            tokenizer,
            max_seq_len,
            tokenizer_path=tokenizer_path,
        )

        batch_size = int(training_config.get("batch_size", 8))
        eval_batch_size = int(training_config.get("eval_batch_size", batch_size))
        num_workers = int(training_config.get("num_workers", get_num_workers(0)))
        eval_num_workers = int(training_config.get("eval_num_workers", num_workers))
        seed = int(config.get("seed", 42))
        train_loader = make_loader(
            train_dataset,
            batch_size,
            num_workers,
            shuffle=True,
            device=device,
            seed=seed,
            distributed=distributed,
            for_training=True,
        )
        eval_shuffle = bool(
            training_config.get(
                "eval_shuffle",
                training_config.get("eval_batches") is not None,
            )
        )
        val_loader = (
            make_loader(
                val_dataset,
                eval_batch_size,
                eval_num_workers,
                shuffle=eval_shuffle,
                device=device,
                seed=seed,
                distributed=distributed,
                for_training=False,
                max_batches=(
                    int(training_config["eval_batches"])
                    if training_config.get("eval_batches") is not None
                    else None
                ),
            )
            if val_dataset
            else None
        )

        # The exact dataset identity is only available after construction.
        # Validate it before archiving or overwriting any canonical run
        # metadata, so a bad resume attempt is genuinely non-mutating.
        if resume_checkpoint is not None:
            current_run_identity = build_training_identity(config, train_loader.dataset)
            resume_precision = resolve_precision(training_config, device.type)
            validate_resume_training_state(
                resume_checkpoint,
                training_config,
                world_size=distributed.world_size,
                precision_name=resume_precision.name,
                seed=int(config.get("seed", 42)),
                current_run_identity=current_run_identity,
            )
            if distributed.is_main_process:
                output_preparation = archive_resume_metadata(output_preparation)
            barrier(distributed)

        if bool(training_config.get("write_diagnostics", True)):
            if distributed.is_main_process:
                diagnostic_samples = int(
                    training_config.get(
                        "diagnostic_samples",
                        data_config.get("diagnostic_samples", 128),
                    )
                )
                diagnostics_path, diagnostics = write_pretrain_diagnostics(
                    config,
                    output_dir=training_config.get("output_dir", "outputs/run"),
                    max_samples=diagnostic_samples,
                    world_size=distributed.world_size,
                )
                warnings = diagnostics.get("algorithm_warnings") or []
                print(f"pretrain diagnostics saved to: {diagnostics_path}", flush=True)
                for warning in warnings:
                    print(f"diagnostic warning: {warning}", flush=True)
            barrier(distributed)

        if distributed.is_main_process and bool(
            training_config.get("write_run_manifest", True)
        ):
            manifest_path = write_initial_run_manifest(
                output_dir=training_config.get("output_dir", "outputs/run"),
                config=config,
                diagnostics_path=diagnostics_path,
                cwd=ROOT,
                command=sys.argv,
            )

        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            config,
            device=device,
            pad_token_id=tokenizer.pad_id,
            distributed=distributed,
            tokenizer=tokenizer,
            output_preparation=output_preparation,
            resume_checkpoint=resume_checkpoint,
        )
        result = trainer.fit()
        if distributed.is_main_process and manifest_path is not None:
            update_run_manifest(
                manifest_path,
                updates={
                    "training_result": result,
                    "final_checkpoint": result.get("final_checkpoint"),
                    "global_step": result.get("global_step"),
                    "seen_train_tokens": result.get("seen_train_tokens"),
                    "best_val_loss": result.get("best_val_loss"),
                    "final_val_loss": result.get("final_val_loss"),
                },
                status=str(result.get("status") or "completed"),
            )
    except Exception as exc:
        if distributed.is_main_process and manifest_path is not None:
            update_run_manifest(
                manifest_path,
                updates={
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                },
                status="failed",
            )
        raise


def main() -> None:
    args = parse_args()
    try:
        run_training(args.config)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
