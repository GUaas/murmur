from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.dataset import token_cache_dtype
from muddywater.templates import DEFAULT_SYSTEM_PROMPT
from muddywater.tokenizer import CharacterTokenizer
from muddywater.utils import file_sha256
from scripts.prepare_pretrain_cache import (
    TokenShardWriter,
    encode_document,
    iter_text_records,
    write_meta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build token cache from already-fixed train and validation JSONL splits."
    )
    parser.add_argument("--train-input", nargs="+", required=True)
    parser.add_argument("--val-input", nargs="*", default=[])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jsonl-text-key", default="text")
    parser.add_argument("--jsonl-format", choices=["text", "instruction", "messages"], default="text")
    parser.add_argument("--instruction-key", default="instruction")
    parser.add_argument("--input-key", default="input")
    parser.add_argument("--output-key", default="output")
    parser.add_argument("--chat-template", default="chatml")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-tokens-per-shard", type=int, default=0)
    parser.add_argument("--add-bos", dest="add_bos", action="store_true", default=True)
    parser.add_argument("--no-add-bos", dest="add_bos", action="store_false")
    parser.add_argument("--progress", type=int, default=2000)
    return parser.parse_args()


def build_common_meta(
    args: argparse.Namespace,
    tokenizer: CharacterTokenizer,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    dtype,
) -> dict:
    return {
        "dtype": str(dtype),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_type": tokenizer.__class__.__name__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": [*map(str, args.train_input), *map(str, args.val_input)],
        "split_inputs": {
            "train": [*map(str, args.train_input)],
            "val": [*map(str, args.val_input)],
        },
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "add_bos": bool(args.add_bos),
        "jsonl_format": args.jsonl_format,
        "jsonl_text_key": args.jsonl_text_key,
        "chat_template": args.chat_template,
        "max_tokens_per_shard": int(args.max_tokens_per_shard),
    }


def encode_split(
    split_name: str,
    inputs: list[str],
    writer: TokenShardWriter,
    tokenizer: CharacterTokenizer,
    args: argparse.Namespace,
) -> dict:
    stats = {
        "documents": 0,
        "tokens": 0,
        "unk_tokens": 0,
        "unk_documents": 0,
    }
    if not inputs:
        return stats

    unk_id = getattr(tokenizer, "unk_id", None)
    started = time.time()
    for text, _source in iter_text_records(
        inputs,
        args.jsonl_text_key,
        args.jsonl_format,
        args.min_chars,
        instruction_key=args.instruction_key,
        input_key=args.input_key,
        output_key=args.output_key,
        chat_template=args.chat_template,
        system_prompt=args.system_prompt,
    ):
        ids = encode_document(tokenizer, text, add_bos=args.add_bos)
        if unk_id is not None:
            unk_count = ids.count(unk_id)
            if unk_count:
                stats["unk_tokens"] += unk_count
                stats["unk_documents"] += 1
        stats["documents"] += 1
        stats["tokens"] += writer.append_document(ids)
        if args.progress > 0 and stats["documents"] % args.progress == 0:
            elapsed = time.time() - started
            print(
                f"{split_name}_docs={stats['documents']} {split_name}_tokens={stats['tokens']} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return stats


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 8) if denominator else None


def main() -> None:
    args = parse_args()
    tokenizer_path = Path(args.tokenizer)
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    tokenizer_sha256 = file_sha256(tokenizer_path)
    dtype = token_cache_dtype(tokenizer.vocab_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_meta = build_common_meta(args, tokenizer, tokenizer_path, tokenizer_sha256, dtype)
    train_writer = TokenShardWriter(
        output_dir=output_dir,
        prefix="train",
        dtype=dtype,
        common_meta={**common_meta, "split": "train"},
        max_tokens_per_shard=args.max_tokens_per_shard,
    )
    val_writer = TokenShardWriter(
        output_dir=output_dir,
        prefix="val",
        dtype=dtype,
        common_meta={**common_meta, "split": "val"},
        max_tokens_per_shard=args.max_tokens_per_shard,
    )

    try:
        train_stats = encode_split("train", args.train_input, train_writer, tokenizer, args)
        val_stats = encode_split("val", args.val_input, val_writer, tokenizer, args)
    finally:
        train_writer.close()
        val_writer.close()

    total_tokens = int(train_stats["tokens"] + val_stats["tokens"])
    total_documents = int(train_stats["documents"] + val_stats["documents"])
    manifest = {
        **common_meta,
        "train_shards": train_writer.shards,
        "val_shards": val_writer.shards,
        "train_tokens": train_stats["tokens"],
        "val_tokens": val_stats["tokens"],
        "train_documents": train_stats["documents"],
        "val_documents": val_stats["documents"],
        "actual_val_token_ratio": ratio(val_stats["tokens"], total_tokens),
        "actual_val_document_ratio": ratio(val_stats["documents"], total_documents),
    }
    stats = {
        **manifest,
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    write_meta(output_dir / "manifest.json", manifest)
    write_meta(output_dir / "prepare_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
