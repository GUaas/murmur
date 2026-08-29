from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.dataset import discover_data_files, token_cache_dtype
from muddywater.packing import route_to_validation, stable_hash_fraction
from muddywater.templates import DEFAULT_SYSTEM_PROMPT, format_messages, format_sft_example
from muddywater.tokenizer import CharacterTokenizer
from muddywater.utils import atomic_write_text, file_sha256


CACHE_ARTIFACT_PATTERNS = (
    "train*.bin",
    "val*.bin",
    "*.bin.meta.json",
    "*.bin.doc_starts.npy",
    "manifest.json",
    "prepare_stats.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert text JSONL/TXT into EOS-packed token-id caches for pretraining."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input .jsonl/.txt paths or globs.")
    parser.add_argument(
        "--val-input",
        nargs="*",
        default=None,
        help="Optional explicit validation .jsonl/.txt paths. When set, --input is written fully to train.",
    )
    parser.add_argument("--tokenizer", required=True, help="Path to SentencePiece BPE tokenizer (.model or .json).")
    parser.add_argument("--output-dir", required=True, help="Directory for train.bin and val.bin.")
    parser.add_argument("--jsonl-text-key", default="text")
    parser.add_argument("--jsonl-format", choices=["text", "instruction", "messages"], default="text")
    parser.add_argument("--instruction-key", default="instruction")
    parser.add_argument("--input-key", default="input")
    parser.add_argument("--output-key", default="output")
    parser.add_argument("--chat-template", default="chatml")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--split-source-key",
        default=None,
        help="Optional JSONL field used with text for stable train/val routing.",
    )
    parser.add_argument(
        "--split-unit",
        choices=["auto", "text", "source", "source_text"],
        default="auto",
        help=(
            "Hashing unit for train/val split. auto preserves legacy behavior: "
            "text normally, source_text when --split-source-key is set."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["hash", "random"],
        default="hash",
        help="Use hash for order-independent train/val splits, or random for legacy behavior.",
    )
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument(
        "--max-tokens-per-shard",
        type=int,
        default=0,
        help=(
            "Split train/val caches into prefix_00000.bin shards after roughly this many "
            "tokens. Leave 0 to keep legacy train.bin/val.bin outputs."
        ),
    )
    parser.add_argument(
        "--add-bos",
        dest="add_bos",
        action="store_true",
        default=True,
        help="Add BOS before every document. Enabled by default.",
    )
    parser.add_argument(
        "--no-add-bos",
        dest="add_bos",
        action="store_false",
        help="Disable BOS insertion for compatibility with older caches.",
    )
    parser.add_argument("--progress", type=int, default=100000)
    return parser.parse_args()


def field_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(field_to_text(item) for item in value)
    return str(value)


def iter_text_records(
    paths: Iterable[str],
    jsonl_text_key: str,
    jsonl_format: str,
    min_chars: int,
    split_source_key: str | None = None,
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
    chat_template: str = "chatml",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Iterable[tuple[str, str]]:
    for path in discover_data_files(paths):
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                    if jsonl_format == "text":
                        text = field_to_text(obj.get(jsonl_text_key, ""))
                    elif jsonl_format == "instruction":
                        if instruction_key not in obj or output_key not in obj:
                            raise KeyError(
                                f"Missing instruction/output keys at {path}:{line_no}: "
                                f"{instruction_key!r}, {output_key!r}"
                            )
                        text = format_sft_example(
                            instruction=field_to_text(obj.get(instruction_key, "")),
                            input_text=field_to_text(obj.get(input_key, "")),
                            output=field_to_text(obj.get(output_key, "")),
                            chat_template=chat_template,
                            system_prompt=system_prompt,
                        )
                    elif jsonl_format == "messages":
                        messages = obj.get("messages")
                        if not isinstance(messages, list) or not messages:
                            raise ValueError(f"Expected non-empty messages list at {path}:{line_no}")
                        text = format_messages(
                            messages=messages,
                            chat_template=chat_template,
                            system_prompt=system_prompt,
                        )
                    else:
                        raise ValueError("jsonl_format must be one of: text, instruction, messages")

                    if isinstance(text, str) and len(text) >= min_chars and text.strip():
                        source = (
                            str(obj.get(split_source_key, path))
                            if split_source_key
                            else str(path)
                        )
                        yield text.strip(), source
        elif suffix == ".txt":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if len(text) >= min_chars:
                        yield text, str(path)
        else:
            raise ValueError(f"Unsupported input file: {path}")


def iter_texts(paths: Iterable[str], jsonl_text_key: str, min_chars: int) -> Iterable[str]:
    for text, _ in iter_text_records(paths, jsonl_text_key, "text", min_chars):
        yield text


def append_ids(handle, ids: list[int], dtype: np.dtype) -> int:
    if not ids:
        return 0
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(handle)
    return int(arr.size)


def encode_document(tokenizer: CharacterTokenizer, text: str, add_bos: bool = True) -> list[int]:
    return tokenizer.encode(text, add_bos=add_bos, add_eos=True)


def append_encoded_document(
    writer: "TokenShardWriter",
    tokenizer: CharacterTokenizer,
    text: str,
    add_bos: bool,
    unk_id: int | None,
) -> tuple[int, int]:
    ids = encode_document(tokenizer, text, add_bos=add_bos)
    unk_count = ids.count(unk_id) if unk_id is not None else 0
    written = writer.append_document(ids)
    return written, unk_count


def resolve_split_unit(split_source_key: str | None = None, split_unit: str = "auto") -> str:
    unit = str(split_unit or "auto").strip().lower()
    if unit == "auto":
        return "source_text" if split_source_key else "text"
    if unit not in {"text", "source", "source_text"}:
        raise ValueError("split_unit must be one of: auto, text, source, source_text")
    return unit


def resolve_split_route_text(
    text: str,
    source: str,
    split_source_key: str | None = None,
    split_unit: str = "auto",
) -> str:
    unit = resolve_split_unit(split_source_key=split_source_key, split_unit=split_unit)
    if unit == "text":
        return text
    if unit == "source":
        return source
    if unit == "source_text":
        return f"{source}\0{text}"
    raise AssertionError(f"Unexpected split unit: {unit}")


def find_hash_split_overrides(
    route_texts: Iterable[str],
    val_ratio: float,
    seed: int,
) -> dict[str, bool]:
    """Find whole-group overrides only when a hash split would be empty.

    The scan stops as soon as both splits are represented, so normal large
    corpora pay only a small prefix read.  For tiny corpora, the selected
    fallback key is overridden as a complete duplicate/source group.
    """

    first_key: str | None = None
    has_second_group = False
    has_train = False
    has_validation = False
    lowest: tuple[float, str] | None = None
    highest: tuple[float, str] | None = None
    for raw_key in route_texts:
        key = str(raw_key)
        if first_key is None:
            first_key = key
        elif key != first_key:
            has_second_group = True
        fraction = stable_hash_fraction(key, seed=seed)
        is_validation = fraction < float(val_ratio)
        has_validation = has_validation or is_validation
        has_train = has_train or not is_validation
        candidate = (fraction, key)
        if lowest is None or candidate < lowest:
            lowest = candidate
        if highest is None or candidate > highest:
            highest = candidate
        if has_train and has_validation:
            return {}

    if first_key is None:
        return {}
    if not has_second_group:
        # A single group cannot form two leak-free splits; preserve training.
        return {first_key: False} if not has_train else {}
    if not has_validation and lowest is not None and float(val_ratio) > 0:
        return {lowest[1]: True}
    if not has_train and highest is not None:
        return {highest[1]: False}
    return {}


def iter_split_route_texts(args: argparse.Namespace) -> Iterable[str]:
    for text, source in iter_text_records(
        args.input,
        args.jsonl_text_key,
        args.jsonl_format,
        args.min_chars,
        split_source_key=args.split_source_key,
        instruction_key=args.instruction_key,
        input_key=args.input_key,
        output_key=args.output_key,
        chat_template=args.chat_template,
        system_prompt=args.system_prompt,
    ):
        yield resolve_split_route_text(
            text=text,
            source=source,
            split_source_key=args.split_source_key,
            split_unit=args.split_unit,
        )


def write_meta(path: Path, payload: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
        overwrite=True,
    )


def ensure_cache_output_is_clean(output_dir: Path) -> None:
    """Refuse to mix a new tokenization pass with stale cache artifacts."""

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return
    artifacts = sorted(
        {
            path.resolve()
            for pattern in CACHE_ARTIFACT_PATTERNS
            for path in output_dir.glob(pattern)
            if path.is_file()
        }
    )
    if artifacts:
        preview = ", ".join(path.name for path in artifacts[:5])
        raise FileExistsError(
            f"Token-cache output directory already contains cache artifacts: {preview}. "
            "Use a new empty staging directory so old and new shards cannot be mixed."
        )


def write_doc_starts(path: Path, starts: list[int]) -> None:
    np.save(path, np.asarray(starts, dtype=np.uint64))


class TokenShardWriter:
    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        dtype: np.dtype,
        common_meta: dict,
        max_tokens_per_shard: int = 0,
    ) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.dtype = dtype
        self.common_meta = dict(common_meta)
        self.max_tokens_per_shard = max(0, int(max_tokens_per_shard))
        self.shards: list[dict] = []
        self.total_tokens = 0
        self.total_documents = 0
        self._index = 0
        self._handle = None
        self._path: Path | None = None
        self._doc_starts: list[int] = []
        self._tokens = 0
        self._documents = 0
        self._open_next()

    def _next_path(self) -> Path:
        if self.max_tokens_per_shard > 0:
            return self.output_dir / f"{self.prefix}_{self._index:05d}.bin"
        return self.output_dir / f"{self.prefix}.bin"

    def _open_next(self) -> None:
        self._path = self._next_path()
        self._handle = self._path.open("wb")
        self._doc_starts = []
        self._tokens = 0
        self._documents = 0

    def _close_current(self) -> None:
        if self._handle is None or self._path is None:
            return
        self._handle.close()
        doc_starts_path = self._path.with_suffix(self._path.suffix + ".doc_starts.npy")
        write_doc_starts(doc_starts_path, self._doc_starts)
        write_meta(
            self._path.with_suffix(self._path.suffix + ".meta.json"),
            {
                **self.common_meta,
                "num_tokens": self._tokens,
                "num_documents": self._documents,
                "doc_starts_file": doc_starts_path.name,
                "shard_index": self._index,
            },
        )
        self.shards.append(
            {
                "file": self._path.name,
                "num_tokens": self._tokens,
                "num_documents": self._documents,
                "doc_starts_file": doc_starts_path.name,
            }
        )
        self._handle = None

    def append_document(self, ids: list[int]) -> int:
        if not ids:
            return 0
        if (
            self.max_tokens_per_shard > 0
            and self._tokens > 0
            and self._tokens + len(ids) > self.max_tokens_per_shard
        ):
            self._close_current()
            self._index += 1
            self._open_next()
        assert self._handle is not None
        self._doc_starts.append(self._tokens)
        written = append_ids(self._handle, ids, self.dtype)
        self._tokens += written
        self._documents += 1
        self.total_tokens += written
        self.total_documents += 1
        return written

    def close(self) -> None:
        self._close_current()


def main() -> None:
    args = parse_args()
    tokenizer = CharacterTokenizer.load(args.tokenizer)
    tokenizer_path = Path(args.tokenizer)
    tokenizer_sha256 = file_sha256(tokenizer_path)
    dtype = token_cache_dtype(tokenizer.vocab_size)
    output_dir = Path(args.output_dir)
    ensure_cache_output_is_clean(output_dir)

    rng = random.Random(args.seed)
    effective_split_unit = resolve_split_unit(
        split_source_key=args.split_source_key,
        split_unit=args.split_unit,
    )
    random_source_assignments: dict[str, bool] = {}
    hash_split_overrides = (
        find_hash_split_overrides(
            iter_split_route_texts(args),
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        if not args.val_input and args.split_mode == "hash"
        else {}
    )
    started = time.time()

    stats = {
        "documents": 0,
        "train_documents": 0,
        "val_documents": 0,
        "train_tokens": 0,
        "val_tokens": 0,
        "unk_tokens": 0,
        "unk_documents": 0,
        "dtype": str(dtype),
        "tokenizer_type": tokenizer.__class__.__name__,
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_sha256": tokenizer_sha256,
        "train_input": [str(path) for path in args.input],
        "val_input": [str(path) for path in args.val_input] if args.val_input else [],
        "explicit_val_input": bool(args.val_input),
        "val_ratio": args.val_ratio,
        "split_mode": args.split_mode,
        "split_source_key": args.split_source_key,
        "split_unit": args.split_unit,
        "effective_split_unit": effective_split_unit,
        "hash_split_fallback_groups": len(hash_split_overrides),
        "jsonl_format": args.jsonl_format,
        "jsonl_text_key": args.jsonl_text_key,
        "chat_template": args.chat_template,
        "add_bos": bool(args.add_bos),
        "max_tokens_per_shard": int(args.max_tokens_per_shard),
    }
    unk_id = getattr(tokenizer, "unk_id", None)

    common_meta = {
        "dtype": str(dtype),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_type": tokenizer.__class__.__name__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": [str(path) for path in args.input],
        "train_input": [str(path) for path in args.input],
        "val_input": [str(path) for path in args.val_input] if args.val_input else [],
        "explicit_val_input": bool(args.val_input),
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_sha256": tokenizer_sha256,
        "add_bos": bool(args.add_bos),
        "split_mode": args.split_mode,
        "split_source_key": args.split_source_key,
        "split_unit": args.split_unit,
        "effective_split_unit": effective_split_unit,
        "jsonl_format": args.jsonl_format,
        "jsonl_text_key": args.jsonl_text_key,
        "chat_template": args.chat_template,
        "max_tokens_per_shard": int(args.max_tokens_per_shard),
    }
    train_writer = TokenShardWriter(
        output_dir=output_dir,
        prefix="train",
        dtype=dtype,
        common_meta=common_meta,
        max_tokens_per_shard=args.max_tokens_per_shard,
    )
    val_writer = TokenShardWriter(
        output_dir=output_dir,
        prefix="val",
        dtype=dtype,
        common_meta=common_meta,
        max_tokens_per_shard=args.max_tokens_per_shard,
    )
    try:
        if args.val_input:
            for split_name, paths, writer in (
                ("train", args.input, train_writer),
                ("val", args.val_input, val_writer),
            ):
                for text, _source in iter_text_records(
                    paths,
                    args.jsonl_text_key,
                    args.jsonl_format,
                    args.min_chars,
                    split_source_key=args.split_source_key,
                    instruction_key=args.instruction_key,
                    input_key=args.input_key,
                    output_key=args.output_key,
                    chat_template=args.chat_template,
                    system_prompt=args.system_prompt,
                ):
                    stats["documents"] += 1
                    stats[f"{split_name}_documents"] += 1
                    written, unk_count = append_encoded_document(
                        writer=writer,
                        tokenizer=tokenizer,
                        text=text,
                        add_bos=args.add_bos,
                        unk_id=unk_id,
                    )
                    stats[f"{split_name}_tokens"] += written
                    if unk_count:
                        stats["unk_tokens"] += unk_count
                        stats["unk_documents"] += 1
                    if args.progress and stats["documents"] % args.progress == 0:
                        elapsed = time.time() - started
                        print(
                            "docs={documents} train_tokens={train_tokens} "
                            "val_tokens={val_tokens} unk_tokens={unk_tokens} "
                            "elapsed={elapsed:.1f}s".format(
                                elapsed=elapsed,
                                **stats,
                            ),
                            flush=True,
                        )
        else:
            for text, source in iter_text_records(
                args.input,
                args.jsonl_text_key,
                args.jsonl_format,
                args.min_chars,
                split_source_key=args.split_source_key,
                instruction_key=args.instruction_key,
                input_key=args.input_key,
                output_key=args.output_key,
                chat_template=args.chat_template,
                system_prompt=args.system_prompt,
            ):
                stats["documents"] += 1
                route_text = resolve_split_route_text(
                    text=text,
                    source=source,
                    split_source_key=args.split_source_key,
                    split_unit=args.split_unit,
                )
                if args.split_mode == "random" and effective_split_unit == "source":
                    if route_text not in random_source_assignments:
                        random_source_assignments[route_text] = route_to_validation(
                            text=route_text,
                            val_ratio=args.val_ratio,
                            seed=args.seed,
                            split_mode=args.split_mode,
                            rng=rng,
                        )
                    is_validation = random_source_assignments[route_text]
                else:
                    is_validation = route_to_validation(
                        text=route_text,
                        val_ratio=args.val_ratio,
                        seed=args.seed,
                        split_mode=args.split_mode,
                        rng=rng,
                    )
                if route_text in hash_split_overrides:
                    is_validation = hash_split_overrides[route_text]
                split_name = "val" if is_validation else "train"
                writer = val_writer if is_validation else train_writer
                stats[f"{split_name}_documents"] += 1
                written, unk_count = append_encoded_document(
                    writer=writer,
                    tokenizer=tokenizer,
                    text=text,
                    add_bos=args.add_bos,
                    unk_id=unk_id,
                )
                stats[f"{split_name}_tokens"] += written
                if unk_count:
                    stats["unk_tokens"] += unk_count
                    stats["unk_documents"] += 1

                if args.progress and stats["documents"] % args.progress == 0:
                    elapsed = time.time() - started
                    print(
                        "docs={documents} train_tokens={train_tokens} "
                        "val_tokens={val_tokens} unk_tokens={unk_tokens} "
                        "elapsed={elapsed:.1f}s".format(
                            elapsed=elapsed,
                            **stats,
                        ),
                        flush=True,
                    )
    finally:
        train_writer.close()
        val_writer.close()

    stats["train_shards"] = train_writer.shards
    stats["val_shards"] = val_writer.shards
    total_tokens = int(stats["train_tokens"] + stats["val_tokens"])
    total_documents = int(stats["train_documents"] + stats["val_documents"])
    stats["actual_val_token_ratio"] = (
        round(stats["val_tokens"] / total_tokens, 8) if total_tokens else None
    )
    stats["actual_val_document_ratio"] = (
        round(stats["val_documents"] / total_documents, 8) if total_documents else None
    )
    write_meta(
        output_dir / "manifest.json",
        {
            **common_meta,
            "train_shards": train_writer.shards,
            "val_shards": val_writer.shards,
            "train_tokens": stats["train_tokens"],
            "val_tokens": stats["val_tokens"],
            "actual_val_token_ratio": stats["actual_val_token_ratio"],
            "actual_val_document_ratio": stats["actual_val_document_ratio"],
            "train_documents": stats["train_documents"],
            "val_documents": stats["val_documents"],
        },
    )
    write_meta(output_dir / "prepare_stats.json", stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
