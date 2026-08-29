from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config
from muddywater.dataset import discover_data_files
from muddywater.packing import route_to_validation
from muddywater.tokenizer import BPETokenizer, prepare_sentencepiece_training_texts
from muddywater.utils import as_list


DEFAULT_USER_DEFINED_SYMBOLS = ["<|im_start|>", "<|im_end|>"]
UNICODE_FALLBACK_PROBES = ["\U00020bb7", "\U0001f642", "\u03a9", "\u200d"]
SUPPORTED_TOKENIZER_CONFIG_KEYS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE/Unigram tokenizer.")
    parser.add_argument("--config", default="configs/experiment_loss_descent_10k.yaml")
    parser.add_argument("--input", nargs="*", default=None)
    parser.add_argument(
        "--diagnostic-input",
        nargs="*",
        default=None,
        help="Optional held-out JSONL/TXT paths used only for tokenizer diagnostics.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--text-key", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--diagnostic-size", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--progress", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=None,
        help=(
            "Exclude the deterministic hash-routed validation fraction from tokenizer "
            "training. Defaults to data.validation_split, or zero."
        ),
    )
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default=None)
    parser.add_argument("--character-coverage", type=float, default=None)
    parser.add_argument("--max-sentencepiece-length", type=int, default=None)
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=None,
        help="Maximum UTF-8 byte length accepted by SentencePiece for one training text.",
    )
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--byte-fallback", dest="byte_fallback", action="store_true")
    parser.add_argument("--no-byte-fallback", dest="byte_fallback", action="store_false")
    parser.set_defaults(byte_fallback=None)
    parser.add_argument("--split-digits", dest="split_digits", action="store_true")
    parser.add_argument("--no-split-digits", dest="split_digits", action="store_false")
    parser.set_defaults(split_digits=None)
    parser.add_argument("--hard-vocab-limit", dest="hard_vocab_limit", action="store_true")
    parser.add_argument("--no-hard-vocab-limit", dest="hard_vocab_limit", action="store_false")
    parser.set_defaults(hard_vocab_limit=None)
    return parser.parse_args()


def iter_texts(paths: Iterable[str | Path], text_key: str, min_chars: int) -> Iterable[str]:
    for path in discover_data_files(paths):
        suffix = path.suffix.lower()
        if suffix == ".txt":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if len(text) >= min_chars:
                        yield text
            continue

        with path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                text = obj.get(text_key)
                if isinstance(text, str) and len(text) >= min_chars and text.strip():
                    yield text.strip()


def collect_reservoir_sample(
    paths: Iterable[str | Path],
    text_key: str,
    min_chars: int,
    sample_size: int,
    progress: int | None,
    seed: int,
    validation_ratio: float = 0.0,
    split_seed: int | None = None,
) -> tuple[list[str], int]:
    started = time.time()
    rng = random.Random(seed)
    sample: list[str] = []
    total = 0
    seen = 0
    for text in iter_texts(paths, text_key=text_key, min_chars=min_chars):
        seen += 1
        if route_to_validation(
            text=text,
            val_ratio=float(validation_ratio),
            seed=int(seed if split_seed is None else split_seed),
            split_mode="hash",
        ):
            if progress and progress > 0 and seen % progress == 0:
                elapsed = time.time() - started
                print(
                    f"sample_scan_texts={seen} train_texts={total} "
                    f"sample={len(sample)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            continue
        total += 1
        if len(sample) < sample_size:
            sample.append(text)
        else:
            replacement_index = rng.randrange(total)
            if replacement_index < sample_size:
                sample[replacement_index] = text
        if progress and progress > 0 and seen % progress == 0:
            elapsed = time.time() - started
            print(
                f"sample_scan_texts={seen} train_texts={total} "
                f"sample={len(sample)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return sample, total


def normalize_sentencepiece_model_type(value: str | None) -> str:
    normalized = str(value or "unigram").strip().lower().replace("-", "_")
    aliases = {
        "sentencepiece": "unigram",
        "sp": "unigram",
        "spm": "unigram",
        "sentencepiece_unigram": "unigram",
        "sp_unigram": "unigram",
        "sentencepiece_bpe": "bpe",
        "sp_bpe": "bpe",
    }
    model_type = aliases.get(normalized, normalized)
    if model_type not in {"bpe", "unigram"}:
        raise ValueError("tokenizer.model_type must be one of: sentencepiece_unigram, sentencepiece_bpe")
    return model_type


def ignored_tokenizer_config_keys(tokenizer_config: dict) -> list[str]:
    return sorted(key for key in tokenizer_config if key not in SUPPORTED_TOKENIZER_CONFIG_KEYS)


def resolve_user_defined_symbols(tokenizer_config: dict) -> list[str]:
    symbols = [str(symbol) for symbol in as_list(tokenizer_config.get("user_defined_symbols", DEFAULT_USER_DEFINED_SYMBOLS))]
    return [symbol for symbol in dict.fromkeys(symbols) if symbol]


def tokenizer_diagnostics(tokenizer: BPETokenizer, texts: list[str]) -> dict:
    total_chars = 0
    total_tokens = 0
    unk_tokens = 0
    empty_encodings = 0
    for text in texts:
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        total_chars += len(text)
        total_tokens += len(ids)
        if not ids:
            empty_encodings += 1
        unk_tokens += sum(1 for token_id in ids if int(token_id) == int(tokenizer.unk_id))
    return {
        "documents": len(texts),
        "chars": total_chars,
        "tokens": total_tokens,
        "chars_per_token": round(total_chars / total_tokens, 6) if total_tokens else None,
        "unk_tokens": unk_tokens,
        "unk_token_rate": round(unk_tokens / total_tokens, 8) if total_tokens else None,
        "empty_encodings": empty_encodings,
    }


def unicode_fallback_diagnostics(tokenizer: BPETokenizer) -> dict:
    probes: list[dict[str, object]] = []
    total_unk = 0
    for text in UNICODE_FALLBACK_PROBES:
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        pieces = tokenizer.processor.EncodeAsPieces(text)
        unk_tokens = sum(1 for token_id in ids if int(token_id) == int(tokenizer.unk_id))
        total_unk += unk_tokens
        probes.append(
            {
                "text": text,
                "pieces": pieces,
                "ids": [int(token_id) for token_id in ids],
                "unk_tokens": unk_tokens,
                "round_trip": tokenizer.decode(ids, skip_special_tokens=False),
            }
        )
    return {"unicode_probe_unk_tokens": total_unk, "unicode_probes": probes}


def validate_unicode_fallback(tokenizer: BPETokenizer, byte_fallback: bool) -> None:
    if not byte_fallback:
        return
    diagnostics = unicode_fallback_diagnostics(tokenizer)
    if int(diagnostics["unicode_probe_unk_tokens"]) > 0:
        raise ValueError(
            "SentencePiece tokenizer still emits <unk> for Unicode fallback probes. "
            "Keep tokenizer.byte_fallback=true and rebuild with a compatible sentencepiece version."
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config.get("data", {})
    tokenizer_config = config.get("tokenizer", {})
    ignored_config_keys = ignored_tokenizer_config_keys(tokenizer_config)

    paths = args.input or as_list(data_config.get("train_paths"))
    if not paths:
        raise ValueError("No tokenizer input paths provided.")
    diagnostic_paths = args.diagnostic_input or as_list(
        tokenizer_config.get("diagnostic_paths")
    )

    config_model_type = tokenizer_config.get("model_type", "sentencepiece_unigram")
    model_type = normalize_sentencepiece_model_type(args.model_type or config_model_type)
    output = Path(args.output or tokenizer_config.get("path", f"outputs/tokenizer/sp_{model_type}_32k.model"))
    vocab_size = int(args.vocab_size or tokenizer_config.get("vocab_size", 32000))
    sample_size = int(args.sample_size or tokenizer_config.get("sample_size", 1000000))
    diagnostic_size = int(args.diagnostic_size or tokenizer_config.get("diagnostic_size", min(sample_size, 50000)))
    text_key = args.text_key or data_config.get("jsonl_text_key", "text")
    min_chars = int(args.min_chars or data_config.get("min_chars", 1))
    progress = int(args.progress or tokenizer_config.get("progress", 100000))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    validation_ratio = float(
        args.validation_ratio
        if args.validation_ratio is not None
        else data_config.get("validation_split", 0.0)
    )
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation ratio used for tokenizer training must be in [0, 1)")
    character_coverage = float(
        args.character_coverage
        if args.character_coverage is not None
        else tokenizer_config.get("character_coverage", 0.9995)
    )
    max_sentencepiece_length = int(
        args.max_sentencepiece_length
        if args.max_sentencepiece_length is not None
        else tokenizer_config.get("max_sentencepiece_length", 16)
    )
    max_sentence_length = int(
        args.max_sentence_length
        if args.max_sentence_length is not None
        else tokenizer_config.get("max_sentence_length", 16384)
    )
    num_threads = int(args.num_threads if args.num_threads is not None else tokenizer_config.get("num_threads", 16))
    byte_fallback = (
        bool(args.byte_fallback)
        if args.byte_fallback is not None
        else bool(tokenizer_config.get("byte_fallback", True))
    )
    split_digits = (
        bool(args.split_digits)
        if args.split_digits is not None
        else bool(tokenizer_config.get("split_digits", True))
    )
    hard_vocab_limit = (
        bool(args.hard_vocab_limit)
        if args.hard_vocab_limit is not None
        else bool(tokenizer_config.get("hard_vocab_limit", False))
    )
    normalization_rule_name = str(tokenizer_config.get("normalization_rule_name", "identity"))
    remove_extra_whitespaces = bool(tokenizer_config.get("remove_extra_whitespaces", False))
    user_defined_symbols = resolve_user_defined_symbols(tokenizer_config)

    print(
        json.dumps(
            {
                "backend": "sentencepiece",
                "inputs": [str(path) for path in paths],
                "diagnostic_inputs": [str(path) for path in diagnostic_paths],
                "output": str(output),
                "model_type": model_type,
                "vocab_size": vocab_size,
                "sample_size": sample_size,
                "diagnostic_size": diagnostic_size,
                "character_coverage": character_coverage,
                "byte_fallback": byte_fallback,
                "split_digits": split_digits,
                "hard_vocab_limit": hard_vocab_limit,
                "normalization_rule_name": normalization_rule_name,
                "remove_extra_whitespaces": remove_extra_whitespaces,
                "max_sentencepiece_length": max_sentencepiece_length,
                "max_sentence_length": max_sentence_length,
                "num_threads": num_threads,
                "user_defined_symbols": user_defined_symbols,
                "ignored_config_keys": ignored_config_keys,
                "seed": seed,
                "excluded_validation_ratio": validation_ratio,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if ignored_config_keys:
        print(
            "warning: ignored tokenizer config keys for SentencePiece: "
            + ", ".join(ignored_config_keys),
            flush=True,
        )

    sample_texts, total_texts = collect_reservoir_sample(
        paths=paths,
        text_key=text_key,
        min_chars=min_chars,
        sample_size=sample_size,
        progress=progress,
        seed=seed,
        validation_ratio=validation_ratio,
        split_seed=seed,
    )
    if not sample_texts:
        raise ValueError("No texts were collected for tokenizer training.")
    training_texts, length_diagnostics = prepare_sentencepiece_training_texts(
        sample_texts,
        max_sentence_length=max_sentence_length,
    )
    if not training_texts:
        raise ValueError("No byte-safe texts remain for tokenizer training.")

    print(
        f"training_sentencepiece model_type={model_type} source_sample={len(sample_texts)} "
        f"training_sentences={len(training_texts)} total_texts={total_texts} "
        f"chunked={length_diagnostics['chunked_documents']} "
        f"rejected={length_diagnostics['rejected_documents']}",
        flush=True,
    )
    tokenizer = BPETokenizer.train(
        texts=training_texts,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        split_digits=split_digits,
        max_sentencepiece_length=max_sentencepiece_length,
        max_sentence_length=max_sentence_length,
        hard_vocab_limit=hard_vocab_limit,
        normalization_rule_name=normalization_rule_name,
        remove_extra_whitespaces=remove_extra_whitespaces,
        user_defined_symbols=user_defined_symbols,
        num_threads=num_threads,
    )
    validate_unicode_fallback(tokenizer, byte_fallback=byte_fallback)

    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output)

    diagnostic_texts, diagnostic_total = collect_reservoir_sample(
        paths=diagnostic_paths or paths,
        text_key=text_key,
        min_chars=min_chars,
        sample_size=diagnostic_size,
        progress=None,
        seed=seed + 17,
        validation_ratio=validation_ratio,
        split_seed=seed,
    )
    diagnostics = tokenizer_diagnostics(tokenizer, diagnostic_texts or sample_texts)
    diagnostics.update(unicode_fallback_diagnostics(tokenizer))
    diagnostics.update(
        {
            "diagnostic_documents_seen": diagnostic_total,
            "diagnostic_sample_size": len(diagnostic_texts or sample_texts),
            "training_sample_size": len(sample_texts),
            "total_training_texts_seen": total_texts,
            "excluded_validation_ratio": validation_ratio,
            "validation_split_seed": seed,
            "backend": "sentencepiece",
            "model_type": model_type,
            "byte_fallback": byte_fallback,
            "split_digits": split_digits,
            "character_coverage": character_coverage,
            "max_sentencepiece_length": max_sentencepiece_length,
            "max_sentence_length": max_sentence_length,
            "sentence_length_diagnostics": length_diagnostics,
            "hard_vocab_limit": hard_vocab_limit,
            "normalization_rule_name": normalization_rule_name,
            "remove_extra_whitespaces": remove_extra_whitespaces,
            "user_defined_symbols": user_defined_symbols,
            "ignored_config_keys": ignored_config_keys,
        }
    )
    diagnostics_path = output.with_suffix(output.suffix + ".diagnostics.json")
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Tokenizer saved to: {output}", flush=True)
    print(f"Vocab size: {tokenizer.vocab_size}", flush=True)
    print(f"Tokenizer diagnostics saved to: {diagnostics_path}", flush=True)
    # Keep the machine-readable summary printable on Windows consoles whose
    # active code page cannot encode every Unicode fallback probe.
    print(json.dumps(diagnostics, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
