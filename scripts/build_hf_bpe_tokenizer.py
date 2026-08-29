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

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from muddywater.config import load_config
from muddywater.dataset import discover_data_files
from muddywater.utils import as_list


BASE_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
DEFAULT_USER_DEFINED_SYMBOLS = ["<|im_start|>", "<|im_end|>"]
UNICODE_FALLBACK_PROBES = ["\U00020bb7", "\U0001f642", "\u03a9", "\u200d"]
SUPPORTED_TOKENIZER_CONFIG_KEYS = {
    "path",
    "model_type",
    "pre_tokenizer",
    "vocab_size",
    "byte_fallback",
    "split_digits",
    "user_defined_symbols",
    "sample_size",
    "progress",
    "min_frequency",
    "diagnostic_size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fast HuggingFace BPE tokenizer.")
    parser.add_argument("--config", default="configs/experiment_loss_descent_10k.yaml")
    parser.add_argument("--input", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--text-key", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--min-frequency", type=int, default=None)
    parser.add_argument("--progress", type=int, default=None)
    parser.add_argument("--pre-tokenizer", choices=["metaspace", "bytelevel"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--diagnostic-size", type=int, default=None)
    parser.add_argument("--split-digits", dest="split_digits", action="store_true")
    parser.add_argument("--no-split-digits", dest="split_digits", action="store_false")
    parser.set_defaults(split_digits=None)
    parser.add_argument("--byte-fallback", dest="byte_fallback", action="store_true")
    parser.add_argument("--no-byte-fallback", dest="byte_fallback", action="store_false")
    parser.set_defaults(byte_fallback=None)
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


def collect_sample_and_alphabet(
    paths: Iterable[str | Path],
    text_key: str,
    min_chars: int,
    sample_size: int,
    progress: int | None,
    seed: int,
) -> tuple[list[str], list[str], int]:
    started = time.time()
    rng = random.Random(seed)
    sample: list[str] = []
    alphabet: set[str] = set()
    total = 0
    for text in iter_texts(paths, text_key=text_key, min_chars=min_chars):
        total += 1
        if len(sample) < sample_size:
            sample.append(text)
        else:
            replacement_index = rng.randrange(total)
            if replacement_index < sample_size:
                sample[replacement_index] = text
        alphabet.update(text)
        if progress and progress > 0 and total % progress == 0:
            elapsed = time.time() - started
            print(
                f"alphabet_scan_texts={total} sample={len(sample)} "
                f"alphabet={len(alphabet)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return sample, sorted(alphabet), total


def collect_reservoir_sample(
    paths: Iterable[str | Path],
    text_key: str,
    min_chars: int,
    sample_size: int,
    seed: int,
) -> tuple[list[str], int]:
    if sample_size <= 0:
        return [], 0
    rng = random.Random(seed)
    sample: list[str] = []
    total = 0
    for text in iter_texts(paths, text_key=text_key, min_chars=min_chars):
        total += 1
        if len(sample) < sample_size:
            sample.append(text)
            continue
        replacement_index = rng.randrange(total)
        if replacement_index < sample_size:
            sample[replacement_index] = text
    return sample, total


def build_bpe_model(byte_fallback: bool) -> models.BPE:
    try:
        return models.BPE(unk_token="<unk>", byte_fallback=byte_fallback)
    except TypeError:
        if byte_fallback:
            print(
                "warning: installed tokenizers does not support BPE byte_fallback; "
                "falling back to unk_token-only BPE",
                flush=True,
            )
    return models.BPE(unk_token="<unk>")


def build_pre_tokenizer(pre_tokenizer_name: str, split_digits: bool):
    parts = []
    if pre_tokenizer_name == "bytelevel":
        parts.append(pre_tokenizers.ByteLevel(add_prefix_space=False))
    else:
        parts.append(pre_tokenizers.Metaspace(replacement="\u2581"))
    if split_digits:
        parts.append(pre_tokenizers.Digits(individual_digits=True))
    if len(parts) == 1:
        return parts[0]
    return pre_tokenizers.Sequence(parts)


def build_decoder(pre_tokenizer_name: str, byte_fallback: bool):
    decoder_stack = []
    if byte_fallback and hasattr(decoders, "ByteFallback"):
        decoder_stack.append(decoders.ByteFallback())
    if pre_tokenizer_name == "bytelevel":
        decoder_stack.append(decoders.ByteLevel())
    else:
        decoder_stack.append(decoders.Metaspace(replacement="\u2581"))
    if len(decoder_stack) == 1:
        return decoder_stack[0]
    if hasattr(decoders, "Sequence"):
        return decoders.Sequence(decoder_stack)
    return decoder_stack[-1]


def validate_byte_fallback_config(pre_tokenizer_name: str, byte_fallback: bool) -> None:
    if byte_fallback and pre_tokenizer_name != "bytelevel":
        raise ValueError(
            "tokenizer.byte_fallback=true is only reliable with pre_tokenizer=bytelevel "
            "in this HuggingFace BPE builder. Use bytelevel, or set byte_fallback=false "
            "and accept that unseen Unicode may map to <unk>."
        )


def resolve_special_tokens(tokenizer_config: dict) -> list[str]:
    raw_symbols = tokenizer_config.get("user_defined_symbols", DEFAULT_USER_DEFINED_SYMBOLS)
    user_symbols = [str(symbol) for symbol in as_list(raw_symbols) if str(symbol)]
    special_tokens: list[str] = []
    for token in [*BASE_SPECIAL_TOKENS, *user_symbols]:
        if token not in special_tokens:
            special_tokens.append(token)
    return special_tokens


def validate_vocab_size_constraints(
    vocab_size: int,
    special_tokens: Iterable[str],
    initial_alphabet: Iterable[str],
) -> int:
    """Validate the hard lower bound imposed by specials and alphabet."""

    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if vocab_size > 2_147_483_647:
        raise ValueError("vocab_size exceeds the supported signed 32-bit token id range")
    required_tokens = set(str(token) for token in special_tokens)
    required_tokens.update(str(token) for token in initial_alphabet)
    required_size = len(required_tokens)
    if vocab_size < required_size:
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than the {required_size} tokens "
            "required by special_tokens + initial_alphabet. Increase vocab_size "
            "or reduce the forced alphabet."
        )
    return required_size


def ignored_tokenizer_config_keys(tokenizer_config: dict) -> list[str]:
    return sorted(key for key in tokenizer_config if key not in SUPPORTED_TOKENIZER_CONFIG_KEYS)


def tokenizer_diagnostics(tokenizer: Tokenizer, texts: list[str]) -> dict:
    unk_id = tokenizer.token_to_id("<unk>")
    total_chars = 0
    total_tokens = 0
    unk_tokens = 0
    empty_encodings = 0
    for text in texts:
        encoded = tokenizer.encode(text)
        total_chars += len(text)
        total_tokens += len(encoded.ids)
        if not encoded.ids:
            empty_encodings += 1
        if unk_id is not None:
            unk_tokens += sum(1 for token_id in encoded.ids if int(token_id) == int(unk_id))

    return {
        "documents": len(texts),
        "chars": total_chars,
        "tokens": total_tokens,
        "chars_per_token": round(total_chars / total_tokens, 6) if total_tokens else None,
        "unk_tokens": unk_tokens,
        "unk_token_rate": round(unk_tokens / total_tokens, 8) if total_tokens else None,
        "empty_encodings": empty_encodings,
    }


def unicode_fallback_diagnostics(tokenizer: Tokenizer) -> dict:
    unk_id = tokenizer.token_to_id("<unk>")
    probes: list[dict[str, object]] = []
    total_unk = 0
    for text in UNICODE_FALLBACK_PROBES:
        encoded = tokenizer.encode(text)
        unk_tokens = (
            sum(1 for token_id in encoded.ids if int(token_id) == int(unk_id))
            if unk_id is not None
            else 0
        )
        total_unk += unk_tokens
        probes.append(
            {
                "text": text,
                "tokens": encoded.tokens,
                "ids": [int(token_id) for token_id in encoded.ids],
                "unk_tokens": unk_tokens,
            }
        )
    return {"unicode_probe_unk_tokens": total_unk, "unicode_probes": probes}


def validate_unicode_fallback(tokenizer: Tokenizer, byte_fallback: bool) -> None:
    if not byte_fallback:
        return
    diagnostics = unicode_fallback_diagnostics(tokenizer)
    if int(diagnostics["unicode_probe_unk_tokens"]) > 0:
        raise ValueError(
            "Tokenizer still emits <unk> for Unicode fallback probes. "
            "Use pre_tokenizer=bytelevel and rebuild the tokenizer."
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

    model_type = str(tokenizer_config.get("model_type", "bpe")).lower()
    if model_type != "bpe":
        raise ValueError("build_hf_bpe_tokenizer.py only supports tokenizer.model_type=bpe")
    output = Path(args.output or tokenizer_config.get("path", "outputs/tokenizer/bpe_merged_24k.json"))
    vocab_size = int(args.vocab_size or tokenizer_config.get("vocab_size", 16000))
    sample_size = int(args.sample_size or tokenizer_config.get("sample_size", 10000))
    text_key = args.text_key or data_config.get("jsonl_text_key", "text")
    min_chars = int(args.min_chars or data_config.get("min_chars", 1))
    progress = int(args.progress or tokenizer_config.get("progress", 2000))
    pre_tokenizer_name = args.pre_tokenizer or tokenizer_config.get("pre_tokenizer", "metaspace")
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    min_frequency = int(
        args.min_frequency
        if args.min_frequency is not None
        else tokenizer_config.get("min_frequency", 2)
    )
    diagnostic_size = int(
        args.diagnostic_size
        if args.diagnostic_size is not None
        else tokenizer_config.get("diagnostic_size", min(sample_size, 50000))
    )
    split_digits = (
        bool(args.split_digits)
        if args.split_digits is not None
        else bool(tokenizer_config.get("split_digits", False))
    )
    byte_fallback = (
        bool(args.byte_fallback)
        if args.byte_fallback is not None
        else bool(tokenizer_config.get("byte_fallback", pre_tokenizer_name == "bytelevel"))
    )
    validate_byte_fallback_config(pre_tokenizer_name, byte_fallback)
    special_tokens = resolve_special_tokens(tokenizer_config)

    print(
        json.dumps(
            {
                "backend": "huggingface_bpe",
                "inputs": [str(path) for path in paths],
                "output": str(output),
                "vocab_size": vocab_size,
                "sample_size": sample_size,
                "min_frequency": min_frequency,
                "pre_tokenizer": pre_tokenizer_name,
                "split_digits": split_digits,
                "byte_fallback": byte_fallback,
                "diagnostic_size": diagnostic_size,
                "special_tokens": special_tokens,
                "ignored_config_keys": ignored_config_keys,
                "seed": seed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if ignored_config_keys:
        print(
            "warning: ignored tokenizer config keys for HF BPE: "
            + ", ".join(ignored_config_keys),
            flush=True,
        )

    sample_texts, alphabet, total_texts = collect_sample_and_alphabet(
        paths=paths,
        text_key=text_key,
        min_chars=min_chars,
        sample_size=sample_size,
        progress=progress,
        seed=seed,
    )
    if not sample_texts:
        raise ValueError("No texts were collected for tokenizer training.")

    tokenizer = Tokenizer(build_bpe_model(byte_fallback=byte_fallback))
    tokenizer.pre_tokenizer = build_pre_tokenizer(
        pre_tokenizer_name=pre_tokenizer_name,
        split_digits=split_digits,
    )
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet() if pre_tokenizer_name == "bytelevel" else alphabet
    minimum_vocab_size = validate_vocab_size_constraints(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
    )
    tokenizer.decoder = build_decoder(pre_tokenizer_name, byte_fallback=byte_fallback)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        show_progress=True,
    )

    print(
        f"training_tokenizer sample={len(sample_texts)} total_texts={total_texts} "
        f"alphabet={len(alphabet)}",
        flush=True,
    )
    tokenizer.train_from_iterator(sample_texts, trainer=trainer, length=len(sample_texts))
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size > vocab_size:
        raise RuntimeError(
            f"Tokenizer produced {actual_vocab_size} tokens, exceeding requested "
            f"vocab_size={vocab_size}."
        )
    if actual_vocab_size < minimum_vocab_size:
        raise RuntimeError(
            f"Tokenizer produced {actual_vocab_size} tokens, below required minimum "
            f"{minimum_vocab_size}."
        )
    validate_unicode_fallback(tokenizer, byte_fallback=byte_fallback)

    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    diagnostic_texts, diagnostic_total = collect_reservoir_sample(
        paths=paths,
        text_key=text_key,
        min_chars=min_chars,
        sample_size=diagnostic_size,
        seed=seed + 17,
    )
    diagnostics = tokenizer_diagnostics(tokenizer, diagnostic_texts or sample_texts)
    diagnostics.update(unicode_fallback_diagnostics(tokenizer))
    diagnostics.update(
        {
            "diagnostic_documents_seen": diagnostic_total,
            "diagnostic_sample_size": len(diagnostic_texts or sample_texts),
            "training_sample_size": len(sample_texts),
            "total_training_texts_seen": total_texts,
            "pre_tokenizer": pre_tokenizer_name,
            "split_digits": split_digits,
            "byte_fallback": byte_fallback,
            "min_frequency": min_frequency,
            "requested_vocab_size": vocab_size,
            "actual_vocab_size": actual_vocab_size,
            "minimum_vocab_size": minimum_vocab_size,
            "special_tokens": special_tokens,
            "ignored_config_keys": ignored_config_keys,
        }
    )
    diagnostics_path = output.with_suffix(output.suffix + ".diagnostics.json")
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Tokenizer saved to: {output}", flush=True)
    print(f"Vocab size: {actual_vocab_size}", flush=True)
    print(f"Tokenizer diagnostics saved to: {diagnostics_path}", flush=True)
    print(json.dumps(diagnostics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
