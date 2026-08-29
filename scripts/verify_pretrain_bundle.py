from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config
from muddywater.dataset import TokenBlockDataset
from muddywater.document_boundaries import resolve_document_boundary_settings
from muddywater.tokenizer import CharacterTokenizer
from muddywater.utils import as_list, atomic_write_text, file_sha256


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a prepared JSONL/tokenizer/token-cache training bundle without training."
    )
    parser.add_argument("--config", required=True, help="Training YAML that references the bundle.")
    parser.add_argument(
        "--expected-train-documents",
        type=int,
        default=None,
        help="Expected training rows. Defaults to manifest.json when omitted.",
    )
    parser.add_argument(
        "--expected-val-documents",
        type=int,
        default=None,
        help="Expected validation rows. Defaults to manifest.json when omitted.",
    )
    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument("--output", default=None, help="Optional JSON validation report path.")
    report_group.add_argument(
        "--no-write-report",
        action="store_true",
        help="Validate and print the report without writing validation_report.json.",
    )
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def single_config_path(config: dict[str, Any], key: str) -> Path:
    values = as_list(config.get(key))
    require(len(values) == 1, f"Expected exactly one {key}, got {values!r}")
    return project_path(str(values[0]))


def resolve_token_paths(
    cache_dir: Path,
    configured: Any,
    *,
    fallback: str,
    config_key: str,
) -> list[Path]:
    """Resolve token-cache filenames exactly as training does, but fail on misses."""

    patterns = as_list(configured) or [fallback]
    resolved: dict[Path, Path] = {}
    for raw_pattern in patterns:
        require(str(raw_pattern).strip() != "", f"{config_key} contains an empty path")
        pattern = Path(str(raw_pattern))
        if not pattern.is_absolute():
            pattern = cache_dir / pattern
        pattern_text = str(pattern)
        if any(character in pattern_text for character in "*?["):
            matches = sorted(Path(match) for match in glob.glob(pattern_text))
            file_matches = [match for match in matches if match.is_file()]
            require(file_matches, f"{config_key} pattern matched no files: {raw_pattern}")
        else:
            require(pattern.is_file(), f"Token cache not found: {pattern}")
            file_matches = [pattern]
        for match in file_matches:
            resolved[match.resolve()] = match

    paths = sorted(resolved.values(), key=lambda item: str(item.resolve()))
    require(paths, f"{config_key} resolved to no token shards")
    return paths


def resolve_expected_documents(
    explicit: int | None,
    manifest: dict[str, Any],
    manifest_key: str,
    cli_flag: str,
) -> int:
    value = explicit if explicit is not None else manifest.get(manifest_key)
    require(
        value is not None,
        f"Provide {cli_flag} or add {manifest_key!r} to the cache manifest",
    )
    parsed = int(value)
    require(parsed > 0, f"{cli_flag}/{manifest_key} must be positive")
    return parsed


def optional_positive_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    parsed = int(value)
    require(parsed > 0, f"training.{key} must be positive")
    return parsed


def validate_training_plan(training_config: dict[str, Any]) -> dict[str, Any]:
    max_steps = optional_positive_int(training_config, "max_steps")
    max_train_tokens = optional_positive_int(training_config, "max_train_tokens")
    max_epochs = optional_positive_int(training_config, "max_epochs")
    require(max_epochs is not None, "Bundle config needs training.max_epochs > 0")
    require(
        not (max_steps is not None and max_train_tokens is not None),
        "Use only one bounded training limit: max_steps or max_train_tokens",
    )
    eval_interval = optional_positive_int(training_config, "eval_interval")
    save_interval = optional_positive_int(training_config, "save_interval")
    require(eval_interval is not None, "Bundle config needs training.eval_interval > 0")
    require(save_interval is not None, "Bundle config needs training.save_interval > 0")

    if max_train_tokens is not None:
        limit_kind = "tokens"
    elif max_steps is not None:
        limit_kind = "steps"
    else:
        limit_kind = "epochs"
    return {
        "limit_kind": limit_kind,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "max_train_tokens": max_train_tokens,
        "eval_interval": eval_interval,
        "save_interval": save_interval,
    }


def validate_jsonl(
    path: Path,
    expected_documents: int,
    text_key: str,
    min_chars: int,
) -> tuple[dict[str, Any], set[bytes]]:
    require(path.exists(), f"JSONL file not found: {path}")
    fingerprints: set[bytes] = set()
    duplicate_texts = 0
    count = 0
    total_chars = 0
    min_observed: int | None = None
    max_observed = 0

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            require(bool(line.strip()), f"Blank JSONL row at {path}:{line_no}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            require(isinstance(record, dict), f"Expected object at {path}:{line_no}")
            text = record.get(text_key)
            require(isinstance(text, str), f"Missing string {text_key!r} at {path}:{line_no}")
            require(len(text) >= min_chars, f"Text shorter than {min_chars} at {path}:{line_no}")
            require(bool(text.strip()), f"Empty text at {path}:{line_no}")

            count += 1
            text_length = len(text)
            total_chars += text_length
            min_observed = text_length if min_observed is None else min(min_observed, text_length)
            max_observed = max(max_observed, text_length)
            fingerprint = hashlib.sha256(text.encode("utf-8")).digest()
            if fingerprint in fingerprints:
                duplicate_texts += 1
            fingerprints.add(fingerprint)

    require(count == expected_documents, f"{path} has {count} rows, expected {expected_documents}")
    require(duplicate_texts == 0, f"{path} contains {duplicate_texts} duplicate texts")
    return (
        {
            "path": display_path(path),
            "documents": count,
            "sha256": file_sha256(path),
            "min_chars": min_observed,
            "max_chars": max_observed,
            "mean_chars": round(total_chars / count, 4) if count else None,
            "duplicate_texts": duplicate_texts,
        },
        fingerprints,
    )


def validate_configured_jsonl(
    data_config: dict[str, Any],
    expected_train_documents: int,
    expected_val_documents: int,
    cache_split_mode: str = "hash",
) -> dict[str, Any]:
    """Validate explicit JSONL splits or one source split during cache creation."""

    text_key = str(data_config.get("jsonl_text_key", "text"))
    min_chars = int(data_config.get("min_chars", 1))
    val_paths = as_list(data_config.get("val_paths"))
    if val_paths:
        train_jsonl = single_config_path(data_config, "train_paths")
        val_jsonl = single_config_path(data_config, "val_paths")
        train_report, train_fingerprints = validate_jsonl(
            train_jsonl,
            expected_train_documents,
            text_key,
            min_chars,
        )
        val_report, val_fingerprints = validate_jsonl(
            val_jsonl,
            expected_val_documents,
            text_key,
            min_chars,
        )
        overlap = len(train_fingerprints.intersection(val_fingerprints))
        require(overlap == 0, f"Train/validation text overlap: {overlap}")
        return {
            "mode": "explicit_files",
            "train": train_report,
            "validation": val_report,
            "cross_split_text_overlap": overlap,
        }

    require(
        str(cache_split_mode).lower() == "hash",
        "A single source JSONL can only prove split isolation for deterministic hash splitting",
    )
    source_jsonl = single_config_path(data_config, "train_paths")
    source_report, _ = validate_jsonl(
        source_jsonl,
        expected_train_documents + expected_val_documents,
        text_key,
        min_chars,
    )
    return {
        "mode": "cache_hash_split_from_single_source",
        "source": source_report,
        "cross_split_text_overlap": 0,
        "cross_split_text_overlap_basis": (
            "exact duplicate texts route together under deterministic cache splitting"
        ),
    }


def validate_tokenizer(
    tokenizer_path: Path,
    config: dict[str, Any],
    expected_documents: int,
) -> tuple[CharacterTokenizer, dict[str, Any]]:
    require(tokenizer_path.exists(), f"Tokenizer not found: {tokenizer_path}")
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    expected_vocab_size = int(config.get("vocab_size", tokenizer.vocab_size))
    require(
        tokenizer.vocab_size == expected_vocab_size,
        f"Tokenizer vocab is {tokenizer.vocab_size}, expected {expected_vocab_size}",
    )
    require(
        [tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id] == [0, 1, 2, 3],
        "Tokenizer special token IDs must be pad/unk/bos/eos = 0/1/2/3",
    )

    diagnostics_path = tokenizer_path.with_suffix(tokenizer_path.suffix + ".diagnostics.json")
    diagnostics = read_json(diagnostics_path)
    configured_sample_size = int(config.get("sample_size", expected_documents))
    configured_diagnostic_size = int(config.get("diagnostic_size", configured_sample_size))
    require(configured_sample_size > 0, "tokenizer.sample_size must be positive")
    require(configured_diagnostic_size > 0, "tokenizer.diagnostic_size must be positive")
    expected_training_sample = min(configured_sample_size, expected_documents)
    expected_diagnostic_sample = min(configured_diagnostic_size, expected_documents)
    require(
        int(diagnostics.get("training_sample_size", -1)) == expected_training_sample,
        "Tokenizer training sample count mismatch",
    )
    require(
        int(diagnostics.get("diagnostic_sample_size", -1)) == expected_diagnostic_sample,
        "Tokenizer diagnostic sample count mismatch",
    )
    require(int(diagnostics.get("unk_tokens", -1)) == 0, "Tokenizer diagnostics contain unknown tokens")
    require(int(diagnostics.get("unicode_probe_unk_tokens", -1)) == 0, "Unicode probes contain unknown tokens")

    return tokenizer, {
        "path": display_path(tokenizer_path),
        "sha256": file_sha256(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": {
            "pad": tokenizer.pad_id,
            "unk": tokenizer.unk_id,
            "bos": tokenizer.bos_id,
            "eos": tokenizer.eos_id,
        },
        "diagnostic_documents": int(diagnostics["diagnostic_sample_size"]),
        "training_documents": int(diagnostics["training_sample_size"]),
        "diagnostic_tokens": int(diagnostics["tokens"]),
        "chars_per_token": diagnostics.get("chars_per_token"),
        "unk_tokens": int(diagnostics["unk_tokens"]),
        "unicode_probe_unk_tokens": int(diagnostics["unicode_probe_unk_tokens"]),
    }


def validate_cache_shard(
    token_path: Path,
    tokenizer_sha256: str,
    vocab_size: int,
    add_bos: bool,
    max_seq_len: int,
    document_attention: bool,
    ignore_cross_document_targets: bool,
) -> dict[str, Any]:
    meta_path = token_path.with_suffix(token_path.suffix + ".meta.json")
    require(token_path.exists(), f"Token cache not found: {token_path}")
    require(meta_path.exists(), f"Token cache metadata not found: {meta_path}")
    meta = read_json(meta_path)
    num_documents = int(meta.get("num_documents", -1))
    require(num_documents > 0, f"Invalid document count in metadata: {token_path}")
    require(int(meta.get("vocab_size", -1)) == vocab_size, f"Vocab mismatch: {token_path}")
    require(str(meta.get("tokenizer_sha256", "")).lower() == tokenizer_sha256.lower(), f"Tokenizer SHA mismatch: {token_path}")
    require(bool(meta.get("add_bos")) == add_bos, f"BOS setting mismatch: {token_path}")

    require("dtype" in meta, f"Token cache metadata missing dtype: {meta_path}")
    dtype = np.dtype(str(meta["dtype"]))
    require(np.issubdtype(dtype, np.integer), f"Token cache dtype must be integer: {token_path}")
    num_tokens = int(meta.get("num_tokens", -1))
    require(num_tokens > 0, f"Invalid token count in metadata: {token_path}")
    require(token_path.stat().st_size == num_tokens * dtype.itemsize, f"Binary size mismatch: {token_path}")
    tokens = np.memmap(token_path, dtype=dtype, mode="r")
    require(tokens.size == num_tokens, f"Token count mismatch: {token_path}")
    require(int(tokens.max()) < vocab_size, f"Out-of-vocabulary ID in {token_path}")
    require(int(tokens.min()) >= 0, f"Negative token ID in {token_path}")

    doc_starts_file = str(meta.get("doc_starts_file", "")).strip()
    require(doc_starts_file != "", f"Metadata missing doc_starts_file: {meta_path}")
    configured_doc_starts_path = Path(doc_starts_file)
    doc_starts_path = (
        configured_doc_starts_path
        if configured_doc_starts_path.is_absolute()
        else token_path.parent / configured_doc_starts_path
    )
    require(doc_starts_path.exists(), f"Document starts not found: {doc_starts_path}")
    doc_starts = np.load(doc_starts_path, mmap_mode="r")
    require(doc_starts.ndim == 1, f"Document starts must be 1D: {doc_starts_path}")
    require(
        np.issubdtype(doc_starts.dtype, np.integer),
        f"Document starts must use an integer dtype: {doc_starts_path}",
    )
    require(doc_starts.size == num_documents, f"Document starts count mismatch: {doc_starts_path}")
    require(int(doc_starts[0]) == 0, f"First document must start at zero: {doc_starts_path}")
    require(bool(np.all(doc_starts[1:] > doc_starts[:-1])), f"Document starts are not increasing: {doc_starts_path}")
    require(int(doc_starts[-1]) < num_tokens, f"Last document starts outside cache: {doc_starts_path}")

    dataset = TokenBlockDataset(
        token_path=token_path,
        max_seq_len=max_seq_len,
        expected_vocab_size=vocab_size,
        expected_tokenizer_sha256=tokenizer_sha256,
        expected_add_bos=add_bos,
        strict_meta=True,
        document_attention=document_attention,
        ignore_cross_document_targets=ignore_cross_document_targets,
    )
    require(len(dataset) > 0, f"No {max_seq_len}-token training windows in {token_path}")
    first = dataset[0]
    last = dataset[len(dataset) - 1]
    require(first["input_ids"].numel() == max_seq_len, f"Invalid first window: {token_path}")
    require(last["input_ids"].numel() == max_seq_len, f"Invalid last window: {token_path}")

    return {
        "path": display_path(token_path),
        "sha256": file_sha256(token_path),
        "dtype": str(dtype),
        "bytes": token_path.stat().st_size,
        "tokens": num_tokens,
        "documents": num_documents,
        "training_windows": len(dataset),
        "doc_starts_file": doc_starts_file,
        "doc_starts_path": display_path(doc_starts_path),
        "doc_starts_sha256": file_sha256(doc_starts_path),
        "meta_path": display_path(meta_path),
        "meta_sha256": file_sha256(meta_path),
        "shard_index": int(meta["shard_index"]) if meta.get("shard_index") is not None else None,
        "min_token_id": int(tokens.min()),
        "max_token_id": int(tokens.max()),
    }


def validate_cache_split(
    token_paths: Path | str | Sequence[Path | str],
    expected_documents: int,
    tokenizer_sha256: str,
    vocab_size: int,
    add_bos: bool,
    max_seq_len: int,
    document_attention: bool,
    ignore_cross_document_targets: bool,
) -> dict[str, Any]:
    """Validate every shard and return both aggregate and per-shard facts."""

    if isinstance(token_paths, (str, Path)):
        paths = [Path(token_paths)]
    else:
        paths = [Path(path) for path in token_paths]
    require(paths, "A cache split must contain at least one token shard")
    require(
        len({path.resolve() for path in paths}) == len(paths),
        "A cache split contains duplicate token shards",
    )
    shard_reports = [
        validate_cache_shard(
            token_path=path,
            tokenizer_sha256=tokenizer_sha256,
            vocab_size=vocab_size,
            add_bos=add_bos,
            max_seq_len=max_seq_len,
            document_attention=document_attention,
            ignore_cross_document_targets=ignore_cross_document_targets,
        )
        for path in paths
    ]

    documents = sum(int(shard["documents"]) for shard in shard_reports)
    tokens = sum(int(shard["tokens"]) for shard in shard_reports)
    require(
        documents == expected_documents,
        f"Cache split has {documents} documents across {len(paths)} shards, "
        f"expected {expected_documents}",
    )
    dtypes = {str(shard["dtype"]) for shard in shard_reports}
    require(len(dtypes) == 1, f"Cache split uses inconsistent dtypes: {sorted(dtypes)}")
    aggregate = {
        "shard_count": len(shard_reports),
        "paths": [str(shard["path"]) for shard in shard_reports],
        "dtype": next(iter(dtypes)),
        "bytes": sum(int(shard["bytes"]) for shard in shard_reports),
        "tokens": tokens,
        "documents": documents,
        "training_windows": sum(int(shard["training_windows"]) for shard in shard_reports),
        "min_token_id": min(int(shard["min_token_id"]) for shard in shard_reports),
        "max_token_id": max(int(shard["max_token_id"]) for shard in shard_reports),
        "shards": shard_reports,
    }
    if len(shard_reports) == 1:
        # Preserve the legacy single-file fields while also exposing the new list.
        aggregate = {**shard_reports[0], **aggregate}
    return aggregate


def validate_cache_manifest(
    cache_dir: Path,
    train_documents: int,
    val_documents: int,
    tokenizer_sha256: str,
    *,
    train_paths: Sequence[Path] | None = None,
    val_paths: Sequence[Path] | None = None,
    train_report: dict[str, Any] | None = None,
    val_report: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    stats_path = cache_dir / "prepare_stats.json"
    manifest = manifest if manifest is not None else read_json(manifest_path)
    stats = read_json(stats_path)
    require(int(manifest.get("train_documents", -1)) == train_documents, "Cache manifest train count mismatch")
    require(int(manifest.get("val_documents", -1)) == val_documents, "Cache manifest val count mismatch")
    require(str(manifest.get("tokenizer_sha256", "")).lower() == tokenizer_sha256.lower(), "Cache manifest tokenizer SHA mismatch")
    require(int(stats.get("unk_tokens", -1)) == 0, "Cache contains unknown tokens")
    require(int(stats.get("unk_documents", -1)) == 0, "Cache contains documents with unknown tokens")

    manifest_train_tokens = int(manifest.get("train_tokens", -1))
    manifest_val_tokens = int(manifest.get("val_tokens", -1))
    require(manifest_train_tokens > 0, "Cache manifest train token count must be positive")
    require(manifest_val_tokens > 0, "Cache manifest validation token count must be positive")

    split_inputs = (
        ("train", train_paths, train_report, train_documents, manifest_train_tokens),
        ("val", val_paths, val_report, val_documents, manifest_val_tokens),
    )
    manifest_shards_validated: dict[str, bool] = {}
    shard_counts: dict[str, int] = {}
    for split, paths, split_report, expected_docs, expected_tokens in split_inputs:
        if paths is None or split_report is None:
            manifest_shards_validated[split] = False
            shard_counts[split] = 0
            continue
        require(
            int(split_report.get("documents", -1)) == expected_docs,
            f"{split} cache document total does not match manifest",
        )
        require(
            int(split_report.get("tokens", -1)) == expected_tokens,
            f"{split} cache token total does not match manifest",
        )
        shards = split_report.get("shards")
        require(isinstance(shards, list), f"Missing per-shard report for {split}")
        require(len(shards) == len(paths), f"Per-shard report count mismatch for {split}")
        shard_counts[split] = len(paths)

        manifest_entries = manifest.get(f"{split}_shards")
        if manifest_entries is None:
            # Older single-file manifests did not enumerate shards. Aggregate
            # document/token checks above still provide backwards compatibility.
            require(len(paths) == 1, f"Sharded {split} cache requires manifest {split}_shards")
            manifest_shards_validated[split] = False
            continue
        require(isinstance(manifest_entries, list), f"manifest.{split}_shards must be a list")
        require(
            len(manifest_entries) == len(paths),
            f"manifest.{split}_shards has {len(manifest_entries)} entries, "
            f"but config resolves {len(paths)} files",
        )

        actual_by_path = {
            path.resolve(): shard_report
            for path, shard_report in zip(paths, shards, strict=True)
        }
        manifest_by_path: dict[Path, tuple[int, dict[str, Any]]] = {}
        for position, entry in enumerate(manifest_entries):
            require(isinstance(entry, dict), f"manifest.{split}_shards[{position}] must be an object")
            filename = str(entry.get("file", "")).strip()
            require(filename != "", f"manifest.{split}_shards[{position}] is missing file")
            listed_path = Path(filename)
            if not listed_path.is_absolute():
                listed_path = cache_dir / listed_path
            listed_path = listed_path.resolve()
            require(
                listed_path not in manifest_by_path,
                f"Duplicate shard in manifest.{split}_shards: {filename}",
            )
            manifest_by_path[listed_path] = (position, entry)

        require(
            set(manifest_by_path) == set(actual_by_path),
            f"Configured {split} token shards do not match manifest.{split}_shards",
        )
        entry_documents = 0
        entry_tokens = 0
        for path, shard_report in actual_by_path.items():
            position, entry = manifest_by_path[path]
            entry_num_documents = int(entry.get("num_documents", -1))
            entry_num_tokens = int(entry.get("num_tokens", -1))
            require(
                entry_num_documents == int(shard_report["documents"]),
                f"Manifest document count mismatch for shard: {path}",
            )
            require(
                entry_num_tokens == int(shard_report["tokens"]),
                f"Manifest token count mismatch for shard: {path}",
            )
            entry_documents += entry_num_documents
            entry_tokens += entry_num_tokens

            listed_doc_starts = str(entry.get("doc_starts_file", "")).strip()
            if listed_doc_starts:
                listed_doc_starts_path = Path(listed_doc_starts)
                if not listed_doc_starts_path.is_absolute():
                    listed_doc_starts_path = cache_dir / listed_doc_starts_path
                meta_doc_starts_path = Path(str(shard_report["doc_starts_file"]))
                if not meta_doc_starts_path.is_absolute():
                    meta_doc_starts_path = path.parent / meta_doc_starts_path
                require(
                    listed_doc_starts_path.resolve() == meta_doc_starts_path.resolve(),
                    f"Manifest doc_starts_file mismatch for shard: {path}",
                )

            meta_shard_index = shard_report.get("shard_index")
            if meta_shard_index is not None:
                require(
                    int(meta_shard_index) == position,
                    f"Metadata shard_index mismatch for shard: {path}",
                )

        require(entry_documents == expected_docs, f"manifest.{split}_shards document sum mismatch")
        require(entry_tokens == expected_tokens, f"manifest.{split}_shards token sum mismatch")
        manifest_shards_validated[split] = True

    for key, expected in (
        ("train_documents", train_documents),
        ("val_documents", val_documents),
        ("train_tokens", manifest_train_tokens),
        ("val_tokens", manifest_val_tokens),
    ):
        if key in stats:
            require(int(stats[key]) == expected, f"prepare_stats {key} mismatch")

    return {
        "manifest_path": display_path(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "prepare_stats_path": display_path(stats_path),
        "prepare_stats_sha256": file_sha256(stats_path),
        "train_tokens": manifest_train_tokens,
        "val_tokens": manifest_val_tokens,
        "train_documents": train_documents,
        "val_documents": val_documents,
        "train_shard_count": shard_counts["train"],
        "val_shard_count": shard_counts["val"],
        "manifest_train_shards_validated": manifest_shards_validated["train"],
        "manifest_val_shards_validated": manifest_shards_validated["val"],
        "unk_tokens": int(stats["unk_tokens"]),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config_path = project_path(args.config)
    config = load_config(config_path)
    data_config = config.get("data", {})
    tokenizer_config = config.get("tokenizer", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    training_plan = validate_training_plan(training_config)

    cache_dir = project_path(str(data_config["token_cache_dir"]))
    require(cache_dir.is_dir(), f"Cache directory not found: {cache_dir}")
    manifest = read_json(cache_dir / "manifest.json")
    expected_train_documents = resolve_expected_documents(
        args.expected_train_documents,
        manifest,
        "train_documents",
        "--expected-train-documents",
    )
    expected_val_documents = resolve_expected_documents(
        args.expected_val_documents,
        manifest,
        "val_documents",
        "--expected-val-documents",
    )

    jsonl_report = validate_configured_jsonl(
        data_config,
        expected_train_documents=expected_train_documents,
        expected_val_documents=expected_val_documents,
        cache_split_mode=str(manifest.get("split_mode", "")),
    )

    tokenizer_path = project_path(str(tokenizer_config["path"]))
    tokenizer, tokenizer_report = validate_tokenizer(
        tokenizer_path, tokenizer_config, expected_train_documents
    )
    tokenizer_sha256 = str(tokenizer_report["sha256"])

    train_token_paths = resolve_token_paths(
        cache_dir,
        data_config.get("train_token_files"),
        fallback="train.bin",
        config_key="data.train_token_files",
    )
    val_token_paths = resolve_token_paths(
        cache_dir,
        data_config.get("val_token_files"),
        fallback="val.bin",
        config_key="data.val_token_files",
    )
    max_seq_len = int(model_config["max_seq_len"])
    add_bos = bool(data_config.get("add_bos", True))
    boundary_settings = resolve_document_boundary_settings(data_config)

    train_cache_report = validate_cache_split(
        train_token_paths,
        expected_train_documents,
        tokenizer_sha256,
        tokenizer.vocab_size,
        add_bos,
        max_seq_len,
        boundary_settings.document_attention,
        boundary_settings.ignore_cross_document_targets,
    )
    val_cache_report = validate_cache_split(
        val_token_paths,
        expected_val_documents,
        tokenizer_sha256,
        tokenizer.vocab_size,
        add_bos,
        max_seq_len,
        boundary_settings.document_attention,
        boundary_settings.ignore_cross_document_targets,
    )
    cache_manifest_report = validate_cache_manifest(
        cache_dir,
        expected_train_documents,
        expected_val_documents,
        tokenizer_sha256,
        train_paths=train_token_paths,
        val_paths=val_token_paths,
        train_report=train_cache_report,
        val_report=val_cache_report,
        manifest=manifest,
    )

    report = {
        "status": "passed",
        "config": display_path(config_path),
        "config_sha256": file_sha256(config_path),
        "training_plan": {
            **training_plan,
            "max_seq_len": max_seq_len,
            "document_boundary_policy": boundary_settings.policy,
            "document_attention": boundary_settings.document_attention,
            "ignore_cross_document_targets": boundary_settings.ignore_cross_document_targets,
        },
        "jsonl": jsonl_report,
        "tokenizer": tokenizer_report,
        "cache": {
            "directory": display_path(cache_dir),
            "train": train_cache_report,
            "validation": val_cache_report,
            **cache_manifest_report,
        },
    }
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if args.no_write_report:
        print("Validation report not written (--no-write-report).", flush=True)
    else:
        output_path = project_path(args.output) if args.output else cache_dir / "validation_report.json"
        atomic_write_text(
            output_path,
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Validation report saved to: {display_path(output_path)}", flush=True)
    return report


if __name__ == "__main__":
    main()
