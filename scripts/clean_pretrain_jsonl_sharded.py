from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.dataset import discover_data_files
from scripts.clean_pretrain_jsonl import (
    RunningStats,
    cjk_ratio,
    clean_text,
    dedupe_digest,
    quality_reject_reason,
    split_long_text,
)


STATE_FILE = "cleaning_state.json"
REPORT_FILE = "cleaning_report.json"


def reset_stage_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


class JsonlShardWriter:
    def __init__(self, output_dir: Path, prefix: str, max_records_per_shard: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_records_per_shard = max(1, int(max_records_per_shard))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shards: list[dict[str, Any]] = []
        self._index = 0
        self._records_in_current = 0
        self._total_records = 0
        self._handle = None
        self._path: Path | None = None
        self._open_next()

    @property
    def total_records(self) -> int:
        return self._total_records

    def _next_path(self) -> Path:
        return self.output_dir / f"{self.prefix}_{self._index:05d}.jsonl"

    def _open_next(self) -> None:
        self._path = self._next_path()
        self._handle = self._path.open("w", encoding="utf-8", newline="\n")
        self._records_in_current = 0

    def _close_current(self) -> None:
        if self._handle is None or self._path is None:
            return
        self._handle.close()
        self.shards.append(
            {
                "file": self._path.name,
                "records": self._records_in_current,
            }
        )
        self._handle = None

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            self._open_next()
        if self._records_in_current >= self.max_records_per_shard:
            self._close_current()
            self._index += 1
            self._open_next()
        assert self._handle is not None
        self._handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._records_in_current += 1
        self._total_records += 1

    def close(self) -> None:
        self._close_current()


def iter_jsonl_objects(paths: Iterable[str | Path]) -> Iterable[tuple[Path, int, dict[str, Any] | None]]:
    for path in discover_data_files(paths):
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Sharded cleaner only supports JSONL input, got: {path}")
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    yield path, line_no, None
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    yield path, line_no, None
                    continue
                if not isinstance(payload, dict):
                    yield path, line_no, None
                    continue
                yield path, line_no, payload


def clean_record_texts(obj: dict[str, Any], args: argparse.Namespace) -> list[str] | str:
    value = obj.get(args.text_key)
    if not isinstance(value, str):
        return "missing_or_non_string_text"
    text = clean_text(value, normalize_unicode=bool(args.unicode_normalize))
    if not text:
        return "empty_text_after_clean"
    if "\ufffd" in text:
        return "replacement_character"
    if len(text) > args.max_len:
        if args.long_text_mode == "drop":
            return "too_long"
        chunks = split_long_text(text, max_len=args.max_len, min_len=args.min_len)
        return chunks if chunks else "too_long_no_valid_chunk"
    return [text]


def quality_filter_text(text: str, args: argparse.Namespace) -> str | None:
    if len(text) < args.min_len:
        return "too_short"
    if len(text) > args.max_len:
        return "too_long"
    if cjk_ratio(text) < args.min_cjk_ratio:
        return "low_cjk_ratio"
    return quality_reject_reason(
        text,
        max_repeated_line_ratio=float(args.max_repeated_line_ratio),
        max_url_ratio=float(args.max_url_ratio),
        max_html_ratio=float(args.max_html_ratio),
        max_punctuation_ratio=float(args.max_punctuation_ratio),
        max_digit_ratio=float(args.max_digit_ratio),
    )


def clean_to_candidate_shards(args: argparse.Namespace) -> dict[str, Any]:
    candidate_dir = Path(args.output_dir) / "candidates"
    reset_stage_dir(candidate_dir)
    writer = JsonlShardWriter(candidate_dir, prefix="candidate", max_records_per_shard=args.records_per_shard)
    drop_reasons: Counter[str] = Counter()
    raw_length_stats = RunningStats()
    clean_length_stats = RunningStats()
    total_lines = 0
    try:
        for path, line_no, obj in iter_jsonl_objects(args.input):
            total_lines += 1
            if obj is None:
                drop_reasons["invalid_or_empty_jsonl"] += 1
                continue
            value = obj.get(args.text_key)
            if isinstance(value, str):
                raw_length_stats.add(len(value))
            cleaned = clean_record_texts(obj, args)
            if isinstance(cleaned, str):
                drop_reasons[cleaned] += 1
                continue
            for text in cleaned:
                clean_length_stats.add(len(text))
                reason = quality_filter_text(text, args)
                if reason:
                    drop_reasons[reason] += 1
                    continue
                record = dict(obj) if args.keep_extra_fields else {}
                record[args.text_key] = text
                record["_source_file"] = str(path)
                record["_source_line"] = line_no
                writer.write(record)
            if args.progress_interval > 0 and total_lines % args.progress_interval == 0:
                print(
                    f"stage=clean lines={total_lines} candidates={writer.total_records} "
                    f"dropped={sum(drop_reasons.values())}",
                    flush=True,
                )
    finally:
        writer.close()
    return {
        "candidate_dir": str(candidate_dir),
        "candidate_shards": writer.shards,
        "total_lines": total_lines,
        "candidate_records": writer.total_records,
        "drop_reasons": dict(drop_reasons),
        "raw_length": raw_length_stats.as_dict(),
        "clean_length": clean_length_stats.as_dict(),
    }


def bucket_for_digest(digest: bytes | None, bucket_count: int) -> int:
    if digest is None:
        return 0
    return int.from_bytes(digest[:4], byteorder="big", signed=False) % max(1, int(bucket_count))


def iter_candidate_records(candidate_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(candidate_dir.glob("candidate_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def dedupe_candidate_shards(args: argparse.Namespace, clean_report: dict[str, Any]) -> dict[str, Any]:
    candidate_dir = Path(clean_report["candidate_dir"])
    final_dir = Path(args.output_dir) / "cleaned"
    bucket_dir = Path(args.output_dir) / "dedupe_buckets"
    reset_stage_dir(final_dir)
    reset_stage_dir(bucket_dir)

    if args.dedupe_mode == "none":
        writer = JsonlShardWriter(final_dir, prefix="clean", max_records_per_shard=args.records_per_shard)
        try:
            for record in iter_candidate_records(candidate_dir):
                record.pop("_source_file", None)
                record.pop("_source_line", None)
                writer.write(record)
        finally:
            writer.close()
        return {
            "dedupe_mode": "none",
            "duplicates": 0,
            "final_records": writer.total_records,
            "final_shards": writer.shards,
        }

    bucket_counts = [0 for _ in range(int(args.buckets))]
    bucket_handles = [
        (bucket_dir / f"bucket_{idx:05d}.jsonl").open("w", encoding="utf-8", newline="\n")
        for idx in range(int(args.buckets))
    ]
    try:
        for record in iter_candidate_records(candidate_dir):
            text = str(record.get(args.text_key, ""))
            digest = dedupe_digest(text, args.dedupe_mode)
            bucket_idx = bucket_for_digest(digest, args.buckets)
            bucket_counts[bucket_idx] += 1
            bucket_handles[bucket_idx].write(
                json.dumps(
                    {
                        "digest": digest.hex() if digest is not None else None,
                        "record": record,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    finally:
        for handle in bucket_handles:
            handle.close()

    writer = JsonlShardWriter(final_dir, prefix="clean", max_records_per_shard=args.records_per_shard)
    duplicates = 0
    try:
        for bucket_path in sorted(bucket_dir.glob("bucket_*.jsonl")):
            seen: set[str] = set()
            with bucket_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    digest = payload.get("digest")
                    if digest in seen:
                        duplicates += 1
                        continue
                    seen.add(digest)
                    record = payload["record"]
                    record.pop("_source_file", None)
                    record.pop("_source_line", None)
                    writer.write(record)
    finally:
        writer.close()

    bucket_stats = summarize_bucket_counts(bucket_counts)
    warnings = []
    largest_ratio = bucket_stats.get("largest_bucket_ratio")
    if isinstance(largest_ratio, (int, float)) and largest_ratio > float(args.bucket_skew_warning_ratio):
        warnings.append(
            "largest dedupe bucket exceeds skew warning ratio; increase --buckets or inspect hash distribution"
        )
    return {
        "dedupe_mode": args.dedupe_mode,
        "bucket_count": int(args.buckets),
        "bucket_distribution": bucket_stats,
        "warnings": warnings,
        "duplicates": duplicates,
        "final_records": writer.total_records,
        "final_shards": writer.shards,
        "bucket_dir": str(bucket_dir),
    }


def summarize_bucket_counts(counts: list[int]) -> dict[str, Any]:
    if not counts:
        return {
            "bucket_records_min": 0,
            "bucket_records_max": 0,
            "bucket_records_mean": 0.0,
            "largest_bucket_ratio": 0.0,
        }
    total = sum(counts)
    return {
        "bucket_records_min": min(counts),
        "bucket_records_max": max(counts),
        "bucket_records_mean": round(total / len(counts), 4),
        "largest_bucket_ratio": round(max(counts) / total, 8) if total else 0.0,
    }


def args_fingerprint(args: argparse.Namespace) -> str:
    ignored = {"resume", "overwrite"}
    payload = {
        key: value
        for key, value in vars(args).items()
        if key not in ignored
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def state_path(output_dir: Path) -> Path:
    return output_dir / STATE_FILE


def report_path(output_dir: Path) -> Path:
    return output_dir / REPORT_FILE


def read_state(output_dir: Path) -> dict[str, Any] | None:
    path = state_path(output_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(output_dir: Path, state: dict[str, Any]) -> None:
    state = dict(state)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state_path(output_dir).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output_dir(args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    output_dir = Path(args.output_dir)
    existing_state = read_state(output_dir) if output_dir.exists() else None
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
        existing_state = None
    non_empty_output = output_dir.exists() and any(output_dir.iterdir())
    if non_empty_output and args.resume and existing_state is None:
        raise ValueError(
            f"Cannot resume without {STATE_FILE}: {output_dir}. Use --overwrite to rebuild."
        )
    if non_empty_output and not args.resume:
        raise ValueError(f"Output directory is not empty: {output_dir}. Use --resume or --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = args_fingerprint(args)
    if existing_state is not None:
        if existing_state.get("args_fingerprint") != fingerprint:
            raise ValueError(
                "Existing cleaning_state.json was created with different arguments. "
                "Use --overwrite to rebuild."
            )
        if existing_state.get("stage") == "completed" and report_path(output_dir).exists():
            return output_dir, existing_state
        return output_dir, existing_state
    write_state(
        output_dir,
        {
            "stage": "initialized",
            "args_fingerprint": fingerprint,
            "input": [str(path) for path in args.input],
        },
    )
    return output_dir, existing_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean large JSONL corpora into deduped JSONL shards.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--min-len", type=int, default=300)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--long-text-mode", choices=["chunk", "drop"], default="chunk")
    parser.add_argument("--min-cjk-ratio", type=float, default=0.2)
    parser.add_argument("--dedupe-mode", choices=["normalized", "exact", "none"], default="normalized")
    parser.add_argument("--buckets", type=int, default=256)
    parser.add_argument("--records-per-shard", type=int, default=100000)
    parser.add_argument("--bucket-skew-warning-ratio", type=float, default=0.2)
    parser.add_argument("--max-repeated-line-ratio", type=float, default=0.8)
    parser.add_argument("--max-url-ratio", type=float, default=0.4)
    parser.add_argument("--max-html-ratio", type=float, default=0.2)
    parser.add_argument("--max-punctuation-ratio", type=float, default=0.8)
    parser.add_argument("--max-digit-ratio", type=float, default=0.8)
    parser.add_argument("--unicode-normalize", dest="unicode_normalize", action="store_true")
    parser.add_argument("--no-unicode-normalize", dest="unicode_normalize", action="store_false")
    parser.set_defaults(unicode_normalize=False)
    parser.add_argument("--keep-extra-fields", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_len <= 0:
        raise ValueError("--min-len must be positive.")
    if args.max_len < args.min_len:
        raise ValueError("--max-len must be greater than or equal to --min-len.")
    if args.buckets <= 0:
        raise ValueError("--buckets must be positive.")
    output_dir, existing_state = prepare_output_dir(args)
    if existing_state is not None and existing_state.get("stage") == "completed" and report_path(output_dir).exists():
        print(report_path(output_dir).read_text(encoding="utf-8"))
        return

    if existing_state is not None and existing_state.get("cleaning"):
        clean_report = existing_state["cleaning"]
    else:
        write_state(
            output_dir,
            {
                "stage": "cleaning",
                "args_fingerprint": args_fingerprint(args),
                "input": [str(path) for path in args.input],
            },
        )
        clean_report = clean_to_candidate_shards(args)
        write_state(
            output_dir,
            {
                "stage": "deduping",
                "args_fingerprint": args_fingerprint(args),
                "input": [str(path) for path in args.input],
                "cleaning": clean_report,
            },
        )
    dedupe_report = dedupe_candidate_shards(args, clean_report)
    report = {
        "input": [str(path) for path in args.input],
        "output_dir": str(output_dir),
        "filters": {
            "text_key": args.text_key,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "min_cjk_ratio": args.min_cjk_ratio,
            "dedupe_mode": args.dedupe_mode,
            "buckets": args.buckets,
            "records_per_shard": args.records_per_shard,
            "long_text_mode": args.long_text_mode,
            "unicode_normalize": bool(args.unicode_normalize),
            "keep_extra_fields": bool(args.keep_extra_fields),
        },
        "cleaning": clean_report,
        "dedupe": dedupe_report,
        "kept_ratio": (
            round(dedupe_report["final_records"] / clean_report["total_lines"], 8)
            if clean_report["total_lines"]
            else 0
        ),
    }
    report_path(output_dir).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_state(
        output_dir,
        {
            "stage": "completed",
            "args_fingerprint": args_fingerprint(args),
            "input": [str(path) for path in args.input],
            "cleaning": clean_report,
            "dedupe": dedupe_report,
            "report": str(report_path(output_dir)),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
