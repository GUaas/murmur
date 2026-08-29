from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


@dataclass(frozen=True)
class Shard:
    component: str
    path: Path
    records: int
    report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic, disjoint SentencePiece train/diagnostic samples "
            "from completed FineWeb cleaned-shard reports."
        )
    )
    parser.add_argument("--target-root", required=True)
    parser.add_argument(
        "--report",
        action="append",
        dest="reports",
        default=None,
        help="Cleaning report name relative to target-root; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=int, default=1_000_000)
    parser.add_argument("--diagnostic-size", type=int, default=20_000)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_report_shards(target_root: Path, report_names: Iterable[str]) -> list[Shard]:
    shards: list[Shard] = []
    for report_name in report_names:
        report_path = target_root / report_name
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"Cleaning report is not completed: {report_path}")
        output_dir = Path(str(payload["output_dir"]))
        component = output_dir.name
        report_shards = payload.get("output_shards")
        if not isinstance(report_shards, list) or not report_shards:
            raise ValueError(f"Cleaning report has no output_shards: {report_path}")
        for item in report_shards:
            path = output_dir / str(item["file"])
            records = int(item["records"])
            if records <= 0:
                raise ValueError(f"Invalid record count for {path}: {records}")
            if not path.is_file():
                raise FileNotFoundError(f"Cleaned shard is missing: {path}")
            shards.append(
                Shard(
                    component=component,
                    path=path,
                    records=records,
                    report_path=report_path,
                )
            )
    return shards


def allocate_quotas(weights: list[int], target: int) -> list[int]:
    total = sum(weights)
    if target < 0 or target > total:
        raise ValueError(f"Sample target {target} must be in [0, {total}]")
    if target == 0:
        return [0] * len(weights)
    exact = [target * weight / total for weight in weights]
    quotas = [math.floor(value) for value in exact]
    remainder = target - sum(quotas)
    order = sorted(
        range(len(weights)),
        key=lambda index: (exact[index] - quotas[index], weights[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        quotas[index] += 1
    if sum(quotas) != target:
        raise AssertionError("Quota allocation failed to preserve the sample target")
    return quotas


def stable_seed(seed: int, shard: Shard) -> int:
    identity = f"{seed}\0{shard.component}\0{shard.path.name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def read_text_record(line: str, path: Path, line_number: int) -> str:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing non-empty text at {path}:{line_number}")
    return text


def reservoir_sample_shard(shard: Shard, size: int, rng: random.Random) -> list[str]:
    sample: list[str] = []
    seen = 0
    with shard.path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            text = read_text_record(line, shard.path, line_number)
            seen += 1
            if len(sample) < size:
                sample.append(text)
                continue
            replacement = rng.randrange(seen)
            if replacement < size:
                sample[replacement] = text
    if seen != shard.records:
        raise ValueError(
            f"Shard record mismatch for {shard.path}: report={shard.records}, read={seen}"
        )
    if len(sample) != size:
        raise AssertionError(
            f"Shard sample mismatch for {shard.path}: expected={size}, got={len(sample)}"
        )
    return sample


def select_window(text: str, max_chars: int, rng: random.Random) -> str:
    if len(text) <= max_chars:
        return text
    start = rng.randrange(len(text) - max_chars + 1)
    return text[start : start + max_chars]


def write_records(
    handle: TextIO,
    texts: Iterable[str],
    max_chars: int,
    rng: random.Random,
) -> tuple[int, int]:
    records = 0
    chars = 0
    for text in texts:
        window = select_window(text, max_chars=max_chars, rng=rng)
        handle.write(json.dumps({"text": window}, ensure_ascii=False) + "\n")
        records += 1
        chars += len(window)
    return records, chars


def main() -> None:
    args = parse_args()
    if args.train_size <= 0 or args.diagnostic_size <= 0:
        raise ValueError("train-size and diagnostic-size must be positive")
    if args.max_chars <= 0:
        raise ValueError("max-chars must be positive")

    target_root = Path(args.target_root).resolve()
    report_names = args.reports or [
        "cleaning_report.json",
        "cleaning_extension_report.json",
    ]
    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "train.jsonl"
    diagnostic_path = output_dir / "diagnostic.jsonl"
    manifest_path = output_dir / "sample_manifest.json"
    existing = [path for path in (train_path, diagnostic_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing tokenizer samples: "
            + ", ".join(str(path) for path in existing)
        )

    shards = load_report_shards(target_root, report_names)
    weights = [shard.records for shard in shards]
    train_quotas = allocate_quotas(weights, args.train_size)
    diagnostic_quotas = allocate_quotas(weights, args.diagnostic_size)
    if any(
        train_quota + diagnostic_quota > shard.records
        for shard, train_quota, diagnostic_quota in zip(
            shards, train_quotas, diagnostic_quotas
        )
    ):
        raise ValueError("A shard sample quota exceeds its available records")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_staging = output_dir / "train.jsonl.staging"
    diagnostic_staging = output_dir / "diagnostic.jsonl.staging"
    if train_staging.exists() or diagnostic_staging.exists():
        raise FileExistsError(
            f"Staging tokenizer sample already exists under {output_dir}; inspect it first"
        )

    started = time.time()
    train_records = 0
    train_chars = 0
    diagnostic_records = 0
    diagnostic_chars = 0
    component_counts: dict[str, dict[str, int]] = {}
    try:
        with train_staging.open("x", encoding="utf-8", newline="\n") as train_handle, (
            diagnostic_staging.open("x", encoding="utf-8", newline="\n")
        ) as diagnostic_handle:
            for index, (shard, train_quota, diagnostic_quota) in enumerate(
                zip(shards, train_quotas, diagnostic_quotas),
                start=1,
            ):
                rng = random.Random(stable_seed(args.seed, shard))
                combined = reservoir_sample_shard(
                    shard,
                    size=train_quota + diagnostic_quota,
                    rng=rng,
                )
                rng.shuffle(combined)
                diagnostic_texts = combined[:diagnostic_quota]
                train_texts = combined[diagnostic_quota:]
                written, chars = write_records(
                    train_handle,
                    train_texts,
                    max_chars=args.max_chars,
                    rng=rng,
                )
                train_records += written
                train_chars += chars
                written, chars = write_records(
                    diagnostic_handle,
                    diagnostic_texts,
                    max_chars=args.max_chars,
                    rng=rng,
                )
                diagnostic_records += written
                diagnostic_chars += chars
                component = component_counts.setdefault(
                    shard.component,
                    {"source_records": 0, "train_records": 0, "diagnostic_records": 0},
                )
                component["source_records"] += shard.records
                component["train_records"] += train_quota
                component["diagnostic_records"] += diagnostic_quota
                if (
                    index == 1
                    or index == len(shards)
                    or index % args.progress_interval == 0
                ):
                    print(
                        f"sampled_shards={index}/{len(shards)} "
                        f"train={train_records}/{args.train_size} "
                        f"diagnostic={diagnostic_records}/{args.diagnostic_size} "
                        f"elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
        if train_records != args.train_size:
            raise AssertionError(
                f"Train sample mismatch: expected={args.train_size}, got={train_records}"
            )
        if diagnostic_records != args.diagnostic_size:
            raise AssertionError(
                "Diagnostic sample mismatch: "
                f"expected={args.diagnostic_size}, got={diagnostic_records}"
            )
        train_staging.replace(train_path)
        diagnostic_staging.replace(diagnostic_path)
    except BaseException:
        print(
            f"Sampling interrupted; staging files were retained under {output_dir}",
            file=sys.stderr,
            flush=True,
        )
        raise

    report_digests = {
        report_name: file_sha256(target_root / report_name)
        for report_name in report_names
    }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "seed": args.seed,
        "sampling": "per_shard_reservoir_without_replacement",
        "window_selection": "deterministic_uniform_character_window",
        "max_chars": args.max_chars,
        "source_records": sum(weights),
        "source_shards": len(shards),
        "source_reports": report_digests,
        "components": component_counts,
        "train": {
            "path": str(train_path),
            "records": train_records,
            "chars": train_chars,
            "bytes": train_path.stat().st_size,
            "sha256": file_sha256(train_path),
        },
        "diagnostic": {
            "path": str(diagnostic_path),
            "records": diagnostic_records,
            "chars": diagnostic_chars,
            "bytes": diagnostic_path.stat().st_size,
            "sha256": file_sha256(diagnostic_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
