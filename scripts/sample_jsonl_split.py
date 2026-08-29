from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable


REPORT_OUTPUT_ROW_KEYS = ("output_rows", "kept_rows", "written_rows", "total_rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample exact train/validation JSONL splits from a large JSONL file."
    )
    parser.add_argument("--input", required=True, help="Source JSONL file.")
    parser.add_argument("--output-dir", default="data", help="Directory for sampled JSONL files.")
    parser.add_argument("--prefix", default="minipath_25g", help="Output file prefix.")
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--total-rows", type=int, default=None)
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report containing total_rows, used to avoid a counting pass.",
    )
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--progress", type=int, default=1000000)
    return parser.parse_args()


def load_total_rows(report_path: str | None, explicit_total: int | None) -> int | None:
    if explicit_total is not None:
        if explicit_total <= 0:
            raise ValueError("--total-rows must be positive")
        return explicit_total
    if not report_path:
        return None

    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    total_rows = next(
        (payload[key] for key in REPORT_OUTPUT_ROW_KEYS if payload.get(key) is not None),
        None,
    )
    if total_rows is None:
        return None
    total_rows = int(total_rows)
    if total_rows <= 0:
        raise ValueError(f"Invalid total_rows in report: {total_rows}")
    return total_rows


def count_lines(path: Path, progress: int) -> int:
    started = time.time()
    total = 0
    with path.open("rb") as handle:
        for total, _ in enumerate(handle, start=1):
            if progress > 0 and total % progress == 0:
                elapsed = time.time() - started
                print(f"counted_rows={total} elapsed={elapsed:.1f}s", flush=True)
    return total


def sample_assignments(
    total_rows: int,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[dict[int, str], list[int], list[int]]:
    requested = int(train_count) + int(val_count)
    if train_count <= 0 or val_count < 0:
        raise ValueError("train-count must be positive and val-count must be non-negative")
    if requested > total_rows:
        raise ValueError(f"Requested {requested} rows, but source only has {total_rows}")

    rng = random.Random(seed)
    selected = rng.sample(range(1, total_rows + 1), requested)
    rng.shuffle(selected)
    train_lines = selected[:train_count]
    val_lines = selected[train_count:]
    assignments = {line_no: "train" for line_no in train_lines}
    assignments.update({line_no: "val" for line_no in val_lines})
    return assignments, train_lines, val_lines


def parse_selected_line(line: bytes, path: Path, line_no: int, text_key: str) -> dict:
    try:
        obj = json.loads(line.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}:{line_no}")
    text = obj.get(text_key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing non-empty {text_key!r} at {path}:{line_no}")
    return obj


def collect_selected_records(
    path: Path,
    assignments: dict[int, str],
    text_key: str,
    progress: int,
) -> tuple[dict[int, dict], dict[str, int]]:
    wanted = set(assignments)
    records: dict[int, dict] = {}
    stats = {"seen_rows": 0, "selected_rows": 0}
    started = time.time()
    with path.open("rb") as handle:
        for line_no, line in enumerate(handle, start=1):
            stats["seen_rows"] = line_no
            if line_no in wanted:
                records[line_no] = parse_selected_line(line, path, line_no, text_key)
                stats["selected_rows"] += 1
                if stats["selected_rows"] == len(wanted):
                    break
            if progress > 0 and line_no % progress == 0:
                elapsed = time.time() - started
                print(
                    f"scanned_rows={line_no} selected={stats['selected_rows']} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    missing = wanted.difference(records)
    if missing:
        preview = sorted(missing)[:10]
        raise RuntimeError(f"Did not find {len(missing)} selected rows, first missing: {preview}")
    return records, stats


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for obj in records:
            handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def source_file_summary(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }


def write_split_outputs(
    output_dir: Path,
    prefix: str,
    records: dict[int, dict],
    train_lines: list[int],
    val_lines: list[int],
) -> tuple[Path, Path, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{prefix}_train_{len(train_lines)}.jsonl"
    val_path = output_dir / f"{prefix}_val_{len(val_lines)}.jsonl"
    train_written = write_jsonl(train_path, (records[line_no] for line_no in train_lines))
    val_written = write_jsonl(val_path, (records[line_no] for line_no in val_lines))
    return train_path, val_path, train_written, val_written


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    total_rows = load_total_rows(args.report, args.total_rows)
    if total_rows is None:
        total_rows = count_lines(input_path, progress=args.progress)

    assignments, train_lines, val_lines = sample_assignments(
        total_rows=total_rows,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
    )
    records, scan_stats = collect_selected_records(
        path=input_path,
        assignments=assignments,
        text_key=args.text_key,
        progress=args.progress,
    )
    train_path, val_path, train_written, val_written = write_split_outputs(
        output_dir=output_dir,
        prefix=args.prefix,
        records=records,
        train_lines=train_lines,
        val_lines=val_lines,
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_file_summary(input_path),
        "source_report": str(Path(args.report).resolve()) if args.report else None,
        "seed": args.seed,
        "total_rows": total_rows,
        "train_count": train_written,
        "val_count": val_written,
        "train_path": str(train_path.resolve()),
        "val_path": str(val_path.resolve()),
        "train_line_numbers": train_lines,
        "val_line_numbers": val_lines,
        "scan_stats": scan_stats,
    }
    manifest_path = output_dir / f"{args.prefix}_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("_line_numbers")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
