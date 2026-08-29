from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.corpus_text import (
    RunningStats,
    SimHashDeduper,
    cjk_ratio,
    clean_text,
    dedupe_digest,
    hamming_distance,
    normalized_dedupe_text,
    quality_reject_reason,
    simhash64,
    split_long_text,
    text_digest,
)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_v2{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean pretraining JSONL data with a text field.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", default=None, help="Output cleaned JSONL file.")
    parser.add_argument("--report", default=None, help="Output cleaning report JSON file.")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--min-len", type=int, default=300)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument(
        "--long-text-mode",
        choices=["chunk", "drop"],
        default="chunk",
        help="chunk preserves long documents by splitting them; drop keeps the legacy behavior.",
    )
    parser.add_argument("--min-cjk-ratio", type=float, default=0.2)
    parser.add_argument(
        "--dedupe-mode",
        choices=["normalized", "exact", "none"],
        default="normalized",
        help="normalized ignores punctuation/spacing/case before hashing; exact is byte-level after cleaning.",
    )
    parser.add_argument(
        "--near-dedupe-threshold",
        type=int,
        default=0,
        help="Enable SimHash near-duplicate filtering with this Hamming threshold; 0 disables it.",
    )
    parser.add_argument("--max-repeated-line-ratio", type=float, default=0.8)
    parser.add_argument("--max-url-ratio", type=float, default=0.4)
    parser.add_argument("--max-html-ratio", type=float, default=0.2)
    parser.add_argument("--max-punctuation-ratio", type=float, default=0.8)
    parser.add_argument("--max-digit-ratio", type=float, default=0.8)
    parser.add_argument(
        "--unicode-normalize",
        dest="unicode_normalize",
        action="store_true",
        help="Apply NFKC normalization. This may convert Chinese punctuation to ASCII punctuation.",
    )
    parser.add_argument(
        "--no-unicode-normalize",
        dest="unicode_normalize",
        action="store_false",
        help="Do not apply NFKC normalization. This is the default for Chinese pretraining data.",
    )
    parser.set_defaults(unicode_normalize=False)
    parser.add_argument("--keep-extra-fields", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.min_len <= 0:
        raise ValueError("--min-len must be positive.")
    if args.max_len < args.min_len:
        raise ValueError("--max-len must be greater than or equal to --min-len.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    drop_reasons: Counter[str] = Counter()
    raw_length_stats = RunningStats()
    clean_length_stats = RunningStats()
    kept_length_stats = RunningStats()
    seen: set[bytes] = set()
    near_deduper = (
        SimHashDeduper(args.near_dedupe_threshold)
        if int(args.near_dedupe_threshold) > 0
        else None
    )
    transform_stats: Counter[str] = Counter()

    total_lines = 0
    kept = 0
    normalize_unicode = args.unicode_normalize

    with input_path.open("r", encoding="utf-8-sig") as fin, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            total_lines += 1
            line = line.strip()
            if not line:
                drop_reasons["empty_line"] += 1
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                drop_reasons["invalid_json"] += 1
                continue

            value = obj.get(args.text_key)
            if not isinstance(value, str):
                drop_reasons["missing_or_non_string_text"] += 1
                continue

            raw_length_stats.add(len(value))
            text = clean_text(value, normalize_unicode=normalize_unicode)
            clean_length_stats.add(len(text))

            if not text:
                drop_reasons["empty_text_after_clean"] += 1
                continue
            if "\ufffd" in text:
                drop_reasons["replacement_character"] += 1
                continue

            if len(text) > args.max_len:
                if args.long_text_mode == "drop":
                    drop_reasons["too_long"] += 1
                    continue
                candidate_texts = split_long_text(text, max_len=args.max_len, min_len=args.min_len)
                if not candidate_texts:
                    drop_reasons["too_long_no_valid_chunk"] += 1
                    continue
                transform_stats["long_documents_chunked"] += 1
                transform_stats["chunks_emitted_from_long_documents"] += len(candidate_texts)
            else:
                candidate_texts = [text]

            for candidate_text in candidate_texts:
                if len(candidate_text) < args.min_len:
                    drop_reasons["too_short"] += 1
                    continue
                if len(candidate_text) > args.max_len:
                    drop_reasons["too_long"] += 1
                    continue
                if cjk_ratio(candidate_text) < args.min_cjk_ratio:
                    drop_reasons["low_cjk_ratio"] += 1
                    continue

                quality_reason = quality_reject_reason(
                    candidate_text,
                    max_repeated_line_ratio=float(args.max_repeated_line_ratio),
                    max_url_ratio=float(args.max_url_ratio),
                    max_html_ratio=float(args.max_html_ratio),
                    max_punctuation_ratio=float(args.max_punctuation_ratio),
                    max_digit_ratio=float(args.max_digit_ratio),
                )
                if quality_reason:
                    drop_reasons[quality_reason] += 1
                    continue

                digest = dedupe_digest(candidate_text, args.dedupe_mode)
                if digest is not None:
                    if digest in seen:
                        drop_reasons[f"duplicate_{args.dedupe_mode}"] += 1
                        continue
                    seen.add(digest)
                if near_deduper is not None and near_deduper.is_duplicate(candidate_text):
                    drop_reasons["duplicate_simhash"] += 1
                    continue

                kept_obj = dict(obj) if args.keep_extra_fields else {}
                kept_obj[args.text_key] = candidate_text
                fout.write(json.dumps(kept_obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                kept += 1
                kept_length_stats.add(len(candidate_text))

            if args.progress_interval > 0 and total_lines % args.progress_interval == 0:
                print(f"processed={total_lines} kept={kept} dropped={sum(drop_reasons.values())}")

    dropped = sum(drop_reasons.values())
    processed_items = kept + dropped
    report = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "total_lines": total_lines,
        "kept": kept,
        "dropped": dropped,
        "kept_ratio": round(kept / processed_items, 6) if processed_items else 0,
        "drop_reasons": dict(drop_reasons),
        "filters": {
            "text_key": args.text_key,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "min_cjk_ratio": args.min_cjk_ratio,
            "unicode_normalize": normalize_unicode,
            "dedupe_mode": args.dedupe_mode,
            "near_dedupe_threshold": args.near_dedupe_threshold,
            "max_repeated_line_ratio": args.max_repeated_line_ratio,
            "max_url_ratio": args.max_url_ratio,
            "max_html_ratio": args.max_html_ratio,
            "max_punctuation_ratio": args.max_punctuation_ratio,
            "max_digit_ratio": args.max_digit_ratio,
            "long_text_mode": args.long_text_mode,
            "keep_extra_fields": args.keep_extra_fields,
        },
        "transforms": dict(transform_stats),
        "raw_length": raw_length_stats.as_dict(),
        "clean_length": clean_length_stats.as_dict(),
        "kept_length": kept_length_stats.as_dict(),
        "input_size_bytes": input_path.stat().st_size,
        "output_size_bytes": output_path.stat().st_size,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if kept == 0 or not math.isfinite(report["kept_ratio"]):
        raise RuntimeError("No samples kept after cleaning.")


if __name__ == "__main__":
    main()
