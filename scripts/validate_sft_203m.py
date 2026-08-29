from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.sft_data.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the curated Murmur 203M SFT JSONL files.")
    parser.add_argument("--root", type=Path, default=Path(r"D:\datasets\sft_203m"))
    parser.add_argument("--sample-size", type=int, default=2_000)
    args = parser.parse_args()
    report = validate_dataset(
        args.root / "processed",
        args.root / "assets" / "sp_unigram_32k.model",
        sample_size=args.sample_size,
    )
    report_path = args.root / "reports" / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
