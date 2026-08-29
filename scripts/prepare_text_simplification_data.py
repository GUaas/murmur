from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.text_simplification import PreparationOptions, prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic source/target JSONL for text-simplification SFT."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSON or JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-key", default="data")
    parser.add_argument("--target-key", default="s")
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--source-label", default="<|im_start|>")
    parser.add_argument("--target-label", default="<|im_end|>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = PreparationOptions(
        input_source_key=args.source_key,
        input_target_key=args.target_key,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        source_label=args.source_label,
        target_label=args.target_label,
    )
    report = prepare_dataset(args.input, args.output_dir, options)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
