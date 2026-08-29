from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.sft_data.v4_pipeline import V4Options, build_v4_dataset


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build the Murmur 203M SFT v4 mixture.")
    parser.add_argument("--base-root", type=Path, default=project_root / "data" / "sft_203m_v3")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", type=Path, default=project_root / "tokenizer" / "sp_unigram_32k.model"
    )
    parser.add_argument("--max-tokens", type=int, default=1_024)
    parser.add_argument("--validation-ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_v4_dataset(
        V4Options(
            base_root=args.base_root,
            output_root=args.output_root,
            tokenizer_path=args.tokenizer,
            max_tokens=args.max_tokens,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            download=not args.no_download,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

