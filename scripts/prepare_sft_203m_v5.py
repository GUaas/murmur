from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.sft_data.v5_continuation import (
    V5ContinuationOptions,
    build_v5_continuation_dataset,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build the Murmur 203M v5 new-data continuation mixture."
    )
    parser.add_argument("--base-root", type=Path, default=project_root / "data" / "sft_203m_v3")
    parser.add_argument(
        "--expanded-root", type=Path, default=project_root / "data" / "sft_203m_v4"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_v5_continuation_dataset(
        V5ContinuationOptions(
            base_root=args.base_root,
            expanded_root=args.expanded_root,
            output_root=args.output_root,
            anchor_fraction=args.anchor_fraction,
            seed=args.seed,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
