from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.text_simplification.preflight import validate_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate text-simplification SFT inputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft_203m_text_simplification.yaml"),
    )
    parser.add_argument("--allow-missing-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_setup(
        args.config,
        require_checkpoint=not args.allow_missing_checkpoint,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
