from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.text_simplification.modelscope_assets import (
    download_snapshot,
    find_checkpoint,
    install_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the Murmur base checkpoint from ModelScope.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--checkpoint-name", default="murmur_203m_best_weights_only.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = download_snapshot(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    source = find_checkpoint(snapshot, args.checkpoint_name)
    report = install_checkpoint(
        source,
        args.output,
        expected_sha256=args.expected_sha256,
        overwrite=args.overwrite,
    )
    report.update(
        {
            "model_id": args.model_id,
            "revision": args.revision or "default",
            "snapshot": str(snapshot),
            "source": str(source),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
