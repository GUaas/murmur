from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.release_bundle import build_release_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a flat GitHub Release upload directory from a completed "
            "SentencePiece model and token cache."
        )
    )
    parser.add_argument("--tokenizer", required=True, help="Completed SentencePiece .model path.")
    parser.add_argument("--cache-dir", required=True, help="Completed token-cache directory.")
    parser.add_argument("--output-dir", required=True, help="New/empty release upload directory.")
    parser.add_argument(
        "--materialize",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hard links (no duplicate bytes, same filesystem) or stream copies.",
    )
    parser.add_argument(
        "--no-export-vocab-tsv",
        action="store_true",
        help="Do not export the default id/piece/score/type vocabulary TSV.",
    )
    parser.add_argument(
        "--no-native-vocab",
        action="store_true",
        help="Do not include the trainer-generated sibling .vocab file when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_release_bundle(
        tokenizer_model=args.tokenizer,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        materialize=args.materialize,
        export_vocab_tsv=not args.no_export_vocab_tsv,
        include_native_vocab=not args.no_native_vocab,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
