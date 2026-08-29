from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.hf_dataset import (
    download_dataset_selection,
    list_parquet_files,
    resolve_dataset_revision,
    select_evenly_spaced,
)


DEFAULT_REPO_ID = "opencsg/Fineweb-Edu-Chinese-V2.1"
DEFAULT_SUBSET = "4_5"
DEFAULT_SELECTED_FILES = 2600
UPSTREAM_SCORE45_TOKENS = 46_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a source-balanced, immutable selection of OpenCSG "
            "Fineweb-Edu-Chinese-V2.1 score_4_5 Parquet files."
        )
    )
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument(
        "--revision",
        default=None,
        help="Commit SHA or tag. Omit to resolve the current revision and pin it in the manifest.",
    )
    parser.add_argument("--selected-files", type=int, default=DEFAULT_SELECTED_FILES)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--upstream-subset-tokens",
        type=int,
        default=UPSTREAM_SCORE45_TOKENS,
        help="Approximate upstream token count, used only for planning metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision = resolve_dataset_revision(args.repo_id, revision=args.revision)
    all_paths = list_parquet_files(
        repo_id=args.repo_id,
        subset=args.subset,
        revision=revision,
    )
    selected_paths = select_evenly_spaced(all_paths, count=args.selected_files)
    manifest = download_dataset_selection(
        repo_id=args.repo_id,
        revision=revision,
        subset=args.subset,
        selected_paths=selected_paths,
        all_paths=all_paths,
        target_root=args.target_root,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        upstream_subset_tokens=args.upstream_subset_tokens,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "target_root": str(Path(args.target_root).resolve()),
                "repo_id": args.repo_id,
                "revision": revision,
                "selected_files": len(selected_paths),
                "completed_count": manifest["completed_count"],
                "completed_bytes": manifest["completed_bytes"],
                "estimated_selected_tokens_before_local_cleaning": manifest[
                    "selection"
                ]["estimated_selected_tokens_before_local_cleaning"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
