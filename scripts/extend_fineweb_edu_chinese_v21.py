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
    combine_completed_download_manifests,
    download_dataset_selection,
    list_parquet_files,
    resolve_dataset_revision,
    select_evenly_spaced_extension,
)


DEFAULT_REPO_ID = "opencsg/Fineweb-Edu-Chinese-V2.1"
DEFAULT_SUBSET = "4_5"
DEFAULT_TARGET_TOTAL_FILES = 4900
UPSTREAM_SCORE45_TOKENS = 46_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extend an existing pinned Fineweb-Edu-Chinese-V2.1 selection with "
            "balanced, non-overlapping files."
        )
    )
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--target-total-files", type=int, default=DEFAULT_TARGET_TOTAL_FILES)
    parser.add_argument("--base-manifest-name", default="download_manifest.json")
    parser.add_argument(
        "--extension-manifest-name",
        default="download_extension_manifest.json",
    )
    parser.add_argument(
        "--combined-manifest-name",
        default="download_combined_manifest.json",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--upstream-subset-tokens",
        type=int,
        default=UPSTREAM_SCORE45_TOKENS,
    )
    return parser.parse_args()


def load_completed_base_manifest(
    path: Path,
    *,
    repo_id: str,
    revision: str,
    subset: str,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Base download manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(
            f"Base download must complete before extension: "
            f"status={manifest.get('status')!r}, path={path}"
        )
    expected_dataset = {
        "repo_id": repo_id,
        "revision": revision,
        "subset": subset,
    }
    if manifest.get("dataset", {}) != expected_dataset:
        raise ValueError(
            f"Base manifest dataset mismatch: existing={manifest.get('dataset', {})}, "
            f"expected={expected_dataset}"
        )
    return manifest


def main() -> None:
    args = parse_args()
    root = Path(args.target_root).resolve()
    revision = resolve_dataset_revision(args.repo_id, revision=args.revision)
    base_manifest_path = root / args.base_manifest_name
    base_manifest = load_completed_base_manifest(
        base_manifest_path,
        repo_id=args.repo_id,
        revision=revision,
        subset=args.subset,
    )
    all_paths = list_parquet_files(
        repo_id=args.repo_id,
        subset=args.subset,
        revision=revision,
    )
    base_paths = [str(path) for path in base_manifest.get("selected_repo_paths", [])]
    extension_paths = select_evenly_spaced_extension(
        all_paths,
        base_paths,
        target_total_count=args.target_total_files,
    )
    estimated_combined_tokens = round(
        args.upstream_subset_tokens * args.target_total_files / len(all_paths)
    )
    extension_manifest = download_dataset_selection(
        repo_id=args.repo_id,
        revision=revision,
        subset=args.subset,
        selected_paths=extension_paths,
        all_paths=all_paths,
        target_root=root,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        upstream_subset_tokens=args.upstream_subset_tokens,
        manifest_name=args.extension_manifest_name,
        selection_strategy="centered_even_spacing_over_paths_not_in_base_manifest",
        selection_metadata={
            "base_manifest": args.base_manifest_name,
            "base_selected_files": len(base_paths),
            "target_total_files_after_extension": args.target_total_files,
            "estimated_combined_tokens_before_local_cleaning": estimated_combined_tokens,
        },
    )
    combined_manifest = combine_completed_download_manifests(
        [base_manifest_path, root / args.extension_manifest_name],
        root / args.combined_manifest_name,
    )
    print(
        json.dumps(
            {
                "status": combined_manifest["status"],
                "target_root": str(root),
                "revision": revision,
                "base_files": len(base_paths),
                "extension_files": len(extension_paths),
                "extension_completed_count": extension_manifest["completed_count"],
                "combined_files": combined_manifest["completed_count"],
                "estimated_combined_tokens_before_local_cleaning": (
                    combined_manifest["selection"][
                        "estimated_selected_tokens_before_local_cleaning"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
