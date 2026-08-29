from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.release_bundle import install_github_release_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a dedicated GitHub Release with gh, fully verify it, then "
            "atomically install its tokenizer and token cache."
        )
    )
    parser.add_argument("--repo", required=True, help="GitHub OWNER/REPO.")
    parser.add_argument("--tag", required=True, help="GitHub Release tag.")
    parser.add_argument("--data-root", default="data/full")
    parser.add_argument("--tokenizer-dir-name", default="tokenizer")
    parser.add_argument(
        "--cache-dir-name",
        default="token_cache_sp_unigram_24k_2048",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        default=None,
        help="Optional out-of-band SHA-256 for release_manifest.json.",
    )
    parser.add_argument("--gh", default="gh", help="GitHub CLI executable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_dir, cache_dir = install_github_release_bundle(
        repo=args.repo,
        tag=args.tag,
        data_root=args.data_root,
        tokenizer_dir_name=args.tokenizer_dir_name,
        cache_dir_name=args.cache_dir_name,
        expected_manifest_sha256=args.expected_manifest_sha256,
        gh_executable=args.gh,
    )
    print(f"Installed tokenizer: {tokenizer_dir}")
    print(f"Installed token cache: {cache_dir}")


if __name__ == "__main__":
    main()
