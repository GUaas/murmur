from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.assessment.runner import run_comprehensive_assessment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible comprehensive Murmur assessment suite."
    )
    parser.add_argument("--root", default=str(PACKAGE_ROOT))
    parser.add_argument("--output-dir", default="reports/comprehensive_eval")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--sample-count", type=int, default=160)
    parser.add_argument("--mmlu-per-subject", type=int, default=3)
    parser.add_argument("--ceval-per-subject", type=int, default=3)
    parser.add_argument("--choice-batch-size", type=int, default=4)
    parser.add_argument("--generation-max-tokens", type=int, default=48)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(max(1, int(args.torch_threads)))
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    dataset_cache_dir = Path(args.dataset_cache_dir) if args.dataset_cache_dir else None
    if dataset_cache_dir is not None and not dataset_cache_dir.is_absolute():
        dataset_cache_dir = root / dataset_cache_dir
    results = run_comprehensive_assessment(
        root=root,
        output_dir=output_dir,
        dataset_cache_dir=dataset_cache_dir,
        sample_count=max(1, int(args.sample_count)),
        mmlu_per_subject=max(1, int(args.mmlu_per_subject)),
        ceval_per_subject=max(1, int(args.ceval_per_subject)),
        choice_batch_size=max(1, int(args.choice_batch_size)),
        generation_max_tokens=max(1, int(args.generation_max_tokens)),
        quick=bool(args.quick),
        skip_benchmarks=bool(args.skip_benchmarks),
        skip_generation=bool(args.skip_generation),
    )
    print(f"Completed in {results['run']['wall_seconds']:.2f}s")
    print(f"Results: {output_dir / 'evaluation_results.json'}")
    print(f"Summary: {output_dir / 'evaluation_summary.md'}")


if __name__ == "__main__":
    main()
