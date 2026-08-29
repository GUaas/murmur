from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.pretrained_evaluation import run_pretrained_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and evaluate standalone Murmur pretrained weights."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "model" / "murmur_203m_best_weights_only.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        default=str(PROJECT_ROOT / "tokenizer" / "sp_unigram_32k.model"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "pretrained_203m"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generation-max-tokens", type=int, default=24)
    parser.add_argument("--standard-benchmarks", action="store_true")
    parser.add_argument("--benchmark-sample-count", type=int, default=24)
    parser.add_argument("--mmlu-per-subject", type=int, default=3)
    parser.add_argument("--ceval-per-subject", type=int, default=3)
    parser.add_argument("--choice-batch-size", type=int, default=4)
    parser.add_argument("--benchmark-cache-dir", default=None)
    parser.add_argument("--validation-cache-dir", default=None)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--validation-max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(max(1, int(args.torch_threads)))
    results = run_pretrained_evaluation(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        device_name=args.device,
        quick=args.quick,
        full_context=args.full_context,
        skip_generation=args.skip_generation,
        generation_max_tokens=max(1, args.generation_max_tokens),
        standard_benchmarks=args.standard_benchmarks,
        benchmark_sample_count=max(1, args.benchmark_sample_count),
        mmlu_per_subject=max(1, args.mmlu_per_subject),
        ceval_per_subject=max(1, args.ceval_per_subject),
        choice_batch_size=max(1, args.choice_batch_size),
        benchmark_cache_dir=args.benchmark_cache_dir,
        validation_cache_dir=args.validation_cache_dir,
        validation_batch_size=max(1, args.validation_batch_size),
        validation_max_batches=args.validation_max_batches,
    )
    print(f"完成，用时 {results['run']['wall_seconds']:.2f} 秒")
    print(f"JSON: {results['report_paths']['json']}")
    print(f"Markdown: {results['report_paths']['markdown']}")


if __name__ == "__main__":
    main()
