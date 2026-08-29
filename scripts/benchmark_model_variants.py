from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config
from muddywater.training_benchmark import (
    TrainingBenchmarkSettings,
    choose_recommended_variant,
    release_benchmark_memory,
    run_training_benchmark,
)
from muddywater.utils import atomic_write_text


DEFAULT_CONFIGS = [
    PROJECT_ROOT / "configs" / "pretrain_fineweb_v21_20b_540m_wide.yaml",
    PROJECT_ROOT / "configs" / "pretrain_fineweb_v21_20b_540m_deep.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full forward/backward/optimizer throughput for the parameter-matched "
            "540M wide and deep architectures."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[str(path) for path in DEFAULT_CONFIGS],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        choices=["config", "on", "off"],
        default="config",
    )
    parser.add_argument("--target-tokens-per-second", type=float, default=60_000.0)
    parser.add_argument(
        "--production-margin-tokens-per-second",
        type=float,
        default=65_000.0,
    )
    parser.add_argument("--planned-train-tokens", type=int, default=20_000_000_000)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "amd_540m_variant_benchmark.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TrainingBenchmarkSettings(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        measured_steps=args.steps,
        target_tokens_per_second=args.target_tokens_per_second,
        production_margin_tokens_per_second=args.production_margin_tokens_per_second,
        planned_train_tokens=args.planned_train_tokens,
        compile_mode=args.compile_mode,
    )
    settings.validate()
    results = []
    had_error = False
    for raw_path in args.configs:
        config_path = Path(raw_path).resolve()
        print(f"benchmark_start config={config_path}", flush=True)
        try:
            result = run_training_benchmark(
                config=load_config(config_path),
                config_path=config_path,
                settings=settings,
            )
            print(
                f"benchmark_complete variant={result['variant']} "
                f"tokens_per_second={result['tokens_per_second']:.1f} "
                f"peak_GiB={result['memory']['peak_allocated_bytes'] / 2**30:.2f}",
                flush=True,
            )
        except Exception as exc:
            had_error = True
            result = {
                "status": "error",
                "variant": (
                    "deep"
                    if "deep" in config_path.stem.lower()
                    else "wide"
                    if "wide" in config_path.stem.lower()
                    else config_path.stem
                ),
                "config": str(config_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(
                f"benchmark_failed variant={result['variant']} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
        results.append(result)
        release_benchmark_memory()

    recommendation = choose_recommended_variant(
        results,
        target_tokens_per_second=settings.target_tokens_per_second,
    )
    report = {
        "status": "incomplete" if had_error else "completed",
        "decision_rule": (
            "Use deep when deep reaches target; otherwise use wide when wide reaches "
            "target; otherwise reduce planned tokens or change the runtime setup."
        ),
        "recommended_variant": recommendation,
        "results": results,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if had_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
