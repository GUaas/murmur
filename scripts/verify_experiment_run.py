from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.utils import atomic_write_text


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TRAIN_PATTERN = re.compile(rf"\bstep=(\d+)\s+ce_loss=({FLOAT_PATTERN})")
EVAL_PATTERN = re.compile(rf"\beval step=(\d+)\s+val_ce_loss=({FLOAT_PATTERN})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the bounded real-data experiment demonstrated stable loss descent."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiment_loss_descent_10k_5epoch",
        help="Completed training output directory.",
    )
    parser.add_argument("--min-train-tokens", type=int, default=22_000_000)
    parser.add_argument("--train-window-logs", type=int, default=5)
    parser.add_argument("--max-final-train-ce", type=float, default=6.8)
    parser.add_argument("--min-train-ce-drop", type=float, default=1.5)
    parser.add_argument("--max-best-val-ce", type=float, default=8.0)
    parser.add_argument("--min-val-ce-drop", type=float, default=0.3)
    parser.add_argument(
        "--report",
        default=None,
        help="Report path; defaults to <output-dir>/acceptance_report.json.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def parse_loss_history(path: Path) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    train_history: list[dict[str, float | int]] = []
    val_history: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        train_match = TRAIN_PATTERN.search(line)
        if train_match:
            train_history.append(
                {"step": int(train_match.group(1)), "ce_loss": float(train_match.group(2))}
            )
        eval_match = EVAL_PATTERN.search(line)
        if eval_match:
            val_history.append(
                {"step": int(eval_match.group(1)), "ce_loss": float(eval_match.group(2))}
            )
    return train_history, val_history


def check(name: str, passed: bool, actual: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "requirement": requirement,
    }


def allowed_overflow_skips(precision: str, attempted_steps: int) -> int:
    if precision == "fp16":
        return max(1, math.floor(max(0, attempted_steps) * 0.01))
    return 0


def build_acceptance_report(
    summary: dict[str, Any],
    train_history: list[dict[str, float | int]],
    val_history: list[dict[str, float | int]],
    *,
    min_train_tokens: int,
    train_window_logs: int,
    max_final_train_ce: float,
    min_train_ce_drop: float,
    max_best_val_ce: float,
    min_val_ce_drop: float,
) -> dict[str, Any]:
    window = max(1, int(train_window_logs))
    enough_train_logs = len(train_history) >= 2 * window
    finite_train = bool(train_history) and all(
        math.isfinite(float(item["ce_loss"])) for item in train_history
    )
    finite_val = bool(val_history) and all(
        math.isfinite(float(item["ce_loss"])) for item in val_history
    )

    initial_train_mean = (
        fmean(float(item["ce_loss"]) for item in train_history[:window])
        if enough_train_logs
        else None
    )
    final_train_mean = (
        fmean(float(item["ce_loss"]) for item in train_history[-window:])
        if enough_train_logs
        else None
    )
    train_drop = (
        initial_train_mean - final_train_mean
        if initial_train_mean is not None and final_train_mean is not None
        else None
    )

    first_val_ce = float(val_history[0]["ce_loss"]) if val_history else None
    best_val_ce_raw = summary.get("best_val_loss")
    best_val_ce = float(best_val_ce_raw) if best_val_ce_raw is not None else None
    val_drop = (
        first_val_ce - best_val_ce
        if first_val_ce is not None and best_val_ce is not None
        else None
    )

    stability = summary.get("stability", {})
    runtime = summary.get("runtime", {})
    precision = str(runtime.get("precision", "unknown")).lower()
    attempted_steps = int(stability.get("attempted_optimizer_steps", 0))
    overflow_skips = int(stability.get("amp_overflow_skips", 0))
    overflow_limit = allowed_overflow_skips(precision, attempted_steps)
    nonfinite_skips = int(stability.get("nonfinite_gradient_skips", 0))
    supervised_tokens = int(summary.get("supervised_train_tokens", 0))
    target_reached = bool(
        summary.get("target_reached", summary.get("status") == "completed")
    )
    termination_reason = summary.get("termination_reason")

    checks = [
        check("run_completed", summary.get("status") == "completed", summary.get("status"), "completed"),
        check(
            "training_target_reached",
            target_reached,
            {
                "target_reached": target_reached,
                "termination_reason": termination_reason,
            },
            "target_reached == true",
        ),
        check(
            "supervised_train_tokens",
            supervised_tokens >= int(min_train_tokens),
            supervised_tokens,
            f">= {int(min_train_tokens)}",
        ),
        check(
            "train_log_coverage",
            enough_train_logs,
            len(train_history),
            f">= {2 * window} finite logged train windows",
        ),
        check("finite_train_loss", finite_train, finite_train, "all logged train CE values finite"),
        check("finite_val_loss", finite_val, finite_val, "at least one finite validation CE value"),
        check(
            "final_train_ce",
            final_train_mean is not None and final_train_mean < float(max_final_train_ce),
            final_train_mean,
            f"last {window} logged windows mean < {float(max_final_train_ce)}",
        ),
        check(
            "train_ce_drop",
            train_drop is not None and train_drop >= float(min_train_ce_drop),
            train_drop,
            f"first-to-last window mean drop >= {float(min_train_ce_drop)}",
        ),
        check(
            "best_validation_ce",
            best_val_ce is not None and math.isfinite(best_val_ce) and best_val_ce < float(max_best_val_ce),
            best_val_ce,
            f"< {float(max_best_val_ce)}",
        ),
        check(
            "validation_ce_drop",
            val_drop is not None and val_drop >= float(min_val_ce_drop),
            val_drop,
            f"first eval minus best eval >= {float(min_val_ce_drop)}",
        ),
        check("nonfinite_gradient_skips", nonfinite_skips == 0, nonfinite_skips, "== 0"),
        check(
            "amp_overflow_skips",
            overflow_skips <= overflow_limit,
            overflow_skips,
            f"<= {overflow_limit} for precision={precision}",
        ),
    ]
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "checks": checks,
        "metrics": {
            "train_log_entries": len(train_history),
            "validation_log_entries": len(val_history),
            "initial_train_ce_mean": initial_train_mean,
            "final_train_ce_mean": final_train_mean,
            "train_ce_drop": train_drop,
            "first_validation_ce": first_val_ce,
            "best_validation_ce": best_val_ce,
            "validation_ce_drop": val_drop,
            "supervised_train_tokens": supervised_tokens,
            "precision": precision,
            "attempted_optimizer_steps": attempted_steps,
            "amp_overflow_skips": overflow_skips,
            "nonfinite_gradient_skips": nonfinite_skips,
            "target_reached": target_reached,
            "termination_reason": termination_reason,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    summary_path = output_dir / "training_summary.json"
    log_path = output_dir / "train.log"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    summary = read_json(summary_path)
    train_history, val_history = parse_loss_history(log_path)
    report = build_acceptance_report(
        summary,
        train_history,
        val_history,
        min_train_tokens=args.min_train_tokens,
        train_window_logs=args.train_window_logs,
        max_final_train_ce=args.max_final_train_ce,
        min_train_ce_drop=args.min_train_ce_drop,
        max_best_val_ce=args.max_best_val_ce,
        min_val_ce_drop=args.min_val_ce_drop,
    )
    report.update(
        {
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
            "log_path": str(log_path),
        }
    )
    report_path = Path(args.report) if args.report else output_dir / "acceptance_report.json"
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2),
        overwrite=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Acceptance report saved to: {report_path}", flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
