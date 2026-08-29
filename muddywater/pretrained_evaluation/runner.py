from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from muddywater.assessment.data import HFDatasetClient, build_benchmark_suite
from muddywater.assessment.generation import run_generation_probes
from muddywater.assessment.runner import tokenizer_vocabulary_profile
from muddywater.assessment.scoring import score_choice_examples, score_text_probes
from muddywater.generation_runtime import GenerationRuntime
from muddywater.utils import resolve_device

from .artifacts import load_artifacts
from .checks import benchmark_forward, run_functional_checks
from .heldout import evaluate_heldout_cache
from .probes import GENERATION_PROBES, TEXT_PROBES
from .reporting import write_partial_report, write_reports


Progress = Callable[[str], None]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def run_pretrained_evaluation(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    quick: bool = False,
    full_context: bool = False,
    skip_generation: bool = False,
    generation_max_tokens: int = 24,
    standard_benchmarks: bool = False,
    benchmark_sample_count: int = 24,
    mmlu_per_subject: int = 3,
    ceval_per_subject: int = 3,
    choice_batch_size: int = 4,
    benchmark_cache_dir: str | Path | None = None,
    validation_cache_dir: str | Path | None = None,
    validation_batch_size: int = 1,
    validation_max_batches: int | None = None,
    progress: Progress = print,
) -> dict[str, Any]:
    started_at = _now()
    wall_start = time.perf_counter()
    device = resolve_device(device_name)
    results: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "started_at": started_at,
            "device": str(device),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "quick": bool(quick),
        },
    }

    progress("[1/6] 安全读取权重并执行严格架构加载")
    artifacts = load_artifacts(checkpoint_path, tokenizer_path, device)
    results["artifact"] = artifacts.artifact_audit

    progress("[2/6] 检查数值、因果掩码、KV 缓存与上下文边界")
    results["functional_checks"] = run_functional_checks(
        artifacts.model,
        artifacts.tokenizer,
        device,
        full_context=full_context,
    )

    progress("[3/6] 审计 tokenizer 并计算跨领域短文本损失")
    results["tokenizer_profile"] = tokenizer_vocabulary_profile(artifacts.tokenizer)
    text_probes = TEXT_PROBES[:3] if quick else TEXT_PROBES
    results["text_quality"] = score_text_probes(
        artifacts.model,
        artifacts.tokenizer,
        text_probes,
        device,
    )

    if validation_cache_dir is not None:
        progress("      复测训练时同源验证缓存")
        results["heldout_validation"] = evaluate_heldout_cache(
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            tokenizer_path=artifacts.tokenizer_path,
            checkpoint_config=artifacts.checkpoint_metadata.get("config", {}),
            cache_dir=validation_cache_dir,
            device=device,
            batch_size=validation_batch_size,
            max_batches=validation_max_batches,
        )
    else:
        results["heldout_validation"] = None

    if skip_generation:
        results["generation"] = {}
    else:
        progress("[4/6] 运行配对的贪心/采样生成探针")
        runtime = GenerationRuntime(
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            device=device,
            generation_config={
                "apply_chat_template": False,
                "return_full_text": False,
                "skip_special_tokens": True,
                "use_cache": True,
            },
            checkpoint_config=artifacts.checkpoint_metadata.get("config", {}),
            checkpoint_path=artifacts.checkpoint_path,
            tokenizer_path=artifacts.tokenizer_path,
            add_bos=True,
        )
        generation_probes = GENERATION_PROBES[:2] if quick else GENERATION_PROBES
        results["generation"] = run_generation_probes(
            runtime,
            generation_probes,
            max_new_tokens=min(8, generation_max_tokens) if quick else max(1, generation_max_tokens),
        )

    if standard_benchmarks:
        progress("[5/6] 下载并评分资源受限标准基准样本")
        cache_dir = Path(benchmark_cache_dir or Path(output_dir) / "dataset_cache")
        client = HFDatasetClient(cache_dir)
        suite, provenance = build_benchmark_suite(
            client,
            sample_count=max(1, benchmark_sample_count),
            mmlu_per_subject=1 if quick else max(1, mmlu_per_subject),
            ceval_per_subject=1 if quick else max(1, ceval_per_subject),
        )
        results["benchmarks"] = {}
        for name, examples in suite.items():
            progress(f"      {name}: {len(examples)} 条")
            results["benchmarks"][name] = score_choice_examples(
                artifacts.model,
                artifacts.tokenizer,
                examples,
                device,
                batch_size=max(1, int(choice_batch_size)),
            )
            write_partial_report(output_dir, results)
        provenance["dataset_requests"] = client.requests
        results["benchmark_provenance"] = provenance
    else:
        results["benchmarks"] = {}

    progress("[6/6] 测量当前设备上的前向性能并生成报告")
    lengths = (8, 32) if quick else (8, 32, 128, 512)
    batch_sizes = (1,) if quick else (1, 2, 4)
    results["efficiency"] = benchmark_forward(
        artifacts.model,
        artifacts.tokenizer,
        device,
        sequence_lengths=lengths,
        batch_sizes=batch_sizes,
        repeats=1 if quick else 2,
    )
    results["coverage"] = {
        "heldout_validation_retested": results["heldout_validation"] is not None,
        "full_context_executed": bool(full_context),
        "standard_benchmarks_executed": bool(standard_benchmarks),
        "generation_executed": not skip_generation,
        "limitations": [
            (
                "The original held-out token cache was not supplied, so recorded best_val_loss is not independently reproduced."
                if results["heldout_validation"] is None
                else "Held-out metrics are reproducible only when the supplied cache is the exact training-time validation split."
            ),
            "Hardware results describe only the current local runtime and device.",
        ],
    }
    results["run"].update(
        {"completed_at": _now(), "wall_seconds": time.perf_counter() - wall_start}
    )
    output_path = Path(output_dir)
    results["report_paths"] = {
        "json": str(output_path / "evaluation_results.json"),
        "markdown": str(output_path / "evaluation_summary.md"),
    }
    write_reports(output_path, results)
    partial_path = output_path / "evaluation_results.partial.json"
    if partial_path.exists():
        partial_path.unlink()
    return results
