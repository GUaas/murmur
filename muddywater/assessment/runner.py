from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from muddywater.config import load_config
from muddywater.generation_runtime import load_generation_runtime

from .data import HFDatasetClient, build_benchmark_suite
from .efficiency import benchmark_runtime, current_process_rss_bytes
from .generation import run_generation_probes
from .probes import DOMAIN_TEXT_PROBES, GENERATION_PROBES, TOKENIZER_PROBES
from .scoring import score_choice_examples, score_text_probes, tokenizer_probe_stats


ProgressCallback = Callable[[str], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_integrity(root: Path) -> dict[str, Any]:
    checksum_path = root / "SHA256SUMS"
    mismatches = []
    missing = []
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        actual = _sha256(path)
        checked += 1
        if actual.lower() != expected.lower():
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return {
        "passed": not missing and not mismatches,
        "checksummed_files": checked,
        "missing": missing,
        "mismatches": mismatches,
        "model_sha256": _sha256(root / "model" / "model_119m.pt"),
        "tokenizer_sha256": _sha256(root / "tokenizer" / "sp_unigram_24k.model"),
    }


def tokenizer_vocabulary_profile(tokenizer: Any) -> dict[str, Any]:
    pieces = [tokenizer.processor.IdToPiece(index) for index in range(tokenizer.vocab_size)]
    byte_pattern = re.compile(r"^<0x[0-9A-F]{2}>$")
    cjk_pattern = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    latin_pattern = re.compile(r"[A-Za-z]")
    kana_pattern = re.compile(r"[\u3040-\u30ff]")
    hangul_pattern = re.compile(r"[\uac00-\ud7af]")
    cyrillic_pattern = re.compile(r"[\u0400-\u04ff]")
    arabic_pattern = re.compile(r"[\u0600-\u06ff]")
    devanagari_pattern = re.compile(r"[\u0900-\u097f]")
    byte_pieces = [piece for piece in pieces if byte_pattern.match(piece)]
    special_pieces = [
        piece for piece in pieces if piece.startswith("<") and piece.endswith(">") and piece not in byte_pieces
    ]
    learned = [piece for piece in pieces if piece not in byte_pieces and piece not in special_pieces]
    cjk = [piece for piece in learned if cjk_pattern.search(piece)]
    multi_cjk = [piece for piece in learned if len(cjk_pattern.findall(piece)) >= 2]
    return {
        "vocab_size": len(pieces),
        "special_piece_count": len(special_pieces),
        "byte_fallback_piece_count": len(byte_pieces),
        "learned_piece_count": len(learned),
        "cjk_piece_count": len(cjk),
        "cjk_piece_ratio_of_learned": len(cjk) / max(1, len(learned)),
        "multi_cjk_piece_count": len(multi_cjk),
        "latin_piece_count": sum(bool(latin_pattern.search(piece)) for piece in learned),
        "latin_piece_ratio_of_learned": sum(
            bool(latin_pattern.search(piece)) for piece in learned
        )
        / max(1, len(learned)),
        "learned_script_piece_counts": {
            "kana": sum(bool(kana_pattern.search(piece)) for piece in learned),
            "hangul": sum(bool(hangul_pattern.search(piece)) for piece in learned),
            "cyrillic": sum(bool(cyrillic_pattern.search(piece)) for piece in learned),
            "arabic": sum(bool(arabic_pattern.search(piece)) for piece in learned),
            "devanagari": sum(bool(devanagari_pattern.search(piece)) for piece in learned),
        },
        "special_pieces": special_pieces,
    }


def audit_existing_generation_reports(root: Path) -> dict[str, Any]:
    files = sorted((root / "reports" / "generation").glob("full_*.json"))
    results = []
    file_summaries = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_summaries[path.name] = payload.get("summary")
        results.extend(payload.get("results", []))
    question_prompt_count = sum(
        bool(str(item.get("prompt", "")))
        and set(str(item.get("prompt", ""))) <= {"?"}
        for item in results
    )
    scored = [item for item in results if item.get("expected") is not None]
    return {
        "report_files": [path.name for path in files],
        "num_generations": len(results),
        "length_finish_count": sum(item.get("finish_reason") == "length" for item in results),
        "eos_finish_count": sum(item.get("finish_reason") == "eos" for item in results),
        "repetition_ge_0_25_count": sum(
            float(item.get("repetition_ratio", 0.0)) >= 0.25 for item in results
        ),
        "literal_question_mark_prompt_count": question_prompt_count,
        "num_correctness_scored": len(scored),
        "file_summaries": file_summaries,
        "limitations": [
            "Generation configs and original prompt files referenced by absolute path are absent.",
            "Greedy and sampled reports use different prompts, so they are not a controlled A/B test.",
            "Held-out references were supplied but not used by the bundled generation scorer.",
        ],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fmt_percent(value: Any) -> str:
    return "NE" if value is None else f"{100.0 * float(value):.1f}%"


def render_summary_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Murmur 119M Comprehensive Evaluation — Reproducible Results",
        "",
        f"Generated: {results['run']['completed_at']}",
        "",
        "> These are resource-bounded local measurements, not official full benchmark submissions.",
        "",
        "## Release and training evidence",
        "",
        f"- Integrity: {'PASS' if results['integrity']['passed'] else 'FAIL'} "
        f"({results['integrity']['checksummed_files']} files)",
        f"- Parameters: {results['model']['parameters']:,}",
        f"- In-domain held-out PPL: {results['packaged_reports']['full_validation']['metrics']['perplexity']:.4f}",
        f"- Supervised training tokens: {results['packaged_reports']['training_summary']['supervised_train_tokens']:,}",
        "",
        "## Tokenizer profile",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]
    vocabulary = results["tokenizer"]["vocabulary_profile"]
    lines.extend(
        [
            f"| Vocabulary | {vocabulary['vocab_size']:,} |",
            f"| Learned pieces containing CJK | {_fmt_percent(vocabulary['cjk_piece_ratio_of_learned'])} |",
            f"| Learned pieces containing Latin | {_fmt_percent(vocabulary['latin_piece_ratio_of_learned'])} |",
            f"| Byte fallback pieces | {vocabulary['byte_fallback_piece_count']:,} |",
            "",
            "## Synthetic cross-domain language modeling",
            "",
            "| Domain | Language | Tokens | PPL | Bits/byte |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in results["cross_domain_language_modeling"]["probes"]:
        lines.append(
            f"| {item['category']} | {item['language']} | {item['tokens']} | "
            f"{item['perplexity']:.2f} | {item['bits_per_byte']:.3f} |"
        )

    lines.extend(["", "## Resource-bounded standard benchmarks", ""])
    if results.get("benchmarks"):
        lines.extend(
            [
                "| Task | N | Primary | Accuracy | Chance | 95% CI |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for name, item in results["benchmarks"].items():
            interval = item["primary_wilson_95"]
            lines.append(
                f"| {name} | {item['num_examples']} | {item['primary_metric']} | "
                f"{_fmt_percent(item['primary_accuracy'])} | {_fmt_percent(item['chance_accuracy'])} | "
                f"{_fmt_percent(interval[0])}–{_fmt_percent(interval[1])} |"
            )
    else:
        lines.append("Skipped.")

    lines.extend(["", "## Paired generation probes", ""])
    generation = results.get("generation")
    if generation:
        lines.extend(
            [
                "| Mode | N | Expected-hit | EOS/stop | Collapse ≥0.25 | tok/s |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode, item in generation["by_mode"].items():
            lines.append(
                f"| {mode} | {item['num_generations']} | "
                f"{_fmt_percent(item['contains_any_expected_rate'])} | "
                f"{_fmt_percent(item['eos_or_stop_rate'])} | "
                f"{_fmt_percent(item['collapse_rate_repetition_ge_0_25'])} | "
                f"{item['decode_tokens_per_second']:.2f} |"
            )
    else:
        lines.append("Skipped.")

    lines.extend(["", "## Local runtime", ""])
    efficiency = results.get("efficiency")
    if efficiency:
        lines.extend(
            [
                f"- Device: {efficiency['device']}",
                f"- Checkpoint: {efficiency['checkpoint_mib']:.2f} MiB",
                f"- FP32 parameter memory: {efficiency['parameter_mib_current_dtype']:.2f} MiB",
                f"- Greedy cached decode: {efficiency['greedy_cached_decode']['tokens_per_second']:.2f} token/s",
                f"- BF16 KV cache at 2048, batch 1: {efficiency['kv_cache']['bf16_mib_at_max_context_batch1']:.2f} MiB",
            ]
        )
    return "\n".join(lines) + "\n"


def run_comprehensive_assessment(
    root: str | Path,
    output_dir: str | Path,
    dataset_cache_dir: str | Path | None = None,
    sample_count: int = 160,
    mmlu_per_subject: int = 3,
    ceval_per_subject: int = 3,
    choice_batch_size: int = 4,
    generation_max_tokens: int = 48,
    quick: bool = False,
    skip_benchmarks: bool = False,
    skip_generation: bool = False,
    progress: ProgressCallback = print,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "evaluation_results.partial.json"
    started = datetime.now(timezone.utc).astimezone().isoformat()
    wall_start = time.perf_counter()
    results: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "started_at": started,
            "root": str(root),
            "output_dir": str(output_dir),
            "quick": bool(quick),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "initial_process_rss_bytes": current_process_rss_bytes(),
        },
    }

    progress("[1/7] Verifying release integrity")
    results["integrity"] = verify_release_integrity(root)
    if not results["integrity"]["passed"]:
        raise RuntimeError("Release integrity verification failed")

    training_summary = json.loads(
        (root / "reports" / "training_summary.json").read_text(encoding="utf-8")
    )
    full_validation = json.loads(
        (root / "reports" / "eval_best_full_val.json").read_text(encoding="utf-8")
    )
    results["packaged_reports"] = {
        "training_summary": training_summary,
        "full_validation": full_validation,
        "acceptance": json.loads(
            (root / "reports" / "acceptance_report.json").read_text(encoding="utf-8")
        ),
        "existing_generation_audit": audit_existing_generation_reports(root),
    }

    progress("[2/7] Loading checkpoint and tokenizer")
    config_path = root / "inference.yaml"
    config = load_config(config_path)
    config["__config_path__"] = str(config_path)
    load_start = time.perf_counter()
    runtime = load_generation_runtime(config)
    load_seconds = time.perf_counter() - load_start
    results["model"] = {
        "parameters": sum(parameter.numel() for parameter in runtime.model.parameters()),
        "config": vars(runtime.model.config),
        "checkpoint_load_seconds": load_seconds,
        "checkpoint_format": "murmur_model_weights_v1",
    }

    progress("[3/7] Auditing vocabulary and tokenizer efficiency")
    results["tokenizer"] = {
        "vocabulary_profile": tokenizer_vocabulary_profile(runtime.tokenizer),
        "probe_results": tokenizer_probe_stats(runtime.tokenizer, TOKENIZER_PROBES),
    }
    _write_json(partial_path, results)

    progress("[4/7] Scoring synthetic cross-domain text probes")
    results["cross_domain_language_modeling"] = score_text_probes(
        runtime.model,
        runtime.tokenizer,
        DOMAIN_TEXT_PROBES,
        runtime.device,
    )
    _write_json(partial_path, results)

    if not skip_benchmarks:
        progress("[5/7] Fetching and scoring standard benchmark samples")
        if quick:
            sample_count = min(sample_count, 24)
            mmlu_per_subject = min(mmlu_per_subject, 1)
            ceval_per_subject = min(ceval_per_subject, 1)
        client = HFDatasetClient(dataset_cache_dir or (output_dir / "dataset_cache"))
        suite, provenance = build_benchmark_suite(
            client,
            sample_count=sample_count,
            mmlu_per_subject=mmlu_per_subject,
            ceval_per_subject=ceval_per_subject,
        )
        results["benchmark_provenance"] = provenance
        results["benchmarks"] = {}
        for task_name, examples in suite.items():
            progress(f"      scoring {task_name}: {len(examples)} examples")
            results["benchmarks"][task_name] = score_choice_examples(
                runtime.model,
                runtime.tokenizer,
                examples,
                runtime.device,
                batch_size=choice_batch_size,
            )
            _write_json(partial_path, results)
        results["benchmark_provenance"]["dataset_requests"] = client.requests
    else:
        results["benchmarks"] = {}

    if not skip_generation:
        progress("[6/7] Running paired greedy/sampled generation probes")
        results["generation"] = run_generation_probes(
            runtime,
            GENERATION_PROBES if not quick else GENERATION_PROBES[:6],
            max_new_tokens=min(generation_max_tokens, 24) if quick else generation_max_tokens,
        )

    progress("[7/7] Benchmarking local runtime")
    results["efficiency"] = benchmark_runtime(
        runtime,
        sequence_lengths=(32, 128) if quick else (32, 128, 512, 1024, 2048),
        repeats=1 if quick else 2,
        generation_tokens=16 if quick else 32,
    )
    results["run"].update(
        {
            "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "wall_seconds": time.perf_counter() - wall_start,
            "final_process_rss_bytes": current_process_rss_bytes(),
        }
    )
    final_path = output_dir / "evaluation_results.json"
    _write_json(final_path, results)
    (output_dir / "evaluation_summary.md").write_text(
        render_summary_markdown(_json_safe(results)),
        encoding="utf-8",
    )
    if partial_path.exists():
        partial_path.unlink()
    return results
