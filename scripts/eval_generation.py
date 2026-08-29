from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config
from muddywater.generation_runtime import (
    generate_from_runtime,
    load_generation_runtime,
    render_generation_prompt,
)
from muddywater.utils import enable_torch_backends, set_seed
from muddywater.paths import resolve_config_path, resolve_path


DEFAULT_PROMPTS = [
    {"id": "cn_story", "prompt": "\u8bf7\u7eed\u5199\u4e00\u6bb5\u96e8\u540e\u6c5f\u5357\u5c0f\u9547\u7684\u573a\u666f\u3002"},
    {"id": "cn_explain", "prompt": "\u7528\u7b80\u6d01\u4e2d\u6587\u89e3\u91ca\u4ec0\u4e48\u662f\u68af\u5ea6\u4e0b\u964d\u3002"},
    {"id": "qa_format", "prompt": "\u95ee\u9898\uff1a\u4e3a\u4ec0\u4e48\u5929\u7a7a\u901a\u5e38\u662f\u84dd\u8272\u7684\uff1f\n\u56de\u7b54\uff1a"},
    {"id": "instruction", "prompt": "\u5217\u51fa\u4e09\u6761\u63d0\u9ad8\u7761\u7720\u8d28\u91cf\u7684\u5efa\u8bae\u3002"},
    {"id": "reasoning", "prompt": "\u5c0f\u660e\u67095\u4e2a\u82f9\u679c\uff0c\u9001\u51fa2\u4e2a\u540e\u53c8\u4e70\u4e864\u4e2a\uff0c\u73b0\u5728\u6709\u51e0\u4e2a\uff1f\u8bf7\u7ed9\u51fa\u8fc7\u7a0b\u3002"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight generation quality probes.")
    parser.add_argument("--config", default="configs/experiment_loss_descent_10k.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--prompts",
        default="configs/generation_probe_prompts.json",
        help="JSON, JSONL, or plain text prompt file.",
    )
    parser.add_argument("--output", default=None, help="Output JSON report path.")
    parser.add_argument("--markdown-output", default=None, help="Optional Markdown review report path.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=None)
    return parser.parse_args()


def _record_from_value(value: Any, index: int) -> dict[str, Any]:
    if isinstance(value, str):
        return {"id": f"prompt_{index:04d}", "prompt": value}
    if isinstance(value, dict):
        prompt = value.get("prompt") or value.get("text") or value.get("instruction")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt record {index} must contain prompt/text/instruction.")
        record = dict(value)
        record["id"] = str(record.get("id", f"prompt_{index:04d}"))
        record["prompt"] = prompt
        return record
    raise ValueError(f"Unsupported prompt record at index {index}: {type(value).__name__}")


def load_prompt_records(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return [dict(record) for record in DEFAULT_PROMPTS]
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    suffix = prompt_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("prompts", [])
        if not isinstance(payload, list):
            raise ValueError("JSON prompt file must be a list or {'prompts': [...]} object.")
        return [_record_from_value(value, index) for index, value in enumerate(payload)]

    if suffix == ".jsonl":
        records = []
        with prompt_path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                records.append(_record_from_value(json.loads(line), line_no))
        return records

    records = []
    with prompt_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if text:
                records.append({"id": f"prompt_{line_no:04d}", "prompt": text})
    return records


def resolve_output_path(raw_output: str | Path | None, runtime) -> Path:
    if raw_output:
        return Path(raw_output)
    return runtime.checkpoint_path.parent / "eval_generation.json"


def resolve_markdown_output_path(raw_output: str | Path | None, json_output: Path) -> Path | None:
    if raw_output is None:
        return None
    if str(raw_output).strip() == "":
        return None
    return Path(raw_output)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _known_special_strings(tokenizer) -> list[str]:
    values: list[str] = []
    for attr in ("special_tokens", "user_defined_symbols"):
        tokens = getattr(tokenizer, attr, ())
        values.extend(str(token) for token in tokens if token)
    return sorted(set(values), key=len, reverse=True)


def repetition_ratio(text: str, n: int = 4) -> float:
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) <= n:
        return 0.0
    ngrams = [compact[index : index + n] for index in range(len(compact) - n + 1)]
    if not ngrams:
        return 0.0
    return round((len(ngrams) - len(set(ngrams))) / len(ngrams), 6)


def normalize_answer(text: str) -> str:
    """Normalize harmless presentation differences for deterministic QA scoring."""

    compact = "".join(str(text).split())
    return compact.rstrip("。！？.!?")


def score_expected_output(output: str, expected: Any) -> dict[str, Any]:
    """Score one generated answer against a string or list of accepted answers."""

    if expected is None:
        return {
            "expected": None,
            "normalized_output": normalize_answer(output),
            "exact_match": None,
            "contains_expected": None,
        }
    raw_values = expected if isinstance(expected, list) else [expected]
    accepted = [str(value) for value in raw_values if str(value).strip()]
    if not accepted:
        raise ValueError("expected must contain at least one non-empty answer")
    normalized_output = normalize_answer(output)
    normalized_expected = [normalize_answer(value) for value in accepted]
    return {
        "expected": accepted[0] if len(accepted) == 1 else accepted,
        "normalized_output": normalized_output,
        "exact_match": normalized_output in normalized_expected,
        "contains_expected": any(value in normalized_output for value in normalized_expected),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_prompts": 0,
            "avg_output_chars": 0.0,
            "avg_generated_tokens": 0.0,
            "avg_repetition_ratio": 0.0,
            "avg_latency_ms": 0.0,
            "eos_or_stop_hit_rate": 0.0,
            "special_token_leak_count": 0,
            "empty_output_count": 0,
            "num_scored_prompts": 0,
            "exact_match_count": 0,
            "exact_match_rate": None,
            "contains_expected_count": 0,
            "contains_expected_rate": None,
        }
    scored = [item for item in results if item.get("exact_match") is not None]
    exact_matches = sum(1 for item in scored if item.get("exact_match") is True)
    contains_matches = sum(1 for item in scored if item.get("contains_expected") is True)
    summary = {
        "num_prompts": len(results),
        "avg_output_chars": round(mean(int(item["output_chars"]) for item in results), 4),
        "avg_generated_tokens": round(mean(int(item["generated_tokens"]) for item in results), 4),
        "avg_repetition_ratio": round(mean(float(item["repetition_ratio"]) for item in results), 6),
        "avg_latency_ms": round(mean(float(item["latency_ms"]) for item in results), 4),
        "eos_or_stop_hit_rate": round(
            sum(1 for item in results if item["eos_or_stop_hit"]) / len(results),
            6,
        ),
        "special_token_leak_count": sum(1 for item in results if item["special_token_leaks"]),
        "empty_output_count": sum(1 for item in results if not str(item["output"]).strip()),
        "num_scored_prompts": len(scored),
        "exact_match_count": exact_matches,
        "exact_match_rate": round(exact_matches / len(scored), 6) if scored else None,
        "contains_expected_count": contains_matches,
        "contains_expected_rate": round(contains_matches / len(scored), 6) if scored else None,
    }
    return summary


def summarize_results_by_category(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        category = str(item.get("category") or "uncategorized")
        grouped.setdefault(category, []).append(item)
    return {
        category: summarize_results(items)
        for category, items in sorted(grouped.items())
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Generation Evaluation",
        "",
        f"- Config: `{report['config']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Prompt file: `{report.get('prompt_file')}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report["summary"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Samples", ""])
    for item in report["results"]:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- Category: `{item.get('category', 'uncategorized')}`",
                "",
                "**Prompt**",
                "",
                "```text",
                str(item["prompt"]),
                "```",
                "",
                "**Rendered Prompt**",
                "",
                "```text",
                str(item["rendered_prompt"]),
                "```",
                "",
                "**Output**",
                "",
                "```text",
                str(item["output"]),
                "```",
                "",
                "**Expected**",
                "",
                f"`{item.get('expected')}`",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                f"| output_chars | {item['output_chars']} |",
                f"| generated_tokens | {item['generated_tokens']} |",
                f"| finish_reason | {item['finish_reason']} |",
                f"| eos_or_stop_hit | {item['eos_or_stop_hit']} |",
                f"| latency_ms | {item['latency_ms']} |",
                f"| repetition_ratio | {item['repetition_ratio']} |",
                f"| special_token_leaks | {', '.join(item['special_token_leaks']) or '-'} |",
                f"| exact_match | {item.get('exact_match')} |",
                f"| contains_expected | {item.get('contains_expected')} |",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path or args.config)
    config["__config_path__"] = str(config_path or args.config)
    set_seed(int(config.get("seed", 42)))
    enable_torch_backends()
    runtime = load_generation_runtime(config, checkpoint_override=args.checkpoint)
    prompt_path = (
        resolve_config_path(args.prompts, config_path=config_path)
        if args.prompts
        else None
    )
    prompt_records = load_prompt_records(prompt_path)
    special_strings = _known_special_strings(runtime.tokenizer)

    overrides: dict[str, Any] = {
        "return_full_text": False,
        "skip_special_tokens": False,
        "return_details": True,
    }
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.do_sample is not None:
        overrides["do_sample"] = args.do_sample

    results = []
    for record in prompt_records:
        generation_config = dict(runtime.generation_config)
        if "input" in record:
            generation_config["input"] = record["input"]
        rendered_prompt = render_generation_prompt(record["prompt"], generation_config)
        synchronize_device(runtime.device)
        started = time.perf_counter()
        generated = generate_from_runtime(
            runtime,
            prompt=rendered_prompt,
            overrides={**generation_config, **overrides},
        )
        synchronize_device(runtime.device)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 4)
        if isinstance(generated, dict):
            output = str(generated.get("text", ""))
            finish_reason = str(generated.get("finish_reason", "unknown"))
            generated_tokens = int(generated.get("generated_tokens", 0) or 0)
            eos_or_stop_hit = bool(generated.get("eos_or_stop_hit", False))
        else:
            output = str(generated)
            finish_reason = "unknown"
            generated_tokens = 0
            eos_or_stop_hit = any(stop in output for stop in generation_config.get("stop_strings", []))
        leaks = [token for token in special_strings if token in output]
        expectation_score = score_expected_output(output, record.get("expected"))
        results.append(
            {
                "id": record["id"],
                "category": str(record.get("category", "uncategorized")),
                "prompt": record["prompt"],
                "rendered_prompt": rendered_prompt,
                "output": output,
                "output_chars": len(output),
                "generated_tokens": generated_tokens,
                "finish_reason": finish_reason,
                "eos_or_stop_hit": eos_or_stop_hit,
                "latency_ms": latency_ms,
                "repetition_ratio": repetition_ratio(output),
                "special_token_leaks": leaks,
                **expectation_score,
            }
        )

    output_path = resolve_output_path(args.output, runtime)
    report = {
        "config": str(Path(args.config)),
        "checkpoint": str(runtime.checkpoint_path),
        "prompt_file": str(prompt_path) if prompt_path else None,
        "output": str(output_path),
        "summary": summarize_results(results),
        "summary_by_category": summarize_results_by_category(results),
        "results": results,
    }

    markdown_path = resolve_markdown_output_path(args.markdown_output, output_path)
    if markdown_path is not None:
        report["markdown_output"] = str(markdown_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path is not None:
        write_markdown_report(markdown_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
