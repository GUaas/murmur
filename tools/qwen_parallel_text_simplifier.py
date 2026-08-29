"""Run one-record Qwen simplification calls concurrently while preserving order."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from kimi_text_simplifier import (
    ApiConfig,
    InputRecord,
    UsageTotals,
    append_result,
    iter_input_records,
    read_completed_records,
    simplify_one,
    validate_resume,
)


DEFAULT_MODEL = "qwen3.7-plus-2026-05-26"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
FATAL_STATUS_CODES = {401, 403, 404, 429}
_THREAD_STATE = threading.local()


@dataclass(frozen=True)
class WorkResult:
    record: InputRecord
    target: str | None
    usage: UsageTotals
    status: str
    error: str | None = None
    fatal: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simplify one JSONL record per Qwen request with ordered concurrent output."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def thread_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    client = getattr(_THREAD_STATE, "client", None)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        _THREAD_STATE.client = client
    return client


def error_status_code(exc: Exception) -> int | None:
    current: BaseException | None = exc
    while current is not None:
        status_code = getattr(current, "status_code", None)
        if status_code is not None:
            return int(status_code)
        current = current.__cause__ or current.__context__
    return None


def process_record(
    record: InputRecord,
    api_key: str,
    base_url: str,
    timeout: float,
    config: ApiConfig,
) -> WorkResult:
    try:
        client = thread_client(api_key, base_url, timeout)
        target, usage = simplify_one(client, config, record.source)
        if usage.validation_fallback:
            return WorkResult(
                record=record,
                target=target,
                usage=usage,
                status="validation_fallback",
                error=usage.fallback_reason,
            )
        return WorkResult(
            record=record,
            target=target,
            usage=usage,
            status="keep" if target == record.source else "simplified",
        )
    except Exception as exc:
        status_code = error_status_code(exc)
        return WorkResult(
            record=record,
            target=None if status_code in FATAL_STATUS_CODES else record.source,
            usage=UsageTotals(attempts=1),
            status="api_failure",
            error=f"{type(exc).__name__}: {exc}",
            fatal=status_code in FATAL_STATUS_CODES,
        )


def append_failure(handle: Any, result: WorkResult) -> None:
    payload = {
        "line_number": result.record.line_number,
        "status": result.status,
        "error": result.error,
        "source": result.record.source,
    }
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def run(args: argparse.Namespace) -> int:
    if args.workers != 20:
        raise ValueError("This run is fixed to exactly 20 workers")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} environment variable is not set")

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    failure_path = output_path.with_suffix(output_path.suffix + ".failures.jsonl")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records = list(iter_input_records(input_path, args.start_line, args.limit))
    if not records:
        print("No records selected.")
        return 0

    completed = read_completed_records(output_path)
    validate_resume(completed, records)
    remaining = records[len(completed) :]
    if not remaining:
        print(f"Already complete: {output_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = ApiConfig(
        provider="qwen",
        model=args.model,
        reasoning_effort="high",
        thinking_enabled=False,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        max_retries=1,
        retry_base_seconds=0.0,
        sleep_seconds=0.0,
    )

    print(
        f"Selected={len(records)} resume={len(completed)} remaining={len(remaining)} "
        f"workers={args.workers} retries=0 model={args.model}",
        flush=True,
    )

    output_mode = "a" if completed else "w"
    failure_mode = "a" if failure_path.exists() else "w"
    started_at = time.monotonic()
    processed_now = 0
    simplified = 0
    kept = 0
    validation_fallbacks = 0
    api_failures = 0
    prompt_tokens = 0
    completion_tokens = 0

    with (
        output_path.open(output_mode, encoding="utf-8", newline="\n") as output_handle,
        failure_path.open(failure_mode, encoding="utf-8", newline="\n") as failure_handle,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        results = executor.map(
            lambda record: process_record(
                record, api_key, args.base_url, args.timeout, config
            ),
            remaining,
            buffersize=args.workers * 2,
        )

        for result in results:
            if result.fatal:
                append_failure(failure_handle, result)
                raise RuntimeError(
                    f"Fatal API error at input line {result.record.line_number}: "
                    f"{result.error}"
                )

            if result.target is None:
                raise RuntimeError(
                    f"Missing target at input line {result.record.line_number}"
                )
            append_result(output_handle, result.record.source, result.target)
            processed_now += 1
            prompt_tokens += result.usage.prompt_tokens
            completion_tokens += result.usage.completion_tokens

            if result.status == "simplified":
                simplified += 1
            elif result.status == "keep":
                kept += 1
            elif result.status == "validation_fallback":
                validation_fallbacks += 1
                append_failure(failure_handle, result)
            elif result.status == "api_failure":
                api_failures += 1
                append_failure(failure_handle, result)

            total_done = len(completed) + processed_now
            if processed_now % args.progress_every == 0 or total_done == len(records):
                elapsed = time.monotonic() - started_at
                rate = processed_now / elapsed if elapsed > 0 else 0.0
                eta = (len(records) - total_done) / rate if rate > 0 else 0.0
                print(
                    f"progress={total_done}/{len(records)} rate={rate:.2f}/s "
                    f"eta={format_duration(eta)} simplified={simplified} kept={kept} "
                    f"validation_fallbacks={validation_fallbacks} "
                    f"api_failures={api_failures}",
                    flush=True,
                )

    elapsed = time.monotonic() - started_at
    print(
        f"Done in {format_duration(elapsed)}. New rows={processed_now}; "
        f"prompt_tokens={prompt_tokens}; completion_tokens={completion_tokens}; "
        f"output={output_path}; failures={failure_path}",
        flush=True,
    )
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
