from __future__ import annotations

import ctypes
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from muddywater.generation_runtime import GenerationRuntime, generate_from_runtime


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def current_process_rss_bytes() -> int | None:
    """Return resident/working-set bytes without adding a psutil dependency."""

    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory_info.restype = wintypes.BOOL
        handle = get_current_process()
        ok = get_memory_info(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.WorkingSetSize) if ok else None
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage * (1 if os.uname().sysname == "Darwin" else 1024))
    except (ImportError, AttributeError, OSError):
        return None


@torch.inference_mode()
def benchmark_runtime(
    runtime: GenerationRuntime,
    sequence_lengths: tuple[int, ...] = (32, 128, 512, 1024, 2048),
    repeats: int = 2,
    generation_tokens: int = 32,
    generation_repeats: int = 3,
) -> dict[str, Any]:
    model = runtime.model
    model.eval()
    device = runtime.device
    max_length = int(model.config.max_seq_len)
    sequence_lengths = tuple(length for length in sequence_lengths if length <= max_length)
    warmup = torch.randint(
        0,
        int(runtime.tokenizer.vocab_size),
        (1, min(32, max_length)),
        device=device,
    )
    outputs = model(warmup)
    del outputs, warmup
    _synchronize(device)

    forward = []
    for length in sequence_lengths:
        input_ids = torch.randint(
            0,
            int(runtime.tokenizer.vocab_size),
            (1, int(length)),
            device=device,
        )
        timings = []
        for _ in range(max(1, int(repeats))):
            _synchronize(device)
            start = time.perf_counter()
            outputs = model(input_ids)
            _synchronize(device)
            timings.append(time.perf_counter() - start)
            del outputs
        median_seconds = statistics.median(timings)
        forward.append(
            {
                "batch_size": 1,
                "sequence_tokens": int(length),
                "median_seconds": median_seconds,
                "tokens_per_second": int(length) / max(median_seconds, 1e-9),
                "repeats": len(timings),
                "note": "full-logit prefill through bundled PyTorch runtime",
            }
        )
        del input_ids

    generation_config = {
        "max_new_tokens": int(generation_tokens),
        "do_sample": False,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "return_details": True,
        "return_full_text": False,
        "use_cache": True,
    }
    generation_timings = []
    details = None
    for _ in range(max(1, int(generation_repeats))):
        start = time.perf_counter()
        details = generate_from_runtime(
            runtime,
            "人工智能的发展",
            overrides=generation_config,
        )
        generation_timings.append(time.perf_counter() - start)
        if not isinstance(details, dict):
            raise TypeError("Detailed generation result was expected")
    assert isinstance(details, dict)
    generation_seconds = statistics.median(generation_timings)

    unique_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    head_dim = int(model.config.n_embd) // int(model.config.n_heads)
    kv_heads = int(model.config.n_kv_heads or model.config.n_heads)
    kv_cache_fp32_per_token = (
        2 * int(model.config.n_layers) * kv_heads * head_dim * 4
    )
    checkpoint_path = Path(runtime.checkpoint_path)
    return {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "unique_parameters": int(unique_parameters),
        "parameter_bytes_current_dtype": int(parameter_bytes),
        "parameter_mib_current_dtype": parameter_bytes / (1024**2),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_mib": checkpoint_path.stat().st_size / (1024**2),
        "process_rss_bytes_after_model_and_benchmarks": current_process_rss_bytes(),
        "kv_cache": {
            "fp32_bytes_per_token_per_batch": kv_cache_fp32_per_token,
            "bf16_bytes_per_token_per_batch": kv_cache_fp32_per_token // 2,
            "fp32_mib_at_max_context_batch1": kv_cache_fp32_per_token
            * max_length
            / (1024**2),
            "bf16_mib_at_max_context_batch1": kv_cache_fp32_per_token
            * max_length
            / 2
            / (1024**2),
        },
        "forward_prefill": forward,
        "greedy_cached_decode": {
            "prompt": "人工智能的发展",
            "generated_tokens": int(details["generated_tokens"]),
            "seconds": generation_seconds,
            "all_seconds": generation_timings,
            "min_seconds": min(generation_timings),
            "max_seconds": max(generation_timings),
            "repeats": len(generation_timings),
            "tokens_per_second": int(details["generated_tokens"])
            / max(generation_seconds, 1e-9),
            "finish_reason": details["finish_reason"],
        },
        "limitations": [
            "Single-process measurements on the current machine; not a GPU benchmark.",
            "The bundled runtime materializes full vocabulary logits during prefill.",
            "No quantized, ONNX, TensorRT, GGUF, or serving-engine export was supplied.",
        ],
    }
