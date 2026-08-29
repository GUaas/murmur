from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from .paths import EvaluationPaths


@dataclass(frozen=True)
class GenerationMeasurement:
    text: str
    latency_ms: float
    generated_tokens: int
    prompt_tokens: int
    finish_reason: str
    used_chunking: bool
    chunk_count: int
    fallback_count: int = 0


class ModelAdapter(Protocol):
    model_id: str
    display_name: str

    def simplify(self, source: str) -> GenerationMeasurement: ...

    def metadata(self) -> dict[str, Any]: ...


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_snapshot() -> dict[str, int | None]:
    try:
        from extreme_eval.system_info import process_memory_bytes

        return process_memory_bytes()
    except Exception:
        return {"working_set": None, "peak_working_set": None, "private_usage": None}


class V1Adapter:
    model_id = "v1"
    display_name = "simpl_best_model（第一版）"

    def __init__(self, paths: EvaluationPaths, threads: int) -> None:
        torch.set_num_threads(threads)
        if str(paths.v1_dir) not in sys.path:
            sys.path.insert(0, str(paths.v1_dir))
        from standalone_inference import TextSimplifier

        self.paths = paths
        self.memory_before_load = _memory_snapshot()
        started = time.perf_counter()
        self.simplifier = TextSimplifier(model_path=str(paths.v1_dir), device="cpu")
        self.load_seconds = time.perf_counter() - started
        self.memory_after_load = _memory_snapshot()
        self.weight_path = paths.v1_dir / "model.safetensors"
        self.config = json.loads((paths.v1_dir / "config.json").read_text(encoding="utf-8"))

    def simplify(self, source: str) -> GenerationMeasurement:
        chunks = (
            self.simplifier._smart_split(source, self.simplifier.MAX_CHARS)
            if len(source) > self.simplifier.MAX_CHARS
            else [source]
        )
        prompt_tokens = sum(
            len(
                self.simplifier.tokenizer.encode(
                    chunk,
                    add_special_tokens=True,
                    max_length=128,
                    truncation=True,
                )
            )
            for chunk in chunks
        )
        started = time.perf_counter()
        output = self.simplifier.simplify(source)
        latency_ms = (time.perf_counter() - started) * 1000.0
        generated_tokens = len(
            self.simplifier.tokenizer.encode(output, add_special_tokens=False)
        )
        return GenerationMeasurement(
            text=output,
            latency_ms=latency_ms,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            finish_reason="opaque_native_api",
            used_chunking=len(chunks) > 1,
            chunk_count=len(chunks),
        )

    def metadata(self) -> dict[str, Any]:
        model = self.simplifier.model
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "architecture": "BART-style encoder-decoder",
            "device": str(self.simplifier.device),
            "threads": torch.get_num_threads(),
            "load_seconds": round(self.load_seconds, 6),
            "memory_before_load": self.memory_before_load,
            "memory_after_load": self.memory_after_load,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "weight_path": str(self.weight_path),
            "weight_bytes": self.weight_path.stat().st_size,
            "weight_sha256": sha256_file(self.weight_path),
            "model_config": self.config,
            "native_generation": {
                "num_beams": 4,
                "repetition_penalty": 1.2,
                "max_output_length_per_chunk": 64,
                "input_tokens_per_chunk": 128,
                "chunk_threshold_chars": self.simplifier.MAX_CHARS,
            },
            "measurement_note": (
                "The released wrapper does not expose EOS/length termination details; "
                "finish_reason is therefore marked opaque_native_api."
            ),
        }


class V2Adapter:
    model_id = "v2"
    display_name = "murmur-203m（第二版）"

    def __init__(self, paths: EvaluationPaths, threads: int) -> None:
        for import_path in (paths.v2_dir, paths.v2_source):
            if str(import_path) not in sys.path:
                sys.path.insert(0, str(import_path))

        from muddywater.config import load_config
        from muddywater.generation_runtime import load_generation_runtime
        from muddywater.text_simplification.inference import LongTextOptions, TextSimplifier
        from muddywater.text_simplification.prompting import format_prompt
        from muddywater.utils import enable_torch_backends, set_seed

        torch.set_num_threads(threads)
        set_seed(20260815)
        enable_torch_backends()
        self.paths = paths
        self.config = load_config(paths.v2_config)
        self.config["__config_path__"] = str(paths.v2_config.resolve())
        self.config["device"] = "cpu"
        self.memory_before_load = _memory_snapshot()
        started = time.perf_counter()
        self.runtime = load_generation_runtime(self.config)
        self.load_seconds = time.perf_counter() - started
        self.memory_after_load = _memory_snapshot()

        task_config = dict(self.config.get("text_simplification", {}))
        long_config = dict(self.config.get("long_text", {}))
        prompt_builder = lambda source: format_prompt(
            source,
            source_label=str(task_config.get("source_label", "<|im_start|>")),
            target_label=str(task_config.get("target_label", "<|im_end|>")),
            sanitize=bool(task_config.get("sanitize_reserved_tags", True)),
        )
        options = LongTextOptions(
            max_prompt_tokens=int(long_config.get("max_prompt_tokens_per_chunk", 160)),
            adaptive_output_tokens=bool(long_config.get("adaptive_output_tokens", True)),
            output_token_ratio=float(long_config.get("output_token_ratio", 1.25)),
            min_new_tokens=int(long_config.get("min_new_tokens", 32)),
            fallback_on_empty=bool(long_config.get("fallback_on_empty", True)),
        )
        self.long_text_mode = str(long_config.get("mode", "auto"))
        self.simplifier = TextSimplifier(
            self.runtime,
            prompt_builder=prompt_builder,
            options=options,
        )

    def simplify(self, source: str) -> GenerationMeasurement:
        result = self.simplifier.simplify(source, mode=self.long_text_mode)
        reasons = [chunk.finish_reason for chunk in result.chunks]
        if any(reason == "length" for reason in reasons):
            finish_reason = "length"
        elif reasons and all(reason in {"eos", "stop_string"} for reason in reasons):
            finish_reason = "eos" if all(reason == "eos" for reason in reasons) else "stop_string"
        else:
            finish_reason = "+".join(sorted(set(reasons))) or "unknown"
        return GenerationMeasurement(
            text=result.text,
            latency_ms=result.elapsed_ms,
            generated_tokens=sum(chunk.generated_tokens for chunk in result.chunks),
            prompt_tokens=result.source_prompt_tokens,
            finish_reason=finish_reason,
            used_chunking=result.used_chunking,
            chunk_count=len(result.chunks),
            fallback_count=sum(chunk.used_fallback for chunk in result.chunks),
        )

    def metadata(self) -> dict[str, Any]:
        weight_path = Path(self.runtime.checkpoint_path)
        config_payload = asdict(self.runtime.model.config)
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "architecture": "decoder-only Transformer (RoPE, RMSNorm, SwiGLU, GQA)",
            "device": str(self.runtime.device),
            "threads": torch.get_num_threads(),
            "load_seconds": round(self.load_seconds, 6),
            "memory_before_load": self.memory_before_load,
            "memory_after_load": self.memory_after_load,
            "model_parameters": sum(parameter.numel() for parameter in self.runtime.model.parameters()),
            "weight_path": str(weight_path),
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": sha256_file(weight_path),
            "model_config": config_payload,
            "native_generation": self.runtime.generation_config,
            "long_text": self.config.get("long_text", {}),
        }


def build_adapter(model_id: str, paths: EvaluationPaths, threads: int) -> ModelAdapter:
    if model_id == "v1":
        return V1Adapter(paths, threads)
    if model_id == "v2":
        return V2Adapter(paths, threads)
    raise ValueError(f"Unsupported model id: {model_id}")
