from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from ..generation_runtime import GenerationRuntime, generate_from_runtime
from .chunking import ChunkPlan, InferenceChunk, plan_inference_chunks


PromptBuilder = Callable[[str], str]
LongTextMode = Literal["auto", "always", "never"]


@dataclass(frozen=True)
class LongTextOptions:
    """Quality and safety controls for chunked text simplification."""

    max_prompt_tokens: int = 160
    adaptive_output_tokens: bool = True
    output_token_ratio: float = 1.25
    min_new_tokens: int = 32
    fallback_on_empty: bool = True

    def validate(self) -> None:
        if isinstance(self.max_prompt_tokens, bool) or self.max_prompt_tokens < 8:
            raise ValueError("max_prompt_tokens must be an integer of at least 8")
        if not math.isfinite(self.output_token_ratio) or self.output_token_ratio <= 0:
            raise ValueError("output_token_ratio must be finite and greater than zero")
        if isinstance(self.min_new_tokens, bool) or self.min_new_tokens < 1:
            raise ValueError("min_new_tokens must be a positive integer")


@dataclass(frozen=True)
class ChunkInferenceResult:
    index: int
    source: str
    output: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
    latency_ms: float
    used_fallback: bool


@dataclass(frozen=True)
class SimplificationResult:
    text: str
    used_chunking: bool
    source_prompt_tokens: int
    elapsed_ms: float
    chunks: tuple[ChunkInferenceResult, ...]

    def to_dict(self, *, include_chunk_text: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_chunk_text:
            for chunk in payload["chunks"]:
                chunk.pop("source", None)
                chunk.pop("output", None)
        return payload


class TextSimplifier:
    """Run direct or sentence-aware long-text simplification on one loaded model."""

    def __init__(
        self,
        runtime: GenerationRuntime,
        prompt_builder: PromptBuilder,
        *,
        options: LongTextOptions | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.prompt_builder = prompt_builder
        self.options = options or LongTextOptions()
        self.options.validate()
        self.generation_overrides = dict(generation_overrides or {})

    def count_prompt_tokens(self, source: str) -> int:
        prompt = self.prompt_builder(source)
        return len(
            self.runtime.tokenizer.encode(
                prompt,
                add_bos=self.runtime.add_bos,
                add_eos=False,
            )
        )

    def simplify(self, text: str, mode: LongTextMode = "auto") -> SimplificationResult:
        if mode not in {"auto", "always", "never"}:
            raise ValueError("mode must be one of: auto, always, never")
        source = str(text)
        started = time.perf_counter()
        prompt_tokens = self.count_prompt_tokens(source) if source else 0
        should_chunk = mode == "always" or (
            mode == "auto" and prompt_tokens > self.options.max_prompt_tokens
        )
        if not should_chunk or not source.strip():
            chunk = self._infer_chunk(
                InferenceChunk(
                    source=source,
                    separator_after="",
                    prompt_tokens=prompt_tokens,
                    sentence_count=1,
                ),
                index=0,
                adaptive=False,
            )
            return SimplificationResult(
                text=chunk.output,
                used_chunking=False,
                source_prompt_tokens=prompt_tokens,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                chunks=(chunk,),
            )

        plan = plan_inference_chunks(
            source,
            token_count=self.count_prompt_tokens,
            max_prompt_tokens=self.options.max_prompt_tokens,
        )
        return self._run_plan(plan, prompt_tokens=prompt_tokens, started=started)

    def _run_plan(
        self,
        plan: ChunkPlan,
        *,
        prompt_tokens: int,
        started: float,
    ) -> SimplificationResult:
        results = tuple(
            self._infer_chunk(chunk, index=index, adaptive=True)
            for index, chunk in enumerate(plan.chunks)
        )
        merged = plan.leading_whitespace + "".join(
            result.output + chunk.separator_after
            for result, chunk in zip(results, plan.chunks)
        )
        return SimplificationResult(
            text=merged,
            used_chunking=len(plan.chunks) > 1,
            source_prompt_tokens=prompt_tokens,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            chunks=results,
        )

    def _adaptive_max_new_tokens(self, prompt_tokens: int) -> int:
        configured = int(
            self.generation_overrides.get(
                "max_new_tokens",
                self.runtime.generation_config.get("max_new_tokens", 128),
            )
        )
        if not self.options.adaptive_output_tokens:
            return configured
        estimated = max(
            self.options.min_new_tokens,
            int(math.ceil(prompt_tokens * self.options.output_token_ratio)),
        )
        return min(configured, estimated)

    def _infer_chunk(
        self,
        chunk: InferenceChunk,
        *,
        index: int,
        adaptive: bool,
    ) -> ChunkInferenceResult:
        if not chunk.source.strip():
            return ChunkInferenceResult(
                index=index,
                source=chunk.source,
                output=chunk.source,
                prompt_tokens=chunk.prompt_tokens,
                generated_tokens=0,
                finish_reason="empty_input",
                latency_ms=0.0,
                used_fallback=False,
            )

        overrides = dict(self.generation_overrides)
        overrides["return_details"] = True
        if adaptive:
            overrides["max_new_tokens"] = self._adaptive_max_new_tokens(chunk.prompt_tokens)
        started = time.perf_counter()
        generated = generate_from_runtime(
            self.runtime,
            prompt=self.prompt_builder(chunk.source),
            overrides=overrides,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if isinstance(generated, dict):
            output = str(generated.get("text", "")).strip()
            generated_tokens = int(generated.get("generated_tokens", 0) or 0)
            finish_reason = str(generated.get("finish_reason", "unknown"))
        else:
            output = str(generated).strip()
            generated_tokens = 0
            finish_reason = "unknown"

        used_fallback = False
        if not output and self.options.fallback_on_empty:
            output = chunk.source
            used_fallback = True
            finish_reason = f"{finish_reason}_empty_fallback"
        return ChunkInferenceResult(
            index=index,
            source=chunk.source,
            output=output,
            prompt_tokens=chunk.prompt_tokens,
            generated_tokens=generated_tokens,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            used_fallback=used_fallback,
        )
