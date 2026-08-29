from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .generation import generate_text
from .model import GPTConfig, GPTLanguageModel
from .paths import resolve_config_path
from .templates import DEFAULT_SYSTEM_PROMPT, format_generation_prompt
from .tokenizer import CharacterTokenizer
from .utils import file_sha256, resolve_device


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(checkpoint_path, **load_kwargs)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary: {checkpoint_path}")
    return checkpoint


def resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        return checkpoint_path

    fallback_names = ("final.pt", "latest.pt") if checkpoint_path.name == "best.pt" else ()
    for fallback_name in fallback_names:
        fallback_path = checkpoint_path.with_name(fallback_name)
        if fallback_path.exists():
            print(
                f"warning: {checkpoint_path} does not exist; using {fallback_path} instead.",
                file=sys.stderr,
            )
            return fallback_path
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def validate_tokenizer_compatibility(
    tokenizer_path: str | Path,
    tokenizer: CharacterTokenizer,
    checkpoint_config: dict[str, Any],
    strict: bool = True,
) -> None:
    model_config = checkpoint_config.get("model", {})
    if isinstance(model_config, dict) and model_config.get("vocab_size") is not None:
        expected_vocab_size = int(model_config["vocab_size"])
        if expected_vocab_size != int(tokenizer.vocab_size):
            raise ValueError(
                "Tokenizer vocab size does not match checkpoint model config: "
                f"tokenizer={tokenizer.vocab_size}, checkpoint={expected_vocab_size}."
            )

    tokenizer_config = checkpoint_config.get("tokenizer", {})
    if not isinstance(tokenizer_config, dict):
        tokenizer_config = {}
    expected_sha = tokenizer_config.get("sha256") or tokenizer_config.get("tokenizer_sha256")
    if expected_sha:
        actual_sha = file_sha256(tokenizer_path)
        if str(actual_sha).lower() != str(expected_sha).lower():
            raise ValueError(
                "Tokenizer file does not match the checkpoint tokenizer SHA-256. "
                "Use the tokenizer that was saved with the training run, or set "
                "generation.strict_tokenizer_match=false only if you know the vocab mapping is identical."
            )
    elif strict:
        print(
            "warning: checkpoint has no tokenizer SHA-256 metadata; vocab size is the only compatibility check.",
            file=sys.stderr,
        )


@dataclass
class GenerationRuntime:
    model: GPTLanguageModel
    tokenizer: CharacterTokenizer
    device: torch.device
    generation_config: dict[str, Any]
    checkpoint_config: dict[str, Any]
    checkpoint_path: Path
    tokenizer_path: Path
    add_bos: bool


def load_generation_runtime(
    config: dict[str, Any],
    checkpoint_override: str | Path | None = None,
) -> GenerationRuntime:
    device = resolve_device(config.get("device", "auto"))
    config_path = config.get("__config_path__")
    tokenizer_path = Path(
        resolve_config_path(
            config.get("tokenizer", {}).get("path", "outputs/tokenizer/bpe_merged_24k.json"),
            config_path=config_path,
        )
    )
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    checkpoint_path = checkpoint_override or config.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("Set checkpoint in config or pass --checkpoint.")
    resolved_checkpoint = resolve_checkpoint_path(
        resolve_config_path(checkpoint_path, config_path=config_path)
    )

    checkpoint = load_checkpoint_payload(resolved_checkpoint)
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}
    generation_config = dict(config.get("generation", {}))
    validate_tokenizer_compatibility(
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        checkpoint_config=checkpoint_config,
        strict=bool(generation_config.get("strict_tokenizer_match", True)),
    )
    checkpoint_model_config = checkpoint_config.get("model", {})
    if not isinstance(checkpoint_model_config, dict):
        checkpoint_model_config = {}
    model_config_dict = dict(checkpoint_model_config)
    override_model_config = config.get("model", {}) or {}
    if not isinstance(override_model_config, dict):
        raise ValueError("generate config 'model' must be a mapping.")
    model_config_dict.update(override_model_config)
    model_config_dict["vocab_size"] = tokenizer.vocab_size
    model = GPTLanguageModel(GPTConfig.from_dict(model_config_dict))
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)

    add_bos = generation_config.get("add_bos")
    if add_bos is None:
        add_bos = checkpoint_config.get("data", {}).get("add_bos", True)

    return GenerationRuntime(
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_config=generation_config,
        checkpoint_config=checkpoint_config,
        checkpoint_path=resolved_checkpoint,
        tokenizer_path=tokenizer_path,
        add_bos=bool(add_bos),
    )


def render_generation_prompt(prompt: str, generation_config: dict[str, Any]) -> str:
    if bool(generation_config.get("apply_chat_template", True)):
        return format_generation_prompt(
            instruction=prompt,
            input_text=generation_config.get("input", ""),
            chat_template=generation_config.get("chat_template", "chatml"),
            system_prompt=generation_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        )
    return prompt


def generate_from_runtime(
    runtime: GenerationRuntime,
    prompt: str,
    overrides: dict[str, Any] | None = None,
) -> str | dict[str, object]:
    generation_config = dict(runtime.generation_config)
    if overrides:
        generation_config.update(overrides)
    return generate_text(
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        prompt=prompt,
        max_new_tokens=int(generation_config.get("max_new_tokens", 128)),
        temperature=float(generation_config.get("temperature", 0.8)),
        top_k=generation_config.get("top_k", 50),
        top_p=generation_config.get("top_p", 0.95),
        do_sample=bool(generation_config.get("do_sample", True)),
        repetition_penalty=float(generation_config.get("repetition_penalty", 1.05)),
        stop_strings=generation_config.get("stop_strings", ["<|im_end|>"]),
        device=runtime.device,
        return_full_text=bool(generation_config.get("return_full_text", True)),
        skip_special_tokens=bool(generation_config.get("skip_special_tokens", True)),
        use_cache=bool(generation_config.get("use_cache", True)),
        add_bos=runtime.add_bos,
        return_details=bool(generation_config.get("return_details", False)),
    )
