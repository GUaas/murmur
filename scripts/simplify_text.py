from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config
from muddywater.generation_runtime import load_generation_runtime
from muddywater.paths import resolve_path
from muddywater.text_simplification.inference import LongTextOptions, TextSimplifier
from muddywater.text_simplification.prompting import format_prompt
from muddywater.utils import enable_torch_backends, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplify Chinese text with Murmur 203M.")
    parser.add_argument(
        "--config",
        default="configs/inference_text_simplification_portable.yaml",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Chinese text to simplify.")
    input_group.add_argument(
        "--text-file",
        type=Path,
        help="UTF-8 text file to simplify; recommended for long documents.",
    )
    parser.add_argument("--output-file", type=Path, help="Optional UTF-8 output path.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--long-text-mode",
        choices=("auto", "always", "never"),
        default=None,
        help="Auto chunks only inputs above the configured prompt-token budget.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="Maximum rendered prompt tokens per independently inferred chunk.",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Print chunk count, latency, and finish reasons to stderr as JSON.",
    )
    return parser.parse_args()


def build_task_prompt(text: str, config: dict) -> str:
    task_config = dict(config.get("text_simplification", {}))
    return format_prompt(
        text,
        source_label=str(task_config.get("source_label", "<|im_start|>")),
        target_label=str(task_config.get("target_label", "<|im_end|>")),
        sanitize=bool(task_config.get("sanitize_reserved_tags", True)),
    )


def load_input_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return args.text_file.read_text(encoding="utf-8-sig")
    return str(args.text)


def build_long_text_options(config: dict, args: argparse.Namespace) -> LongTextOptions:
    long_config = dict(config.get("long_text", {}))
    return LongTextOptions(
        max_prompt_tokens=int(
            args.chunk_tokens
            if args.chunk_tokens is not None
            else long_config.get("max_prompt_tokens_per_chunk", 160)
        ),
        adaptive_output_tokens=bool(long_config.get("adaptive_output_tokens", True)),
        output_token_ratio=float(long_config.get("output_token_ratio", 1.25)),
        min_new_tokens=int(long_config.get("min_new_tokens", 32)),
        fallback_on_empty=bool(long_config.get("fallback_on_empty", True)),
    )


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path or args.config)
    config["__config_path__"] = str(config_path or args.config)
    set_seed(int(config.get("seed", 20260814)))
    enable_torch_backends()

    runtime = load_generation_runtime(config)
    generation_overrides: dict[str, object] = {}
    if args.max_new_tokens is not None:
        generation_overrides["max_new_tokens"] = args.max_new_tokens
    long_config = dict(config.get("long_text", {}))
    long_text_mode = args.long_text_mode or str(long_config.get("mode", "auto"))
    text = load_input_text(args)
    simplifier = TextSimplifier(
        runtime,
        prompt_builder=lambda source: build_task_prompt(source, config),
        options=build_long_text_options(config, args),
        generation_overrides=generation_overrides,
    )
    result = simplifier.simplify(text, mode=long_text_mode)
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(result.text, encoding="utf-8")
    print(result.text)
    if args.show_stats:
        print(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
