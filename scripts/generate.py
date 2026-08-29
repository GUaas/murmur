from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.config import load_config
from muddywater.generation_runtime import (
    GenerationRuntime,
    generate_from_runtime,
    load_checkpoint_payload,
    load_generation_runtime,
    render_generation_prompt,
    resolve_checkpoint_path,
    validate_tokenizer_compatibility,
)
from muddywater.utils import enable_torch_backends, set_seed
from muddywater.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chinese text with MuddyWaterAI-Murmur.")
    parser.add_argument("--config", type=str, default="configs/experiment_loss_descent_10k.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path or args.config)
    config["__config_path__"] = str(config_path or args.config)
    set_seed(int(config.get("seed", 42)))
    enable_torch_backends()

    runtime = load_generation_runtime(config, checkpoint_override=args.checkpoint)
    generation_config = dict(runtime.generation_config)
    prompt = args.prompt if args.prompt is not None else generation_config.get("prompt", "")
    if args.max_new_tokens is not None:
        generation_config["max_new_tokens"] = args.max_new_tokens

    prompt = render_generation_prompt(prompt, generation_config)
    text = generate_from_runtime(
        runtime,
        prompt=prompt,
        overrides=generation_config,
    )
    print(text)


if __name__ == "__main__":
    main()
