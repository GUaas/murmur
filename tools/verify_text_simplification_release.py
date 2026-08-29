from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


MODEL_RELATIVE_PATH = Path("model/murmur_203m_text_simplification_best_weights_only.pt")
TOKENIZER_RELATIVE_PATH = Path("tokenizer/sp_unigram_32k.model")
CONFIG_RELATIVE_PATH = Path("configs/inference.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the Murmur inference release.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Extracted release directory.",
    )
    parser.add_argument("--skip-model-load", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, separator, relative = raw_line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(f"Invalid manifest line: {raw_line!r}")
        expected[Path(relative)] = digest.lower()
    return expected


def verify_hashes(root: Path) -> int:
    manifest_path = root / "SHA256SUMS"
    expected = read_manifest(manifest_path)
    actual_files = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != set(expected):
        missing = sorted(str(path) for path in set(expected) - actual_files)
        unexpected = sorted(str(path) for path in actual_files - set(expected))
        raise RuntimeError(f"Manifest mismatch: missing={missing}, unexpected={unexpected}")
    for relative, expected_digest in expected.items():
        actual_digest = file_sha256(root / relative)
        if actual_digest != expected_digest:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
    return len(expected)


def verify_planner(root: Path) -> dict[str, int]:
    sys.path.insert(0, str(root / "source"))
    from muddywater.text_simplification.chunking import plan_inference_chunks

    source = "第一段需要简化。第二句继续说明。\n\n" + "无标点长文本" * 80
    plan = plan_inference_chunks(
        source,
        token_count=lambda text: len(text) + 2,
        max_prompt_tokens=80,
    )
    rebuilt = plan.leading_whitespace + "".join(
        chunk.source + chunk.separator_after for chunk in plan.chunks
    )
    if rebuilt != source:
        raise RuntimeError("Long-text planner failed lossless reconstruction")
    if not all(chunk.prompt_tokens <= 80 for chunk in plan.chunks):
        raise RuntimeError("Long-text planner exceeded its token budget")
    return {"chunks": len(plan.chunks), "source_characters": len(source)}


def verify_runtime(root: Path, *, skip_model_load: bool) -> dict[str, object]:
    from muddywater.tokenizer import CharacterTokenizer

    tokenizer_path = root / TOKENIZER_RELATIVE_PATH
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    tokenizer_result = {
        "vocab_size": int(tokenizer.vocab_size),
        "sample_tokens": len(tokenizer.encode("这是一条发布包验证文本。")),
    }
    if skip_model_load:
        return {"tokenizer": tokenizer_result, "model_load": "skipped"}

    from muddywater.config import load_config
    from muddywater.generation_runtime import load_generation_runtime

    config_path = root / CONFIG_RELATIVE_PATH
    config = load_config(config_path)
    config["__config_path__"] = str(config_path)
    runtime = load_generation_runtime(config)
    parameter_count = sum(parameter.numel() for parameter in runtime.model.parameters())
    if parameter_count != 203_037_056:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count}")
    return {
        "tokenizer": tokenizer_result,
        "model_load": "passed",
        "parameters": parameter_count,
        "device": str(runtime.device),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    required = [
        root / "SHA256SUMS",
        root / MODEL_RELATIVE_PATH,
        root / TOKENIZER_RELATIVE_PATH,
        root / CONFIG_RELATIVE_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required release files are missing: {missing}")

    result = {
        "status": "passed",
        "root": str(root),
        "verified_files": verify_hashes(root),
        "planner": verify_planner(root),
        "runtime": verify_runtime(root, skip_model_load=args.skip_model_load),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
