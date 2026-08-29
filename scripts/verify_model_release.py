from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.model import GPTConfig, GPTLanguageModel
from muddywater.tokenizer import CharacterTokenizer
from muddywater.utils import count_parameters


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(root: Path) -> int:
    checksum_path = root / "SHA256SUMS"
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        target = root / relative.strip().lstrip("*")
        if not target.is_file():
            raise FileNotFoundError(target)
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        checked += 1
    return checked


def verify_checkpoint(root: Path) -> dict[str, object]:
    checkpoint_path = root / "model" / "model_119m.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "murmur_model_weights_v1":
        raise ValueError("Unexpected release checkpoint format")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("model_state is missing or empty")
    if "optimizer_state" in checkpoint or "scaler_state" in checkpoint:
        raise ValueError("Training optimizer/scaler state leaked into weights-only release")
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Invalid model tensor: {name}")
    model_config = checkpoint.get("config", {}).get("model")
    model = GPTLanguageModel(GPTConfig.from_dict(model_config))
    model.load_state_dict(state, strict=True)
    parameters = count_parameters(model, trainable_only=False)
    if parameters != 119_171_840:
        raise ValueError(f"Unexpected parameter count: {parameters}")
    return {
        "format": checkpoint["format"],
        "step": int(checkpoint["step"]),
        "parameters": parameters,
        "state_tensors": len(state),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def verify_tokenizer(root: Path, checkpoint: dict[str, object] | None = None) -> dict[str, object]:
    tokenizer_path = root / "tokenizer" / "sp_unigram_24k.model"
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != 24_000:
        raise ValueError(f"Unexpected tokenizer vocab: {tokenizer.vocab_size}")
    return {
        "vocab_size": tokenizer.vocab_size,
        "sha256": sha256_file(tokenizer_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Murmur 119M model release.")
    parser.add_argument("--root", type=Path, default=PACKAGE_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    report = {
        "status": "passed",
        "root": str(root),
        "checksummed_files": verify_checksums(root),
        "model": verify_checkpoint(root),
        "tokenizer": verify_tokenizer(root),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
