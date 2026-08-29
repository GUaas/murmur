from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMETERS = 231_709_056
PRETRAINED_MODEL_PATH = "model/murmur_203m_best_weights_only.pt"
REQUIRED_PATHS = (
    "outputs/sft_203m_text_simplification_pass_filtered/murmur_203m_text_simplification_best_weights_only.pt",
    "tokenizer/sp_unigram_32k.model",
    "configs/sft_203m_text_simplification_pass_filtered.yaml",
    "configs/inference_text_simplification_portable.yaml",
    "scripts/pretrain.py",
    "scripts/simplify_text.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_payload(path: Path) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def extract_state(payload: Any) -> dict[str, torch.Tensor]:
    state = payload.get("model_state", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a model state")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("model state contains non-tensor values")
    return state


def verify_model(path: Path) -> dict[str, Any]:
    payload = load_payload(path)
    state = extract_state(payload)
    parameters = sum(tensor.numel() for tensor in state.values())
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(f"unexpected parameter count in {path}: {parameters}")
    if not all(torch.isfinite(tensor).all().item() for tensor in state.values()):
        raise ValueError(f"non-finite tensor found in {path}")
    result = {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "tensors": len(state),
        "parameters": parameters,
        "dtypes": sorted({str(tensor.dtype) for tensor in state.values()}),
    }
    del state, payload
    gc.collect()
    return result


def verify_manifest(path: Path) -> int:
    checked = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        expected, relative = line.split(maxsplit=1)
        target = ROOT / relative.strip().lstrip("*")
        if not target.is_file():
            raise FileNotFoundError(target)
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the portable Murmur text-simplification bundle.")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    missing = [relative for relative in REQUIRED_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    models = [
        verify_model(
            ROOT
            / "outputs/sft_203m_text_simplification_pass_filtered/"
            / "murmur_203m_text_simplification_best_weights_only.pt"
        )
    ]
    pretrained_model = ROOT / PRETRAINED_MODEL_PATH
    if pretrained_model.is_file():
        models.insert(0, verify_model(pretrained_model))
    hashes_checked = 0
    manifest = ROOT / "SHA256SUMS"
    if not args.skip_hashes:
        hashes_checked = verify_manifest(manifest)

    print(
        json.dumps(
            {
                "status": "passed",
                "package_mode": (
                    "complete" if pretrained_model.is_file() else "text_simplification_without_pretrained"
                ),
                "root": str(ROOT),
                "required_files": len(REQUIRED_PATHS),
                "hashes_checked": hashes_checked,
                "models": models,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
