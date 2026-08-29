from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model tensors from a training checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a dictionary: {path}")
    return payload


def model_state_from_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("model_state", payload)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no non-empty model state")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("Model state contains non-tensor values")
    return state


def summarize_state(state: dict[str, torch.Tensor], source: Path, output: Path) -> dict[str, Any]:
    return {
        "source_checkpoint": str(source),
        "output": str(output),
        "tensor_count": len(state),
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
        "dtypes": sorted({str(tensor.dtype) for tensor in state.values()}),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    state = model_state_from_payload(load_checkpoint(source))
    torch.save(state, output)
    metadata = summarize_state(state, source, output)
    metadata["bytes"] = output.stat().st_size
    if args.metadata_output:
        metadata_path = Path(args.metadata_output).resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
