from __future__ import annotations

import math


def validate_embedding_configuration(
    *,
    tie_weights: bool,
    input_rmsnorm: bool,
) -> None:
    """Reject the tied-head/input-normalization combination that copies inputs.

    RMS-normalizing a small randomly initialized token embedding raises its
    magnitude to one.  Reusing that same matrix as the output head then gives
    the current token a large self-dot-product before the transformer has
    learned anything.  Untied heads do not have this failure mode.
    """

    if bool(tie_weights) and bool(input_rmsnorm):
        raise ValueError(
            "input_rmsnorm=true is incompatible with tie_weights=true because "
            "the shared output head creates a dominant current-token copy logit. "
            "Set input_rmsnorm=false or tie_weights=false."
        )


def round_up_to_multiple(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    value = int(value)
    return multiple * math.ceil(value / multiple)


def padded_vocab_size(vocab_size: int, pad_to: int = 1) -> int:
    return round_up_to_multiple(int(vocab_size), int(pad_to or 1))


def has_value_embedding(layer_idx: int, n_layers: int) -> bool:
    return int(layer_idx) % 2 == (int(n_layers) - 1) % 2


def backout_capture_layer(n_layers: int) -> int:
    """Return a balanced mid-stack layer with at least one layer after it."""

    n_layers = int(n_layers)
    if n_layers < 2:
        raise ValueError("backout requires at least two transformer layers")
    return (n_layers - 1) // 2


def layer_window_size(
    pattern: str | None,
    layer_idx: int,
    n_layers: int,
    max_seq_len: int,
) -> int | None:
    """Return a per-layer left attention window, or None for full causal attention."""

    pattern = (pattern or "L").upper()
    if not pattern:
        pattern = "L"
    invalid = sorted({char for char in pattern if char not in {"L", "S"}})
    if invalid:
        raise ValueError(
            "attention_window_pattern may only contain 'L' and 'S'; "
            f"got invalid characters: {''.join(invalid)}"
        )

    char = pattern[int(layer_idx) % len(pattern)]
    if int(layer_idx) == int(n_layers) - 1:
        char = "L"
    if char == "L":
        return None

    # Match nanochat's coarse idea: short layers see roughly a quarter context,
    # rounded to an attention-kernel friendly block.
    short_window = max(1, math.ceil(int(max_seq_len) / 4 / 128) * 128)
    return min(int(max_seq_len), short_window)


def init_depth_scaled_values(
    n_layers: int,
    start: float,
    end: float,
) -> list[float]:
    if n_layers <= 1:
        return [float(start)]
    return [
        float(start) + (float(end) - float(start)) * idx / (n_layers - 1)
        for idx in range(n_layers)
    ]
