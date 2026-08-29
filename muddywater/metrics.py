from __future__ import annotations

import math


def loss_to_bits_per_token(loss: float) -> float:
    """Convert mean cross-entropy in nats/token to bits/token."""

    if not math.isfinite(loss):
        return float("nan")
    return float(loss) / math.log(2.0)


def loss_to_bits_per_byte(loss_sum: float, byte_count: int) -> float | None:
    """Convert summed cross-entropy in nats to bits/byte when bytes are known."""

    if byte_count <= 0 or not math.isfinite(loss_sum):
        return None
    return float(loss_sum) / math.log(2.0) / int(byte_count)


def safe_rate(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, using 0.0 for empty or invalid durations."""

    if denominator <= 0 or not math.isfinite(denominator):
        return 0.0
    return float(numerator) / float(denominator)
