from __future__ import annotations

from collections.abc import Iterable

import torch

from .attention import PastKeyValue


class ModelKVCache(list[PastKeyValue]):
    """List-compatible layer KV cache with optional model-level state.

    Attention layers still see the same ``(key, value)`` pairs as before.  The
    extra state belongs to transformations that happen before the first layer,
    so storing it once on the cache avoids duplicating it for every layer.
    """

    def __init__(
        self,
        values: Iterable[PastKeyValue] = (),
        *,
        smear_input: torch.Tensor | None = None,
    ) -> None:
        super().__init__(values)
        self.smear_input = smear_input

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(index, slice):
            return ModelKVCache(value, smear_input=self.smear_input)
        return value

    def copy(self) -> "ModelKVCache":
        return ModelKVCache(self, smear_input=self.smear_input)


def cached_smear_input(
    past_key_values: list[PastKeyValue] | tuple[PastKeyValue, ...] | None,
) -> torch.Tensor | None:
    """Return the pre-smear input carried by an enriched KV cache, if any."""

    return getattr(past_key_values, "smear_input", None)


def trim_past_key_values(
    past_key_values: list[PastKeyValue] | tuple[PastKeyValue, ...] | None,
    max_cache_len: int,
) -> list[PastKeyValue] | tuple[PastKeyValue, ...] | None:
    """Trim attention K/V tensors while preserving model-level cache state."""

    if past_key_values is None or max_cache_len <= 0:
        return past_key_values

    trimmed = [
        (key[..., -max_cache_len:, :], value[..., -max_cache_len:, :])
        for key, value in past_key_values
    ]
    if isinstance(past_key_values, ModelKVCache):
        return ModelKVCache(trimmed, smear_input=past_key_values.smear_input)
    if isinstance(past_key_values, tuple):
        return tuple(trimmed)
    return trimmed
