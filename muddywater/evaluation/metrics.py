from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from muddywater.metrics import loss_to_bits_per_byte, loss_to_bits_per_token
from muddywater.utils import perplexity_from_loss


@dataclass(frozen=True)
class EvaluationMetrics:
    """Token-weighted language-model evaluation metrics."""

    loss: float
    ce_loss: float
    perplexity: float
    bits_per_token: float
    bits_per_byte: float | None
    token_count: int
    byte_count: int | None
    batch_count: int

    @classmethod
    def empty(cls) -> "EvaluationMetrics":
        return cls(
            loss=float("nan"),
            ce_loss=float("nan"),
            perplexity=float("nan"),
            bits_per_token=float("nan"),
            bits_per_byte=None,
            token_count=0,
            byte_count=None,
            batch_count=0,
        )

    @classmethod
    def from_totals(
        cls,
        loss_sum: float,
        token_count: int,
        batch_count: int,
        byte_count: int | None = None,
    ) -> "EvaluationMetrics":
        if token_count <= 0:
            return cls.empty()
        mean_loss = float(loss_sum) / int(token_count)
        return cls(
            loss=mean_loss,
            ce_loss=mean_loss,
            perplexity=perplexity_from_loss(mean_loss),
            bits_per_token=loss_to_bits_per_token(mean_loss),
            bits_per_byte=loss_to_bits_per_byte(loss_sum, byte_count or 0),
            token_count=int(token_count),
            byte_count=int(byte_count) if byte_count is not None else None,
            batch_count=int(batch_count),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "ce_loss": self.ce_loss,
            "perplexity": self.perplexity,
            "bits_per_token": self.bits_per_token,
            "bits_per_byte": self.bits_per_byte,
            "token_count": self.token_count,
            "byte_count": self.byte_count,
            "batch_count": self.batch_count,
        }
