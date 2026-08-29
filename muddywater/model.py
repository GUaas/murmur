from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import (
    CausalSelfAttention,
    DocumentSegments,
    PastKeyValue,
    build_contiguous_document_segments,
    merge_document_attention_mask,
)
from .cache import ModelKVCache, cached_smear_input
from .losses import (
    apply_logit_softcap,
    chunked_language_model_loss,
    language_model_loss_from_logits,
)
from .model_extras import (
    backout_capture_layer,
    has_value_embedding,
    init_depth_scaled_values,
    layer_window_size,
    padded_vocab_size,
    validate_embedding_configuration,
)
from .optim import build_optimizer


# These keys are inputs to ``scaling.apply_auto_scaling`` rather than runtime
# model fields.  The scaler intentionally leaves them in the resolved config
# for experiment provenance, so the model constructor strips this exact set.
# All other unknown keys are rejected to make misspelled architecture settings
# fail before an expensive training run starts.
AUTO_SCALE_AUXILIARY_KEYS = frozenset(
    {
        "auto_scale",
        "depth",
        "aspect_ratio",
        "head_dim",
        "kv_head_ratio",
    }
)


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True
    gradient_checkpointing: bool = False
    norm_type: str = "layernorm"
    mlp_type: str = "gelu"
    mlp_hidden_multiple_of: int = 1
    position_encoding: str = "learned"
    rope_theta: float = 10000.0
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    qk_norm_scale: float = 1.0
    pad_vocab_size_to: int = 1
    logit_softcap: float | None = None
    attention_window_pattern: str = "L"
    input_rmsnorm: bool = False
    residual_scaling: bool = False
    x0_residual: bool = False
    smear: bool = False
    smear_gate_channels: int = 24
    backout: bool = False
    backout_init: float = 0.2
    value_embeddings: bool = False
    value_embedding_gate_channels: int = 12
    document_attention_backend: str = "dense"
    loss_chunk_size: int = 1024

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GPTConfig":
        if not isinstance(data, Mapping):
            raise TypeError(
                "GPTConfig.from_dict expects a mapping; "
                f"got {type(data).__name__}"
            )
        allowed = {field.name for field in fields(cls)}
        raw = dict(data)
        unknown = sorted(set(raw) - allowed - AUTO_SCALE_AUXILIARY_KEYS)
        if unknown:
            names = ", ".join(repr(name) for name in unknown)
            raise ValueError(
                f"Unknown GPTConfig field(s): {names}. "
                "Fix the spelling or remove unsupported model options."
            )
        return cls(**{key: value for key, value in raw.items() if key in allowed})


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        output = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (output * self.weight.float()).type_as(x)


def build_norm(config: GPTConfig) -> nn.Module:
    norm_type = config.norm_type.lower()
    if norm_type == "layernorm":
        return LayerNorm(config.n_embd, bias=config.bias)
    if norm_type == "rmsnorm":
        return RMSNorm(config.n_embd)
    raise ValueError("norm_type must be one of: layernorm, rmsnorm")


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.mlp_type = config.mlp_type.lower()
        if self.mlp_type in {"gelu", "relu2"}:
            hidden_size = 4 * config.n_embd
            self.c_fc = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
            self.c_proj = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        elif self.mlp_type == "swiglu":
            hidden_size = int(8 * config.n_embd / 3)
            multiple_of = max(1, int(config.mlp_hidden_multiple_of))
            if multiple_of > 1:
                hidden_size = multiple_of * ((hidden_size + multiple_of - 1) // multiple_of)
            self.c_fc = nn.Linear(config.n_embd, 2 * hidden_size, bias=config.bias)
            self.c_proj = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        else:
            raise ValueError("mlp_type must be one of: gelu, swiglu, relu2")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        if self.mlp_type == "gelu":
            x = F.gelu(x)
        elif self.mlp_type == "relu2":
            x = F.relu(x).square()
        else:
            gate, value = x.chunk(2, dim=-1)
            x = F.silu(gate) * value
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        layer_idx: int,
        sliding_window: int | None = None,
        uses_value_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.ln_1 = build_norm(config)
        self.attn = CausalSelfAttention(
            n_embd=config.n_embd,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            bias=config.bias,
            use_rope=config.position_encoding.lower() == "rope",
            rope_theta=config.rope_theta,
            qk_norm=config.qk_norm,
            qk_norm_eps=config.qk_norm_eps,
            qk_norm_scale=config.qk_norm_scale,
            sliding_window=sliding_window,
            value_embedding_gate_channels=(
                config.value_embedding_gate_channels if uses_value_embedding else 0
            ),
        )
        self.ln_2 = build_norm(config)
        self.mlp = FeedForward(config)
        self.layer_idx = int(layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_mask_is_causal: bool = False,
        past_kv: PastKeyValue | None = None,
        use_cache: bool = False,
        position_offset: int | None = None,
        value_embedding: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        document_segments: DocumentSegments | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, PastKeyValue]:
        attn_output = self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            attention_mask_is_causal=attention_mask_is_causal,
            past_kv=past_kv,
            use_cache=use_cache,
            position_offset=position_offset,
            value_embedding=value_embedding,
            document_ids=document_ids,
            document_segments=document_segments,
        )
        present_kv = None
        if use_cache:
            attn_output, present_kv = attn_output
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return (x, present_kv) if use_cache else x


def _checkpoint_transformer_block(
    block: TransformerBlock,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    document_segments: DocumentSegments | None,
    value_embedding: torch.Tensor | None,
    attention_mask_is_causal: bool,
) -> torch.Tensor:
    """Run one block with every recomputation dependency passed explicitly."""

    output = block(
        hidden,
        attention_mask=attention_mask,
        attention_mask_is_causal=attention_mask_is_causal,
        value_embedding=value_embedding,
        document_segments=document_segments,
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError("Checkpointed TransformerBlock must return a tensor")
    return output


class GPTLanguageModel(nn.Module):
    """A compact GPT-style decoder-only Transformer implemented with PyTorch."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        validate_embedding_configuration(
            tie_weights=config.tie_weights,
            input_rmsnorm=config.input_rmsnorm,
        )
        if int(config.loss_chunk_size) <= 0:
            raise ValueError("loss_chunk_size must be a positive number of tokens")
        if config.backout and int(config.n_layers) < 2:
            raise ValueError("backout requires at least two transformer layers")
        self.config = config
        self.position_encoding = config.position_encoding.lower()
        if self.position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding must be one of: learned, rope")
        self.padded_vocab_size = padded_vocab_size(
            config.vocab_size,
            config.pad_vocab_size_to,
        )
        if self.padded_vocab_size != config.vocab_size and config.tie_weights is False:
            # Untied output heads are fully supported; this branch exists only to keep
            # the intent explicit when checkpoint shapes are inspected later.
            pass
        self.value_embedding_layer_ids = {
            idx
            for idx in range(config.n_layers)
            if bool(config.value_embeddings) and has_value_embedding(idx, config.n_layers)
        }
        self.window_sizes = [
            layer_window_size(
                config.attention_window_pattern,
                layer_idx=idx,
                n_layers=config.n_layers,
                max_seq_len=config.max_seq_len,
            )
            for idx in range(config.n_layers)
        ]

        transformer_modules: dict[str, nn.Module] = {
            "wte": nn.Embedding(self.padded_vocab_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList(
                [
                    TransformerBlock(
                        config,
                        layer_idx=idx,
                        sliding_window=self.window_sizes[idx],
                        uses_value_embedding=idx in self.value_embedding_layer_ids,
                    )
                    for idx in range(config.n_layers)
                ]
            ),
            "ln_f": build_norm(config),
        }
        if self.position_encoding == "learned":
            transformer_modules["wpe"] = nn.Embedding(config.max_seq_len, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, self.padded_vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.transformer["wte"].weight

        head_dim = config.n_embd // config.n_heads
        kv_dim = (config.n_kv_heads or config.n_heads) * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(idx): nn.Embedding(self.padded_vocab_size, kv_dim)
                for idx in self.value_embedding_layer_ids
            }
        )
        self.resid_lambdas = (
            nn.Parameter(torch.ones(config.n_layers))
            if config.residual_scaling
            else None
        )
        self.x0_lambdas = (
            nn.Parameter(torch.zeros(config.n_layers))
            if config.x0_residual
            else None
        )
        self.smear_gate_channels = max(1, min(int(config.smear_gate_channels), config.n_embd))
        self.smear_gate = (
            nn.Linear(self.smear_gate_channels, 1, bias=False)
            if config.smear
            else None
        )
        self.smear_lambda = nn.Parameter(torch.zeros(1)) if config.smear else None
        self.backout_lambda = (
            nn.Parameter(torch.tensor(float(config.backout_init)))
            if config.backout
            else None
        )

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / (2 * config.n_layers) ** 0.5,
                )
        self._init_algorithmic_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_algorithmic_parameters(self) -> None:
        if self.resid_lambdas is not None:
            values = init_depth_scaled_values(self.config.n_layers, start=1.15, end=1.05)
            self.resid_lambdas.data.copy_(torch.tensor(values, dtype=self.resid_lambdas.dtype))
        if self.x0_lambdas is not None:
            values = init_depth_scaled_values(self.config.n_layers, start=0.20, end=0.05)
            self.x0_lambdas.data.copy_(torch.tensor(values, dtype=self.x0_lambdas.dtype))
        if self.smear_gate is not None:
            torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        if self.smear_lambda is not None:
            torch.nn.init.zeros_(self.smear_lambda)
        if self.backout_lambda is not None:
            self.backout_lambda.data.fill_(float(self.config.backout_init))
        for block in self.transformer["h"]:
            value_gate = getattr(block.attn, "value_gate", None)
            if value_gate is not None:
                torch.nn.init.uniform_(value_gate.weight, 0.0, 0.02)

    def _apply_input_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.input_rmsnorm:
            return x
        return F.rms_norm(x, (x.size(-1),))

    def _resolve_document_attention(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        past_key_values: list[PastKeyValue] | tuple[PastKeyValue, ...] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor | None, bool, DocumentSegments | None]:
        """Resolve strict document isolation before any transformer layers run."""

        if document_ids is None:
            return attention_mask, False, None
        if past_key_values is not None or use_cache:
            raise ValueError(
                "document_ids attention is only supported for full-sequence forwards"
            )
        if document_ids.shape != input_ids.shape:
            raise ValueError(
                f"document_ids shape {tuple(document_ids.shape)} must match input_ids "
                f"shape {tuple(input_ids.shape)}"
            )

        document_backend = str(self.config.document_attention_backend or "dense").lower()
        if document_backend == "dense":
            return (
                merge_document_attention_mask(document_ids, attention_mask),
                True,
                None,
            )
        if document_backend in {"varlen", "strict_varlen", "segmented"}:
            if attention_mask is not None:
                raise ValueError(
                    "document_attention_backend='varlen' cannot combine document_ids "
                    "with attention_mask without changing segmented-attention semantics; "
                    "use document_attention_backend='dense' to merge both masks"
                )
            return None, False, build_contiguous_document_segments(document_ids)
        raise ValueError("document_attention_backend must be one of: dense, varlen")

    def _apply_smear(
        self,
        x: torch.Tensor,
        previous_input: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mix each token with the preceding unsmeared model input.

        ``previous_input`` bridges forward-call boundaries during incremental
        decoding.  The returned cache always contains the final *unsmeared*
        input because that is exactly what the next token consumes.  At packed
        document boundaries the predecessor contribution is forced to zero.
        """

        if self.smear_gate is None or self.smear_lambda is None:
            return x, None

        if document_ids is not None:
            if tuple(document_ids.shape) != tuple(x.shape[:2]):
                raise ValueError(
                    "document_ids passed to smear must have shape [batch, seq_len]; "
                    f"got {tuple(document_ids.shape)}, expected {tuple(x.shape[:2])}"
                )
            if previous_input is not None:
                raise ValueError(
                    "document-aware smear cannot use cached previous_input because "
                    "the previous document id is unavailable"
                )

        if previous_input is not None:
            if previous_input.dim() == 2:
                previous_input = previous_input.unsqueeze(1)
            expected_shape = (x.size(0), 1, x.size(2))
            if tuple(previous_input.shape) != expected_shape:
                raise ValueError(
                    "Cached smear input must have shape [batch, 1, n_embd]; "
                    f"got {tuple(previous_input.shape)}, expected {expected_shape}"
                )
            smear_sources = torch.cat((previous_input, x[:, :-1]), dim=1)
            smear_targets = x
            unchanged_prefix = None
        else:
            smear_sources = x[:, :-1]
            smear_targets = x[:, 1:]
            unchanged_prefix = x[:, :1]

        if smear_targets.size(1) > 0:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(smear_targets[..., : self.smear_gate_channels])
            )
            if document_ids is not None:
                same_document = document_ids[:, 1:].eq(document_ids[:, :-1])
                gate = gate * same_document.unsqueeze(-1).to(dtype=gate.dtype)
            smeared_targets = smear_targets + gate * smear_sources
            x = (
                smeared_targets
                if unchanged_prefix is None
                else torch.cat((unchanged_prefix, smeared_targets), dim=1)
            )

        return x, smear_targets[:, -1:, :] if smear_targets.size(1) > 0 else x[:, -1:, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        ignore_index: int = -100,
        past_key_values: list[PastKeyValue] | tuple[PastKeyValue, ...] | None = None,
        use_cache: bool = False,
        position_offset: int | None = None,
        z_loss_weight: float = 0.0,
        return_logits: bool = True,
    ) -> dict[str, torch.Tensor | list[PastKeyValue] | None]:
        if input_ids.dim() != 2:
            raise ValueError(
                "input_ids must have shape [batch, seq_len]; "
                f"got {tuple(input_ids.shape)}"
            )
        batch_size, seq_len = input_ids.size()
        if seq_len <= 0:
            raise ValueError("input_ids must contain at least one token")
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; "
                f"max_seq_len is {self.config.max_seq_len}."
            )

        past_len = 0
        if past_key_values:
            past_len = past_key_values[0][0].size(-2)

        (
            attention_mask,
            attention_mask_is_causal,
            document_attention_segments,
        ) = self._resolve_document_attention(
            input_ids=input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        token_embeddings = self.transformer["wte"](input_ids)
        if self.position_encoding == "learned":
            total_len = past_len + seq_len
            if total_len > self.config.max_seq_len:
                raise ValueError(
                    f"Cannot forward sequence with cached length {total_len}; "
                    f"max_seq_len is {self.config.max_seq_len}."
                )
            positions = torch.arange(past_len, total_len, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.transformer["wpe"](positions)
            x = token_embeddings + position_embeddings
        else:
            x = token_embeddings
        x = self._apply_input_rmsnorm(x)
        previous_smear_input = cached_smear_input(past_key_values)
        if self.smear_gate is not None and past_len > 0 and previous_smear_input is None:
            raise ValueError(
                "smear-enabled incremental decoding requires the enriched "
                "past_key_values returned directly by this model"
            )
        x, next_smear_input = self._apply_smear(
            x,
            previous_input=previous_smear_input,
            document_ids=document_ids,
        )
        x = self.transformer["drop"](x)

        present_key_values: ModelKVCache | None = (
            ModelKVCache(smear_input=next_smear_input) if use_cache else None
        )
        x0 = x if self.x0_lambdas is not None else None
        backout_at = (
            backout_capture_layer(self.config.n_layers)
            if self.backout_lambda is not None
            else -1
        )
        x_backout = None
        for block_idx, block in enumerate(self.transformer["h"]):
            if self.resid_lambdas is not None:
                x = self.resid_lambdas[block_idx].to(x.dtype) * x
            if self.x0_lambdas is not None and x0 is not None:
                x = x + self.x0_lambdas[block_idx].to(x.dtype) * x0
            past_kv = past_key_values[block_idx] if past_key_values is not None else None
            value_embedding = (
                self.value_embeds[str(block_idx)](input_ids)
                if str(block_idx) in self.value_embeds
                else None
            )
            if self.config.gradient_checkpointing and self.training and not use_cache:
                x = checkpoint(
                    _checkpoint_transformer_block,
                    block,
                    x,
                    attention_mask,
                    document_attention_segments,
                    value_embedding,
                    attention_mask_is_causal,
                    use_reentrant=False,
                )
            else:
                block_output = block(
                    x,
                    attention_mask=attention_mask,
                    attention_mask_is_causal=attention_mask_is_causal,
                    past_kv=past_kv,
                    use_cache=use_cache,
                    position_offset=position_offset,
                    value_embedding=value_embedding,
                    document_segments=document_attention_segments,
                )
                if use_cache:
                    x, present_kv = block_output
                    present_key_values.append(present_kv)
                else:
                    x = block_output
            if self.backout_lambda is not None and block_idx == backout_at:
                x_backout = x
        if self.backout_lambda is not None and x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = self.transformer["ln_f"](x)
        if labels is None and not return_logits:
            raise ValueError("return_logits=false requires labels so the forward has an output")

        logits = None
        loss_result = None
        if return_logits:
            logits = self.lm_head(x)
            if logits.size(-1) != self.config.vocab_size:
                logits = logits[..., : self.config.vocab_size]
            logits = apply_logit_softcap(logits, self.config.logit_softcap)
            if labels is not None:
                loss_result = language_model_loss_from_logits(
                    logits,
                    labels,
                    ignore_index=ignore_index,
                    z_loss_weight=z_loss_weight,
                )
        elif labels is not None:
            loss_result = chunked_language_model_loss(
                x,
                labels,
                self.lm_head,
                vocab_size=self.config.vocab_size,
                ignore_index=ignore_index,
                logit_softcap=self.config.logit_softcap,
                z_loss_weight=z_loss_weight,
                chunk_size=self.config.loss_chunk_size,
            )

        loss = None if loss_result is None else loss_result.loss
        ce_loss = None if loss_result is None else loss_result.ce_loss
        loss_sum = None if loss_result is None else loss_result.loss_sum
        ce_loss_sum = None if loss_result is None else loss_result.ce_loss_sum
        loss_token_count = None if loss_result is None else loss_result.token_count
        z_loss_sum = None if loss_result is None else loss_result.z_loss_sum
        return {
            "logits": logits,
            "loss": loss,
            "ce_loss": ce_loss,
            "loss_sum": loss_sum,
            "total_loss_sum": loss_sum,
            "ce_loss_sum": ce_loss_sum,
            "loss_token_count": loss_token_count,
            "z_loss_sum": z_loss_sum,
            "past_key_values": present_key_values,
        }

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
        optimizer_type: str = "adamw",
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_update_scale: float = 1.0,
        muon_nesterov: bool = True,
        muon_orthogonalization: str = "polar_express",
        muon_row_equilibration: bool = True,
        muon_renormalize: bool = True,
        embedding_learning_rate: float | None = None,
        value_embedding_learning_rate: float | None = None,
        lm_head_learning_rate: float | None = None,
        matrix_learning_rate: float | None = None,
        scalar_learning_rate: float | None = None,
    ) -> torch.optim.Optimizer:
        return build_optimizer(
            self.named_parameters(remove_duplicate=False),
            optimizer_type=optimizer_type,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            muon_momentum=muon_momentum,
            muon_ns_steps=muon_ns_steps,
            muon_update_scale=muon_update_scale,
            muon_nesterov=muon_nesterov,
            muon_orthogonalization=muon_orthogonalization,
            muon_row_equilibration=muon_row_equilibration,
            muon_renormalize=muon_renormalize,
            embedding_learning_rate=embedding_learning_rate,
            value_embedding_learning_rate=value_embedding_learning_rate,
            lm_head_learning_rate=lm_head_learning_rate,
            matrix_learning_rate=matrix_learning_rate,
            scalar_learning_rate=scalar_learning_rate,
        )
