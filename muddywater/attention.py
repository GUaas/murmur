from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PastKeyValue = tuple[torch.Tensor, torch.Tensor]
DocumentSegments = tuple[tuple[tuple[int, int], ...], ...]


def build_contiguous_document_segments(document_ids: torch.Tensor) -> DocumentSegments:
    """Resolve contiguous document spans once for reuse by every model layer.

    The segmented attention fallback needs Python slice boundaries.  Resolving
    them inside every layer causes a GPU-to-CPU synchronization per batch row
    and repeats again during gradient-checkpoint recomputation.  This helper
    performs one compact transfer per model forward and also rejects a document
    id that reappears after another document, which would otherwise make dense
    and segmented document-attention semantics disagree.
    """

    if document_ids.dim() != 2:
        raise ValueError("document_ids must have shape [batch, seq_len]")
    batch_size, seq_len = document_ids.shape
    if seq_len <= 0:
        raise ValueError("document_ids must contain at least one token")

    rows = document_ids.detach().to(device="cpu").tolist()
    all_segments: list[tuple[tuple[int, int], ...]] = []
    for batch_idx, row in enumerate(rows):
        starts = [0]
        seen = {int(row[0])}
        previous = int(row[0])
        for position in range(1, seq_len):
            current = int(row[position])
            if current == previous:
                continue
            if current in seen:
                raise ValueError(
                    "document_ids must form contiguous segments for segmented "
                    f"attention; id {current} reappears in batch row {batch_idx}"
                )
            seen.add(current)
            starts.append(position)
            previous = current
        ends = [*starts[1:], seq_len]
        all_segments.append(tuple(zip(starts, ends)))
    if len(all_segments) != batch_size:
        raise RuntimeError("Failed to resolve document segments for every batch row")
    return tuple(all_segments)


def _sdpa_supports_enable_gqa() -> bool:
    if not hasattr(F, "scaled_dot_product_attention"):
        return False
    doc = getattr(F.scaled_dot_product_attention, "__doc__", "") or ""
    return "enable_gqa" in doc


SDPA_SUPPORTS_ENABLE_GQA = _sdpa_supports_enable_gqa()


class HeadRMSNorm(nn.Module):
    """RMSNorm over the per-head channel dimension."""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        output = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (output * self.weight.float()).type_as(x)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings for query/key tensors."""

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = 1.0 / (
            float(theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            dtype=self.inv_freq.dtype,
            device=q.device,
        )
        freqs = torch.outer(positions, self.inv_freq.to(device=q.device))
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]
        rotated = torch.empty_like(x_float)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos
        return rotated.type_as(x)


def repeat_kv(x: torch.Tensor, repeat_factor: int) -> torch.Tensor:
    if repeat_factor == 1:
        return x
    batch_size, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch_size, n_kv_heads, repeat_factor, seq_len, head_dim)
    return x.reshape(batch_size, n_kv_heads * repeat_factor, seq_len, head_dim)


def build_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    key_len: int,
    past_len: int,
    device: torch.device,
    sliding_window: int | None = None,
) -> torch.Tensor:
    query_positions = torch.arange(past_len, past_len + seq_len, device=device)[:, None]
    key_positions = torch.arange(key_len, device=device)[None, :]
    allowed = key_positions <= query_positions
    if sliding_window is not None and int(sliding_window) > 0:
        allowed = allowed & (key_positions >= query_positions - int(sliding_window))
    return allowed.view(1, 1, seq_len, key_len).expand(
        batch_size,
        1,
        seq_len,
        key_len,
    )


def build_causal_document_attention_mask(document_ids: torch.Tensor) -> torch.Tensor:
    """Build a [B, 1, T, T] mask where tokens see only prior tokens in their document."""
    if document_ids.dim() != 2:
        raise ValueError("document_ids must have shape [batch, seq_len]")
    batch_size, seq_len = document_ids.shape
    same_document = document_ids[:, :, None].eq(document_ids[:, None, :])
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=document_ids.device).tril()
    return (same_document & causal).view(batch_size, 1, seq_len, seq_len)


def merge_document_attention_mask(
    document_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Combine strict document isolation with an optional user mask.

    ``document_ids`` always contributes both the document boundary and causal
    constraints.  A 2D user mask follows the usual key-padding semantics;
    3D/4D masks are interpreted as explicit query/key visibility masks.  The
    result is boolean because every attention backend in this project treats
    ``True`` as visible.
    """

    document_mask = build_causal_document_attention_mask(document_ids)
    if attention_mask is None:
        return document_mask

    batch_size, seq_len = document_ids.shape
    mask = attention_mask.to(dtype=torch.bool, device=document_ids.device)
    if mask.dim() == 2:
        if tuple(mask.shape) != (batch_size, seq_len):
            raise ValueError(
                "2D attention_mask combined with document_ids must have shape "
                f"[batch, seq_len]; got {tuple(mask.shape)}, expected "
                f"{(batch_size, seq_len)}"
            )
        return document_mask & mask[:, None, None, :]

    if mask.dim() == 3:
        if tuple(mask.shape) != (batch_size, seq_len, seq_len):
            raise ValueError(
                "3D attention_mask combined with document_ids must have shape "
                f"[batch, seq_len, seq_len]; got {tuple(mask.shape)}, expected "
                f"{(batch_size, seq_len, seq_len)}"
            )
        return document_mask & mask[:, None, :, :]

    if mask.dim() == 4:
        if mask.size(0) not in {1, batch_size}:
            raise ValueError(
                "4D attention_mask combined with document_ids must have batch "
                f"dimension 1 or {batch_size}; got {mask.size(0)}"
            )
        if tuple(mask.shape[-2:]) != (seq_len, seq_len):
            raise ValueError(
                "4D attention_mask combined with document_ids must end with "
                f"[seq_len, seq_len]; got {tuple(mask.shape)}, expected "
                f"{(seq_len, seq_len)}"
            )
        return document_mask & mask

    raise ValueError(
        "attention_mask combined with document_ids must be 2D, 3D, or 4D"
    )


def masked_softmax(
    scores: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    """Compute masked attention probabilities in FP32 without all-mask leaks."""

    scores_float = scores.float().masked_fill(~allowed, float("-inf"))
    probabilities = F.softmax(scores_float, dim=-1)
    # An entirely masked query has no valid distribution.  Returning zeros
    # matches SDPA and, unlike a finite sentinel, cannot average hidden values.
    probabilities = torch.nan_to_num(probabilities, nan=0.0)
    return probabilities.to(dtype=scores.dtype)


def combine_attention_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    key_len: int,
    past_len: int,
    device: torch.device,
    causal_already_applied: bool = False,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build a boolean mask where True means the key is visible to the query."""
    if causal_already_applied:
        if attention_mask is None:
            raise ValueError("causal_already_applied requires an attention_mask")
        mask = attention_mask.to(dtype=torch.bool, device=device)
        if mask.dim() == 3:
            if mask.size(0) != batch_size:
                raise ValueError(
                    f"attention_mask batch {mask.size(0)} does not match batch size {batch_size}"
                )
            if mask.size(-2) != seq_len or mask.size(-1) != key_len:
                raise ValueError(
                    "3D causal attention_mask must have shape [batch, query_len, key_len]; "
                    f"got {tuple(mask.shape)}, expected query_len={seq_len}, key_len={key_len}"
                )
            if sliding_window is not None and int(sliding_window) > 0:
                allowed = build_causal_attention_mask(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    key_len=key_len,
                    past_len=past_len,
                    device=device,
                    sliding_window=sliding_window,
                )
                return mask[:, None, :, :] & allowed
            return mask[:, None, :, :]
        if mask.dim() == 4:
            if mask.size(0) not in {1, batch_size}:
                raise ValueError(
                    f"4D attention_mask batch {mask.size(0)} must be 1 or {batch_size}"
                )
            if mask.size(-2) != seq_len or mask.size(-1) != key_len:
                raise ValueError(
                    "4D causal attention_mask must end with [query_len, key_len]; "
                    f"got {tuple(mask.shape)}, expected query_len={seq_len}, key_len={key_len}"
                )
            if sliding_window is not None and int(sliding_window) > 0:
                allowed = build_causal_attention_mask(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    key_len=key_len,
                    past_len=past_len,
                    device=device,
                    sliding_window=sliding_window,
                )
                return mask & allowed
            return mask
        raise ValueError("causal attention_mask must be 3D or 4D")

    allowed = build_causal_attention_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        key_len=key_len,
        past_len=past_len,
        device=device,
        sliding_window=sliding_window,
    )
    if attention_mask is None:
        return allowed

    mask = attention_mask.to(dtype=torch.bool, device=device)
    if mask.dim() == 2:
        if mask.size(0) != batch_size:
            raise ValueError(
                f"attention_mask batch {mask.size(0)} does not match batch size {batch_size}"
            )
        if mask.size(-1) == seq_len and past_len:
            prefix = torch.ones(batch_size, past_len, dtype=torch.bool, device=device)
            mask = torch.cat((prefix, mask), dim=-1)
        if mask.size(-1) != key_len:
            raise ValueError(
                f"attention_mask length {mask.size(-1)} does not match key length {key_len}"
            )
        return allowed & mask[:, None, None, :]

    if mask.dim() == 3:
        if mask.size(0) != batch_size:
            raise ValueError(
                f"attention_mask batch {mask.size(0)} does not match batch size {batch_size}"
            )
        if mask.size(-2) != seq_len or mask.size(-1) != key_len:
            raise ValueError(
                "3D attention_mask must have shape [batch, query_len, key_len]; "
                f"got {tuple(mask.shape)}, expected query_len={seq_len}, key_len={key_len}"
            )
        return allowed & mask[:, None, :, :]

    if mask.dim() == 4:
        if mask.size(0) not in {1, batch_size}:
            raise ValueError(
                f"4D attention_mask batch {mask.size(0)} must be 1 or {batch_size}"
            )
        if mask.size(-2) != seq_len or mask.size(-1) != key_len:
            raise ValueError(
                "4D attention_mask must end with [query_len, key_len]; "
                f"got {tuple(mask.shape)}, expected query_len={seq_len}, key_len={key_len}"
            )
        return allowed & mask

    raise ValueError("attention_mask must be 2D, 3D, or 4D")


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention used by GPT-style decoders."""

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        max_seq_len: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.1,
        bias: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        qk_norm_scale: float = 1.0,
        sliding_window: int | None = None,
        value_embedding_gate_channels: int = 0,
    ) -> None:
        super().__init__()
        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.head_dim = n_embd // n_heads
        self.max_seq_len = max_seq_len
        self.kv_repeat = self.n_heads // self.n_kv_heads
        self.qk_norm_scale = float(qk_norm_scale)
        self.sliding_window = None if sliding_window is None else max(1, int(sliding_window))

        q_size = n_embd
        kv_size = self.n_kv_heads * self.head_dim
        self.c_attn = nn.Linear(n_embd, q_size + 2 * kv_size, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta) if use_rope else None
        self.q_norm = HeadRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = HeadRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.value_embedding_gate_channels = max(0, int(value_embedding_gate_channels))
        self.value_gate = (
            nn.Linear(self.value_embedding_gate_channels, self.n_kv_heads, bias=False)
            if self.value_embedding_gate_channels > 0
            else None
        )

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
        batch_size, seq_len, channels = x.size()
        q_size = self.n_embd
        kv_size = self.n_kv_heads * self.head_dim
        q, k, v = self.c_attn(x).split((q_size, kv_size, kv_size), dim=2)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
            if self.qk_norm_scale != 1.0:
                q = q * self.qk_norm_scale
                k = k * self.qk_norm_scale

        if value_embedding is not None:
            if self.value_gate is None:
                raise ValueError("value_embedding was provided but value embeddings are disabled")
            if value_embedding.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    "value_embedding must have shape [batch, seq_len, n_kv_heads * head_dim]"
                )
            value_embedding = value_embedding.view(
                batch_size,
                seq_len,
                self.n_kv_heads,
                self.head_dim,
            ).transpose(1, 2)
            gate_channels = min(self.value_embedding_gate_channels, x.size(-1))
            gate_input = x[..., :gate_channels]
            if gate_channels < self.value_embedding_gate_channels:
                gate_input = F.pad(
                    gate_input,
                    (0, self.value_embedding_gate_channels - gate_channels),
                )
            gate = 3.0 * torch.sigmoid(self.value_gate(gate_input))
            gate = gate.transpose(1, 2).unsqueeze(-1).type_as(value_embedding)
            v = v + gate * value_embedding

        past_len = 0 if past_kv is None else past_kv[0].size(-2)
        if self.rotary is not None:
            q, k = self.rotary(
                q,
                k,
                position_offset=past_len if position_offset is None else int(position_offset),
            )

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present_kv = (k, v)
        if document_ids is not None or document_segments is not None:
            if document_ids is not None and document_segments is not None:
                raise ValueError("Pass document_ids or document_segments, not both")
            if attention_mask is not None:
                raise ValueError("document attention cannot be combined with attention_mask")
            if past_kv is not None or use_cache:
                raise ValueError("document_ids attention is only supported for full-sequence forwards")
            if document_segments is None:
                assert document_ids is not None
                document_segments = build_contiguous_document_segments(document_ids)
            y = self._document_segment_attention(
                q=q,
                k=k,
                v=v,
                document_segments=document_segments,
                batch_size=batch_size,
                seq_len=seq_len,
                channels=channels,
            )
            y = self.resid_dropout(self.c_proj(y))
            return (y, present_kv) if use_cache else y

        if hasattr(F, "scaled_dot_product_attention"):
            use_native_gqa = self.kv_repeat > 1 and SDPA_SUPPORTS_ENABLE_GQA
            k_for_attn = k if use_native_gqa else repeat_kv(k, self.kv_repeat)
            v_for_attn = v if use_native_gqa else repeat_kv(v, self.kv_repeat)
            attn_mask = None
            is_causal = (
                attention_mask is None
                and past_kv is None
                and self.sliding_window is None
            )
            if not is_causal:
                attn_mask = combine_attention_mask(
                    attention_mask=attention_mask,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    key_len=k_for_attn.size(-2),
                    past_len=past_len,
                    device=x.device,
                    causal_already_applied=attention_mask_is_causal,
                    sliding_window=self.sliding_window,
                )
            sdpa_kwargs = {"enable_gqa": True} if use_native_gqa else {}
            y = F.scaled_dot_product_attention(
                q,
                k_for_attn,
                v_for_attn,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **sdpa_kwargs,
            )
            y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
            y = self.resid_dropout(self.c_proj(y))
            return (y, present_kv) if use_cache else y

        k_for_attn = repeat_kv(k, self.kv_repeat)
        v_for_attn = repeat_kv(v, self.kv_repeat)
        att = (q @ k_for_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_len = k_for_attn.size(-2)
        mask = combine_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            key_len=key_len,
            past_len=past_len,
            device=x.device,
            causal_already_applied=attention_mask_is_causal,
            sliding_window=self.sliding_window,
        )
        att = masked_softmax(att, mask)
        att = self.attn_dropout(att)

        y = att @ v_for_attn
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.resid_dropout(self.c_proj(y))
        return (y, present_kv) if use_cache else y

    def _document_segment_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        document_segments: DocumentSegments,
        batch_size: int,
        seq_len: int,
        channels: int,
    ) -> torch.Tensor:
        """Apply causal attention independently to each contiguous document segment."""

        if len(document_segments) != batch_size:
            raise ValueError(
                "document_segments batch size does not match attention input; "
                f"got {len(document_segments)}, expected {batch_size}"
            )
        y = torch.empty(
            batch_size,
            self.n_heads,
            seq_len,
            self.head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        for batch_idx, row_segments in enumerate(document_segments):
            expected_start = 0
            for start, end in row_segments:
                if end <= start:
                    raise ValueError(f"document segment must be non-empty: [{start}, {end})")
                if start != expected_start or end > seq_len:
                    raise ValueError(
                        "document segments must cover the sequence contiguously; "
                        f"expected start {expected_start}, got [{start}, {end})"
                    )
                q_seg = q[batch_idx : batch_idx + 1, :, start:end, :]
                k_seg = k[batch_idx : batch_idx + 1, :, start:end, :]
                v_seg = v[batch_idx : batch_idx + 1, :, start:end, :]
                y[batch_idx : batch_idx + 1, :, start:end, :] = self._segment_attention(
                    q=q_seg,
                    k=k_seg,
                    v=v_seg,
                )
                expected_start = end
            if expected_start != seq_len:
                raise ValueError(
                    "document segments do not cover the full attention sequence; "
                    f"covered through {expected_start}, seq_len={seq_len}"
                )
        return y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)

    def _segment_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        seg_len = q.size(-2)
        if hasattr(F, "scaled_dot_product_attention"):
            use_native_gqa = self.kv_repeat > 1 and SDPA_SUPPORTS_ENABLE_GQA
            k_for_attn = k if use_native_gqa else repeat_kv(k, self.kv_repeat)
            v_for_attn = v if use_native_gqa else repeat_kv(v, self.kv_repeat)
            attn_mask = None
            is_causal = self.sliding_window is None
            if not is_causal:
                attn_mask = combine_attention_mask(
                    attention_mask=None,
                    batch_size=1,
                    seq_len=seg_len,
                    key_len=seg_len,
                    past_len=0,
                    device=q.device,
                    sliding_window=self.sliding_window,
                )
            sdpa_kwargs = {"enable_gqa": True} if use_native_gqa else {}
            return F.scaled_dot_product_attention(
                q,
                k_for_attn,
                v_for_attn,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **sdpa_kwargs,
            )

        k_for_attn = repeat_kv(k, self.kv_repeat)
        v_for_attn = repeat_kv(v, self.kv_repeat)
        att = (q @ k_for_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = combine_attention_mask(
            attention_mask=None,
            batch_size=1,
            seq_len=seg_len,
            key_len=seg_len,
            past_len=0,
            device=q.device,
            sliding_window=self.sliding_window,
        )
        att = masked_softmax(att, mask)
        att = self.attn_dropout(att)
        return att @ v_for_attn
