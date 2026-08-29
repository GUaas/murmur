from __future__ import annotations

import statistics
import time
from typing import Any

import torch


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


@torch.inference_mode()
def run_functional_checks(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    full_context: bool = False,
) -> dict[str, Any]:
    model.eval()
    text = "人工智能需要可靠的证据。 A robust model checks its assumptions."
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    if len(ids) < 12:
        ids.extend([tokenizer.eos_id] * (12 - len(ids)))
    ids = ids[: min(48, int(model.config.max_seq_len))]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    first = model(input_ids)["logits"]
    second = model(input_ids)["logits"]
    deterministic_difference = _max_difference(first, second)

    prefix_tokens = max(2, input_ids.size(1) // 2)
    changed = input_ids.clone()
    changed[:, prefix_tokens:] = (changed[:, prefix_tokens:] + 17) % int(tokenizer.vocab_size)
    changed_logits = model(changed)["logits"]
    causal_difference = _max_difference(
        first[:, :prefix_tokens], changed_logits[:, :prefix_tokens]
    )

    prefill = model(input_ids[:, :-1], use_cache=True)
    cached = model(
        input_ids[:, -1:],
        past_key_values=prefill["past_key_values"],
        use_cache=True,
    )["logits"]
    cache_difference = _max_difference(first[:, -1:], cached)

    labels = input_ids[:, 1:]
    loss_outputs = model(input_ids[:, :-1], labels=labels, return_logits=False)
    ce_loss = float(loss_outputs["ce_loss"].item())
    loss_is_finite = bool(torch.isfinite(loss_outputs["ce_loss"]).item())

    over_limit_rejected = False
    try:
        model(
            torch.zeros(
                (1, int(model.config.max_seq_len) + 1),
                dtype=torch.long,
                device=device,
            ),
            return_logits=False,
            labels=torch.zeros(
                (1, int(model.config.max_seq_len) + 1),
                dtype=torch.long,
                device=device,
            ),
        )
    except ValueError:
        over_limit_rejected = True

    full_context_result: dict[str, Any] = {"requested": bool(full_context)}
    if full_context:
        length = int(model.config.max_seq_len)
        context_ids = torch.arange(length, device=device).remainder(int(tokenizer.vocab_size))[None]
        started = time.perf_counter()
        context_outputs = model(context_ids, labels=context_ids, return_logits=False)
        full_context_result.update(
            {
                "passed": bool(torch.isfinite(context_outputs["ce_loss"]).item()),
                "seconds": time.perf_counter() - started,
                "ce_loss": float(context_outputs["ce_loss"].item()),
            }
        )

    roundtrip_samples = [
        "中文与 English 123。",
        "空格  保留\n换行与符号：🙂≤≥",
        "def add(a, b):\n    return a + b",
    ]
    roundtrips = []
    for sample in roundtrip_samples:
        encoded = tokenizer.encode(sample, add_bos=False, add_eos=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        roundtrips.append(
            {
                "text": sample,
                "tokens": len(encoded),
                "unknown_tokens": int(tokenizer.count_unknown_tokens(sample)),
                "exact_roundtrip": decoded == sample,
                "decoded": decoded,
            }
        )

    return {
        "forward_logits_all_finite": bool(torch.isfinite(first).all().item()),
        "forward_shape": list(first.shape),
        "determinism": {
            "max_abs_difference": deterministic_difference,
            "passed": deterministic_difference == 0.0,
        },
        "causal_isolation": {
            "prefix_tokens": prefix_tokens,
            "max_abs_difference": causal_difference,
            "passed": causal_difference <= 1e-6,
        },
        "kv_cache_parity": {
            "max_abs_difference": cache_difference,
            "tolerance": 1e-4,
            "passed": cache_difference <= 1e-4,
        },
        "loss": {"ce_loss": ce_loss, "finite": loss_is_finite},
        "context_contract": {
            "max_seq_len": int(model.config.max_seq_len),
            "over_limit_rejected": over_limit_rejected,
            "full_context": full_context_result,
        },
        "tokenizer_roundtrips": roundtrips,
    }


@torch.inference_mode()
def benchmark_forward(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    sequence_lengths: tuple[int, ...],
    batch_sizes: tuple[int, ...] = (1,),
    repeats: int = 2,
) -> dict[str, Any]:
    model.eval()
    results = []
    for batch_size in batch_sizes:
        for length in sequence_lengths:
            if length > int(model.config.max_seq_len):
                continue
            row = torch.arange(length, device=device).remainder(int(tokenizer.vocab_size))
            input_ids = row[None].repeat(max(1, int(batch_size)), 1)
            timings = []
            for _ in range(max(1, repeats)):
                started = time.perf_counter()
                outputs = model(input_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - started)
                del outputs
            median = statistics.median(timings)
            forwarded_tokens = int(batch_size) * length
            results.append(
                {
                    "batch_size": int(batch_size),
                    "sequence_tokens": length,
                    "forwarded_tokens": forwarded_tokens,
                    "median_seconds": median,
                    "tokens_per_second": forwarded_tokens / max(median, 1e-9),
                    "all_seconds": timings,
                }
            )

    head_dim = int(model.config.n_embd) // int(model.config.n_heads)
    kv_heads = int(model.config.n_kv_heads or model.config.n_heads)
    kv_elements_per_token = 2 * int(model.config.n_layers) * kv_heads * head_dim
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    return {
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "parameter_bytes_current_dtype": parameter_bytes,
        "kv_cache": {
            "elements_per_token_per_batch": kv_elements_per_token,
            "fp32_mib_at_max_context_batch1": (
                kv_elements_per_token * 4 * int(model.config.max_seq_len) / (1024**2)
            ),
            "bf16_mib_at_max_context_batch1": (
                kv_elements_per_token * 2 * int(model.config.max_seq_len) / (1024**2)
            ),
        },
        "forward": results,
    }
