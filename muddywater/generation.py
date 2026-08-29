from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .cache import trim_past_key_values
from .model import GPTLanguageModel
from .tokenizer import CharacterTokenizer


def _trim_past_key_values(past_key_values, max_cache_len: int):
    return trim_past_key_values(past_key_values, max_cache_len=max_cache_len)


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    filtered = logits.clone()

    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer or None")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
    if top_p is not None:
        top_p = float(top_p)
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be finite and in the interval (0, 1]")

    if top_k is not None and top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        threshold = torch.topk(filtered, top_k)[0][..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        filtered = filtered.masked_fill(indices_to_remove, float("-inf"))

    return filtered


def _known_special_strings(tokenizer: CharacterTokenizer) -> list[str]:
    values: list[str] = []
    for attr in ("special_tokens", "user_defined_symbols"):
        tokens = getattr(tokenizer, attr, ())
        values.extend(str(token) for token in tokens if token)
    return sorted(set(values), key=len, reverse=True)


def _remove_known_special_strings(text: str, tokenizer: CharacterTokenizer) -> str:
    for token in _known_special_strings(tokenizer):
        text = text.replace(token, "")
    return text


def _truncate_at_stop(text: str, stop_strings: list[str]) -> str:
    cut_at: int | None = None
    for stop in stop_strings:
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0:
            cut_at = index if cut_at is None else min(cut_at, index)
    return text if cut_at is None else text[:cut_at]


def _matched_stop_string(text: str, stop_strings: list[str]) -> str | None:
    matches = [
        (text.find(stop), index, stop)
        for index, stop in enumerate(stop_strings)
        if stop and text.find(stop) >= 0
    ]
    return min(matches)[2] if matches else None


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor | list[int],
    tokenizer: CharacterTokenizer,
    repetition_penalty: float,
) -> torch.Tensor:
    penalty = float(repetition_penalty)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("repetition_penalty must be finite and greater than zero")
    if penalty == 1.0:
        return logits
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=logits.device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.size(0) == 1 and logits.size(0) > 1:
        ids = ids.expand(logits.size(0), -1)
    if ids.dim() != 2 or ids.size(0) != logits.size(0):
        raise ValueError("token_ids must have shape [batch, sequence] matching logits")

    valid = (ids >= 0) & (ids < logits.size(-1))
    valid &= ids != int(tokenizer.pad_id)
    valid &= ids != int(tokenizer.bos_id)
    safe_ids = torch.where(valid, ids, torch.full_like(ids, int(tokenizer.pad_id)))
    seen = torch.zeros_like(logits, dtype=torch.bool)
    seen.scatter_(dim=-1, index=safe_ids, value=True)
    seen[:, int(tokenizer.pad_id)] = False
    seen[:, int(tokenizer.bos_id)] = False
    penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen, penalized, logits)


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    do_sample: bool,
) -> torch.Tensor:
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if not logits.is_floating_point():
        raise TypeError("sampling logits must use a floating-point dtype")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("sampling logits contain NaN or infinity")
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # BF16/FP16 cumulative probability calculations can move a nucleus cutoff
    # by many tokens for large vocabularies.  Keep model execution low precision
    # but perform the tiny final-token distribution calculation in FP32.
    stable_logits = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    stable_logits = stable_logits / temperature
    stable_logits = _top_k_top_p_filtering(
        stable_logits,
        top_k=top_k,
        top_p=top_p,
    )
    if not bool(torch.isfinite(stable_logits).any(dim=-1).all()):
        raise ValueError("top-k/top-p filtering removed every sampling candidate")
    if do_sample:
        probs = F.softmax(stable_logits, dim=-1)
        if not bool(torch.isfinite(probs).all()):
            raise ValueError("sampling probabilities contain NaN or infinity")
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(stable_logits, dim=-1, keepdim=True)


@torch.no_grad()
def generate_text(
    model: GPTLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    stop_strings: list[str] | None = None,
    device: torch.device | str = "cpu",
    return_full_text: bool = True,
    skip_special_tokens: bool = True,
    use_cache: bool = True,
    add_bos: bool = True,
    return_details: bool = False,
) -> str | dict[str, object]:
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not math.isfinite(float(repetition_penalty)) or float(repetition_penalty) <= 0.0:
        raise ValueError("repetition_penalty must be finite and greater than zero")

    model.eval()
    device = torch.device(device)

    input_ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
    if not input_ids:
        input_ids = [tokenizer.bos_id]
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    prompt_token_count = ids.size(1)
    stop_strings = stop_strings or []
    past_key_values = None
    cache_position = 0
    can_slide_kv_cache = model.config.position_encoding.lower() == "rope"
    finish_reason = "length"
    generated_token_count = 0

    for _ in range(max_new_tokens):
        if use_cache:
            if past_key_values is None:
                context = ids[:, -model.config.max_seq_len :]
                outputs = model(context, use_cache=True, position_offset=0)
                cache_position = context.size(1)
            else:
                cached_len = past_key_values[0][0].size(-2)
                if can_slide_kv_cache:
                    max_cache_len = max(1, model.config.max_seq_len - 1)
                    if cached_len > max_cache_len:
                        past_key_values = _trim_past_key_values(
                            past_key_values,
                            max_cache_len=max_cache_len,
                        )
                    outputs = model(
                        ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                        position_offset=cache_position,
                    )
                    cache_position += 1
                elif cached_len >= model.config.max_seq_len:
                    context = ids[:, -model.config.max_seq_len :]
                    outputs = model(context, use_cache=True)
                    cache_position = context.size(1)
                else:
                    outputs = model(
                        ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    cache_position += 1
            past_key_values = outputs.get("past_key_values")
        else:
            context = ids[:, -model.config.max_seq_len :]
            outputs = model(context)
        logits = outputs["logits"][:, -1, :]

        logits = _apply_repetition_penalty(
            logits=logits,
            token_ids=ids,
            tokenizer=tokenizer,
            repetition_penalty=repetition_penalty,
        )
        next_id = _sample_next_token(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

        ids = torch.cat([ids, next_id], dim=1)
        generated_token_count += 1
        if int(next_id.item()) == tokenizer.eos_id:
            finish_reason = "eos"
            break
        if stop_strings:
            continuation_ids = ids[0, prompt_token_count:].tolist()
            continuation = tokenizer.decode(continuation_ids, skip_special_tokens=False)
            if any(stop in continuation for stop in stop_strings):
                finish_reason = "stop_string"
                break

    continuation_ids = ids[0, prompt_token_count:].tolist()
    if continuation_ids and int(continuation_ids[-1]) == tokenizer.eos_id:
        continuation_ids = continuation_ids[:-1]
    raw_continuation = tokenizer.decode(continuation_ids, skip_special_tokens=False)
    matched_stop_string = _matched_stop_string(raw_continuation, stop_strings)
    continuation = _truncate_at_stop(raw_continuation, stop_strings)

    if return_full_text:
        text = prompt + continuation
    else:
        text = continuation

    if skip_special_tokens:
        text = _remove_known_special_strings(text, tokenizer)
    if return_details:
        return {
            "text": text,
            "finish_reason": finish_reason,
            "generated_tokens": generated_token_count,
            "eos_or_stop_hit": finish_reason in {"eos", "stop_string"},
            "matched_stop_string": matched_stop_string,
            "raw_continuation": raw_continuation,
        }
    return text
