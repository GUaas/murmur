from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrecisionConfig:
    name: str
    amp_enabled: bool
    amp_dtype: torch.dtype
    scaler_enabled: bool


def resolve_precision(training_config: dict[str, Any], device_type: str) -> PrecisionConfig:
    """Resolve mixed precision behavior from config and hardware."""

    legacy_amp = bool(training_config.get("amp", True))
    requested = str(training_config.get("precision", "auto")).lower()
    if not legacy_amp and requested == "auto":
        requested = "fp32"

    if requested == "auto":
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            requested = "bf16"
        elif device_type == "cuda":
            requested = "fp16"
        else:
            requested = "fp32"

    aliases = {
        "float32": "fp32",
        "32": "fp32",
        "float16": "fp16",
        "16": "fp16",
        "bfloat16": "bf16",
    }
    name = aliases.get(requested, requested)
    if name not in {"fp32", "fp16", "bf16"}:
        raise ValueError("training.precision must be one of: auto, fp32, fp16, bf16")
    if (
        device_type == "cuda"
        and name == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "training.precision=bf16 was requested, but this CUDA device/runtime "
            "does not report BF16 support. Fix the PyTorch/CUDA environment or use "
            "an explicitly validated precision."
        )

    amp_enabled = device_type == "cuda" and name in {"fp16", "bf16"}
    amp_dtype = torch.bfloat16 if name == "bf16" else torch.float16
    scaler_enabled = amp_enabled and name == "fp16"
    return PrecisionConfig(
        name=name,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler_enabled=scaler_enabled,
    )


def build_grad_scaler(precision: PrecisionConfig):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=precision.scaler_enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=precision.scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=precision.scaler_enabled)


def _newton_schulz_orthogonalize(
    update: torch.Tensor,
    steps: int,
    eps: float,
) -> torch.Tensor:
    """Approximate the polar factor used by Muon for matrix-shaped updates."""

    if update.dim() < 2:
        return update

    steps = int(steps)
    if steps < 1:
        raise ValueError("muon_ns_steps must be at least 1")

    original_shape = update.shape
    matrix = update.reshape(update.shape[0], -1).float()
    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        matrix = matrix.t()

    matrix = matrix / matrix.norm().clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = matrix @ matrix.t()
        matrix = a * matrix + (b * gram + c * (gram @ gram)) @ matrix

    if transposed:
        matrix = matrix.t()
    return matrix.reshape(original_shape).type_as(update)


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _normalize_muon_orthogonalization(method: str) -> str:
    normalized = str(method or "polar_express").lower()
    if normalized in {"polar", "polar_express"}:
        return "polar_express"
    if normalized in {"newton", "newton_schulz"}:
        return "newton_schulz"
    raise ValueError("muon_orthogonalization must be one of: polar_express, newton_schulz")


def _validate_muon_iterations(method: str, steps: int) -> tuple[str, int]:
    """Validate and canonicalize the configured Muon orthogonalization work."""

    normalized = _normalize_muon_orthogonalization(method)
    steps = int(steps)
    if steps < 1:
        raise ValueError("muon_ns_steps must be at least 1")
    if normalized == "polar_express" and steps > len(POLAR_EXPRESS_COEFFS):
        raise ValueError(
            "muon_ns_steps exceeds the available Polar Express coefficient schedule: "
            f"got {steps}, maximum is {len(POLAR_EXPRESS_COEFFS)}"
        )
    return normalized, steps


def _polar_express_orthogonalize(
    update: torch.Tensor,
    steps: int,
    eps: float,
    row_equilibration: bool = True,
    renormalize: bool = True,
) -> torch.Tensor:
    """Approximate a polar factor using the nanochat-style Polar Express update."""

    if update.dim() < 2:
        return update

    _, steps = _validate_muon_iterations("polar_express", steps)

    original_shape = update.shape
    matrix = update.reshape(update.shape[0], -1).float()
    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        matrix = matrix.t()

    x = matrix
    if row_equilibration:
        target = x.norm(dim=(-2, -1), keepdim=True) / (x.size(-2) ** 0.5)
        row_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        x = x * (target / row_norm)

    x = x / (x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps) * 1.01)
    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
        gram = x @ x.t()
        x = a * x + (b * gram + c * (gram @ gram)) @ x

    if renormalize:
        target_norm = min(x.size(-2), x.size(-1)) ** 0.5
        current_norm = x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
        x = x * (target_norm / current_norm)

    if transposed:
        x = x.t()
    return x.reshape(original_shape).type_as(update)


def _orthogonalize_update(
    update: torch.Tensor,
    steps: int,
    eps: float,
    method: str = "polar_express",
    row_equilibration: bool = True,
    renormalize: bool = True,
) -> torch.Tensor:
    method, steps = _validate_muon_iterations(method, steps)
    if method == "polar_express":
        return _polar_express_orthogonalize(
            update,
            steps=steps,
            eps=eps,
            row_equilibration=row_equilibration,
            renormalize=renormalize,
        )
    if method == "newton_schulz":
        return _newton_schulz_orthogonalize(update, steps=steps, eps=eps)
    raise AssertionError(f"Unhandled Muon orthogonalization method: {method}")


def muon_momentum_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int = 400,
    start: float = 0.85,
    peak: float = 0.97,
    final: float = 0.90,
    warmdown_ratio: float = 0.65,
) -> float:
    if total_steps <= 0:
        return float(peak)
    total_steps = max(1, int(total_steps))
    step = min(total_steps, max(0, int(step)))
    if step >= total_steps:
        return float(final)
    warmup_steps = max(0, int(warmup_steps))
    warmdown_steps = max(1, round(float(warmdown_ratio) * total_steps))
    warmdown_start = max(0, total_steps - warmdown_steps)
    # Warmup and warmdown must never overlap. The previous implementation
    # prioritized warmup in the overlap and then jumped abruptly into a later
    # warmdown value, which can destabilize short and medium-sized runs.
    warmup_end = min(warmup_steps, warmdown_start)
    if warmup_end > 0 and step < warmup_end:
        frac = step / warmup_end
        return float(start) * (1 - frac) + float(peak) * frac
    if step < warmdown_start:
        return float(peak)
    if step >= warmdown_start:
        frac = min(1.0, (step - warmdown_start) / warmdown_steps)
        return float(peak) * (1 - frac) + float(final) * frac
    raise AssertionError("unreachable Muon momentum schedule branch")


def cosine_weight_decay_at_step(base_weight_decay: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return float(base_weight_decay)
    progress = min(1.0, max(0.0, int(step) / max(1, int(total_steps))))
    return float(base_weight_decay) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class _NamedParameterEntry:
    parameter: torch.Tensor
    names: tuple[str, ...]


def _collect_trainable_parameters(named_parameters) -> list[_NamedParameterEntry]:
    """Collect parameter aliases once so tied weights cannot enter two groups."""

    aliases_by_id: dict[int, tuple[torch.Tensor, list[str]]] = {}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        key = id(parameter)
        if key not in aliases_by_id:
            aliases_by_id[key] = (parameter, [str(name)])
        else:
            existing, aliases = aliases_by_id[key]
            if existing is not parameter:
                raise RuntimeError("Parameter identity collision while building optimizer groups")
            aliases.append(str(name))
    return [
        _NamedParameterEntry(parameter=parameter, names=tuple(names))
        for parameter, names in aliases_by_id.values()
    ]


def _parameter_role(name: str, parameter: torch.Tensor) -> str:
    if any(token in name for token in ("transformer.wte", "transformer.wpe")):
        return "embedding"
    if "value_embeds" in name:
        return "value_embedding"
    if "lm_head" in name:
        return "lm_head"
    if parameter.dim() < 2 or any(
        token in name
        for token in (
            "resid_lambdas",
            "x0_lambdas",
            "smear",
            "backout_lambda",
        )
    ):
        return "scalar"
    return "matrix"


def _entry_role(entry: _NamedParameterEntry) -> str:
    roles = {_parameter_role(name, entry.parameter) for name in entry.names}
    if roles == {"embedding", "lm_head"}:
        return "tied_embedding_lm_head"
    if len(roles) != 1:
        aliases = ", ".join(entry.names)
        raise ValueError(
            "A shared parameter has incompatible optimizer roles across aliases: "
            f"{aliases}"
        )
    return next(iter(roles))


def _resolve_tied_embedding_head_lr(
    learning_rate: float,
    embedding_learning_rate: float | None,
    lm_head_learning_rate: float | None,
) -> float:
    """Resolve the single LR available to a tied input/output weight."""

    if embedding_learning_rate is not None and lm_head_learning_rate is not None:
        embedding_lr = float(embedding_learning_rate)
        head_lr = float(lm_head_learning_rate)
        if embedding_lr != head_lr:
            raise ValueError(
                "Tied input/output weights cannot use different learning rates: "
                f"embedding_learning_rate={embedding_lr}, "
                f"lm_head_learning_rate={head_lr}. Untie the weights or configure "
                "one shared value."
            )
        return embedding_lr
    if lm_head_learning_rate is not None:
        return float(lm_head_learning_rate)
    if embedding_learning_rate is not None:
        return float(embedding_learning_rate)
    return float(learning_rate)


def _validate_explicit_role_lr_targets(
    entries_by_role: dict[str, list[_NamedParameterEntry]],
    *,
    embedding_learning_rate: float | None,
    value_embedding_learning_rate: float | None,
    lm_head_learning_rate: float | None,
    matrix_learning_rate: float | None,
    scalar_learning_rate: float | None,
) -> None:
    """Never silently accept a role LR that cannot affect a parameter."""

    available = {role for role, entries in entries_by_role.items() if entries}
    requirements = [
        (embedding_learning_rate, {"embedding", "tied_embedding_lm_head"}, "embedding"),
        (value_embedding_learning_rate, {"value_embedding"}, "value_embedding"),
        (lm_head_learning_rate, {"lm_head", "tied_embedding_lm_head"}, "lm_head"),
        (matrix_learning_rate, {"matrix"}, "matrix"),
        (scalar_learning_rate, {"scalar"}, "scalar"),
    ]
    for configured_value, target_roles, label in requirements:
        if configured_value is not None and available.isdisjoint(target_roles):
            raise ValueError(
                f"{label}_learning_rate was configured, but the model has no "
                f"{label} parameter group"
            )


def _role_learning_rates(
    entries_by_role: dict[str, list[_NamedParameterEntry]],
    *,
    learning_rate: float,
    embedding_learning_rate: float | None,
    value_embedding_learning_rate: float | None,
    lm_head_learning_rate: float | None,
    matrix_learning_rate: float | None,
    scalar_learning_rate: float | None,
    muon_defaults: bool,
) -> dict[str, float]:
    has_tied_embedding_head = bool(entries_by_role.get("tied_embedding_lm_head"))
    tied_lr = (
        _resolve_tied_embedding_head_lr(
            learning_rate=learning_rate,
            embedding_learning_rate=embedding_learning_rate,
            lm_head_learning_rate=lm_head_learning_rate,
        )
        if has_tied_embedding_head
        else float(learning_rate)
    )
    embedding_lr = float(
        embedding_learning_rate if embedding_learning_rate is not None else learning_rate
    )
    return {
        "matrix": float(
            matrix_learning_rate if matrix_learning_rate is not None else learning_rate
        ),
        "embedding": embedding_lr,
        "tied_embedding_lm_head": tied_lr,
        "value_embedding": float(
            value_embedding_learning_rate
            if value_embedding_learning_rate is not None
            else (embedding_lr * 0.5 if muon_defaults else learning_rate)
        ),
        "lm_head": float(
            lm_head_learning_rate if lm_head_learning_rate is not None else learning_rate
        ),
        "scalar": float(
            scalar_learning_rate if scalar_learning_rate is not None else learning_rate
        ),
    }


def _next_adamw_step(state: dict[str, Any]) -> int:
    """Advance AdamW's host-side step, migrating legacy tensor state once."""

    previous = state.get("step", 0)
    if isinstance(previous, torch.Tensor):
        if previous.numel() != 1:
            raise ValueError("AdamW step state must be a scalar")
        previous = int(previous.detach().cpu())
    elif not isinstance(previous, int):
        previous = int(previous)
    step = previous + 1
    state["step"] = step
    return step


class HybridMuonAdamW(torch.optim.Optimizer):
    """Muon for hidden weight matrices with AdamW fallback for embeddings/norms.

    Muon is still a newer optimizer family, so this implementation is opt-in.
    It keeps AdamW behavior for parameters where orthogonalized matrix updates
    are a poor fit, such as embeddings, output heads, biases, and norm weights.
    """

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        eps: float = 1e-8,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_update_scale: float = 1.0,
        muon_nesterov: bool = True,
        muon_orthogonalization: str = "polar_express",
        muon_row_equilibration: bool = True,
        muon_renormalize: bool = True,
    ) -> None:
        muon_orthogonalization, muon_ns_steps = _validate_muon_iterations(
            muon_orthogonalization,
            muon_ns_steps,
        )
        defaults = {
            "lr": float(lr),
            "betas": tuple(float(beta) for beta in betas),
            "weight_decay": float(weight_decay),
            "initial_weight_decay": float(weight_decay),
            "eps": float(eps),
            "use_muon": False,
            "muon_momentum": float(muon_momentum),
            "muon_ns_steps": int(muon_ns_steps),
            "muon_update_scale": float(muon_update_scale),
            "muon_nesterov": bool(muon_nesterov),
            "muon_orthogonalization": str(muon_orthogonalization),
            "muon_row_equilibration": bool(muon_row_equilibration),
            "muon_renormalize": bool(muon_renormalize),
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue
            method, steps = _validate_muon_iterations(
                group.get("muon_orthogonalization", muon_orthogonalization),
                group.get("muon_ns_steps", muon_ns_steps),
            )
            group["muon_orthogonalization"] = method
            group["muon_ns_steps"] = steps

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    @torch.no_grad()
    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        momentum = float(group.get("muon_momentum", 0.95))
        ns_steps = int(group.get("muon_ns_steps", 5))
        eps = float(group.get("eps", 1e-8))
        update_scale = float(group.get("muon_update_scale", 1.0))
        nesterov = bool(group.get("muon_nesterov", True))
        orthogonalization = str(group.get("muon_orthogonalization", "polar_express"))
        row_equilibration = bool(group.get("muon_row_equilibration", True))
        renormalize = bool(group.get("muon_renormalize", True))

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("HybridMuonAdamW does not support sparse gradients")

            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)

            state = self.state[param]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(param)
            buffer = state["momentum_buffer"]
            buffer.mul_(momentum).add_(grad)
            update = grad.add(buffer, alpha=momentum) if nesterov else buffer
            update = _orthogonalize_update(
                update,
                steps=ns_steps,
                eps=eps,
                method=orthogonalization,
                row_equilibration=row_equilibration,
                renormalize=renormalize,
            )

            matrix = update.reshape(update.shape[0], -1)
            fan_out, fan_in = matrix.shape
            shape_scale = math.sqrt(max(1.0, fan_out / max(1, fan_in)))
            param.add_(update, alpha=-lr * update_scale * shape_scale)

    @torch.no_grad()
    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = tuple(float(beta) for beta in group.get("betas", (0.9, 0.95)))
        weight_decay = float(group.get("weight_decay", 0.0))
        eps = float(group.get("eps", 1e-8))

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("HybridMuonAdamW does not support sparse gradients")

            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)

            state = self.state[param]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            step = _next_adamw_step(state)

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)


def build_optimizer(
    named_parameters,
    optimizer_type: str,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
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
    entries = _collect_trainable_parameters(named_parameters)
    optimizer_name = str(optimizer_type or "adamw").strip().lower()

    entries_by_role: dict[str, list[_NamedParameterEntry]] = {}
    for entry in entries:
        entries_by_role.setdefault(_entry_role(entry), []).append(entry)
    _validate_explicit_role_lr_targets(
        entries_by_role,
        embedding_learning_rate=embedding_learning_rate,
        value_embedding_learning_rate=value_embedding_learning_rate,
        lm_head_learning_rate=lm_head_learning_rate,
        matrix_learning_rate=matrix_learning_rate,
        scalar_learning_rate=scalar_learning_rate,
    )

    if optimizer_name == "adamw":
        role_lr_configured = any(
            value is not None
            for value in (
                embedding_learning_rate,
                value_embedding_learning_rate,
                lm_head_learning_rate,
                matrix_learning_rate,
                scalar_learning_rate,
            )
        )
        if role_lr_configured:
            role_lrs = _role_learning_rates(
                entries_by_role,
                learning_rate=learning_rate,
                embedding_learning_rate=embedding_learning_rate,
                value_embedding_learning_rate=value_embedding_learning_rate,
                lm_head_learning_rate=lm_head_learning_rate,
                matrix_learning_rate=matrix_learning_rate,
                scalar_learning_rate=scalar_learning_rate,
                muon_defaults=False,
            )
            optim_groups = []
            for role, role_entries in entries_by_role.items():
                for should_decay in (True, False):
                    parameters = [
                        entry.parameter
                        for entry in role_entries
                        if (entry.parameter.dim() >= 2) == should_decay
                    ]
                    if not parameters:
                        continue
                    group_weight_decay = float(weight_decay) if should_decay else 0.0
                    optim_groups.append(
                        {
                            "params": parameters,
                            "parameter_role": role,
                            "lr": role_lrs[role],
                            "initial_lr": role_lrs[role],
                            "weight_decay": group_weight_decay,
                            "initial_weight_decay": group_weight_decay,
                        }
                    )
        else:
            decay_params = [entry.parameter for entry in entries if entry.parameter.dim() >= 2]
            nodecay_params = [entry.parameter for entry in entries if entry.parameter.dim() < 2]
            optim_groups = [
                {
                    "params": decay_params,
                    "weight_decay": weight_decay,
                    "initial_weight_decay": weight_decay,
                    "initial_lr": learning_rate,
                },
                {
                    "params": nodecay_params,
                    "weight_decay": 0.0,
                    "initial_weight_decay": 0.0,
                    "initial_lr": learning_rate,
                },
            ]
        optim_groups = [group for group in optim_groups if group["params"]]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra_args = {"fused": True} if fused_available and device_type == "cuda" else {}
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )

    if optimizer_name not in {"muon", "muon_adamw", "hybrid_muon"}:
        raise ValueError("training.optimizer must be one of: adamw, muon_adamw")

    has_tied_embedding_head = bool(entries_by_role.get("tied_embedding_lm_head"))
    has_distinct_lm_head = bool(entries_by_role.get("lm_head"))
    if lm_head_learning_rate is not None and not (
        has_tied_embedding_head or has_distinct_lm_head
    ):
        raise ValueError(
            "lm_head_learning_rate was configured, but no lm_head parameter or tied "
            "input/output alias was provided to build_optimizer"
        )

    role_lrs = _role_learning_rates(
        entries_by_role,
        learning_rate=learning_rate,
        embedding_learning_rate=embedding_learning_rate,
        value_embedding_learning_rate=value_embedding_learning_rate,
        lm_head_learning_rate=lm_head_learning_rate,
        matrix_learning_rate=matrix_learning_rate,
        scalar_learning_rate=scalar_learning_rate,
        muon_defaults=True,
    )

    muon_params = [entry.parameter for entry in entries_by_role.get("matrix", [])]
    embedding_params = [entry.parameter for entry in entries_by_role.get("embedding", [])]
    tied_embedding_head_params = [
        entry.parameter for entry in entries_by_role.get("tied_embedding_lm_head", [])
    ]
    value_embedding_params = [
        entry.parameter for entry in entries_by_role.get("value_embedding", [])
    ]
    lm_head_params = [entry.parameter for entry in entries_by_role.get("lm_head", [])]
    scalar_params = [entry.parameter for entry in entries_by_role.get("scalar", [])]
    optim_groups = [
        {
            "params": muon_params,
            "parameter_role": "matrix",
            "lr": role_lrs["matrix"],
            "initial_lr": role_lrs["matrix"],
            "weight_decay": weight_decay,
            "initial_weight_decay": weight_decay,
            "use_muon": True,
            "muon_momentum": muon_momentum,
            "muon_ns_steps": muon_ns_steps,
            "muon_update_scale": muon_update_scale,
            "muon_nesterov": muon_nesterov,
            "muon_orthogonalization": muon_orthogonalization,
            "muon_row_equilibration": muon_row_equilibration,
            "muon_renormalize": muon_renormalize,
        },
        {
            "params": embedding_params,
            "parameter_role": "embedding",
            "lr": role_lrs["embedding"],
            "initial_lr": role_lrs["embedding"],
            "weight_decay": weight_decay,
            "initial_weight_decay": weight_decay,
            "use_muon": False,
        },
        {
            "params": tied_embedding_head_params,
            "parameter_role": "tied_embedding_lm_head",
            "lr": role_lrs["tied_embedding_lm_head"],
            "initial_lr": role_lrs["tied_embedding_lm_head"],
            "weight_decay": weight_decay,
            "initial_weight_decay": weight_decay,
            "use_muon": False,
        },
        {
            "params": value_embedding_params,
            "parameter_role": "value_embedding",
            "lr": role_lrs["value_embedding"],
            "initial_lr": role_lrs["value_embedding"],
            "weight_decay": weight_decay,
            "initial_weight_decay": weight_decay,
            "use_muon": False,
        },
        {
            "params": lm_head_params,
            "parameter_role": "lm_head",
            "lr": role_lrs["lm_head"],
            "initial_lr": role_lrs["lm_head"],
            "weight_decay": weight_decay,
            "initial_weight_decay": weight_decay,
            "use_muon": False,
        },
        {
            "params": scalar_params,
            "parameter_role": "scalar",
            "lr": role_lrs["scalar"],
            "initial_lr": role_lrs["scalar"],
            "weight_decay": 0.0,
            "initial_weight_decay": 0.0,
            "use_muon": False,
        },
    ]
    optim_groups = [group for group in optim_groups if group["params"]]
    return HybridMuonAdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        muon_momentum=muon_momentum,
        muon_ns_steps=muon_ns_steps,
        muon_update_scale=muon_update_scale,
        muon_nesterov=muon_nesterov,
        muon_orthogonalization=muon_orthogonalization,
        muon_row_equilibration=muon_row_equilibration,
        muon_renormalize=muon_renormalize,
    )


@dataclass
class CosineLRScheduler:
    learning_rate: float
    min_lr: float
    total_steps: int
    warmup_steps: int = 0
    decay: bool = True

    def lr_at(self, step: int) -> float:
        if not self.decay:
            return self.learning_rate
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        if self.total_steps <= self.warmup_steps:
            return self.learning_rate
        decay_ratio = min(
            1.0,
            (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps),
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def lr_at_tokens(
        self,
        tokens_seen: int,
        total_tokens: int,
        warmup_tokens: int = 0,
    ) -> float:
        if not self.decay:
            return self.learning_rate
        tokens_seen = max(0, int(tokens_seen))
        total_tokens = max(1, int(total_tokens))
        warmup_tokens = max(0, int(warmup_tokens))
        if warmup_tokens > 0 and tokens_seen < warmup_tokens:
            return self.learning_rate * max(1, tokens_seen) / warmup_tokens
        if total_tokens <= warmup_tokens:
            return self.learning_rate
        decay_ratio = min(
            1.0,
            max(0, tokens_seen - warmup_tokens) / max(1, total_tokens - warmup_tokens),
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


def resolve_warmup_steps(
    training_config: dict[str, Any],
    total_steps: int,
    tokens_per_step: int,
) -> int:
    warmup_tokens = training_config.get("warmup_tokens")
    if warmup_tokens is not None:
        if tokens_per_step <= 0:
            raise ValueError("warmup_tokens requires a positive tokens_per_step estimate")
        return max(1, math.ceil(int(warmup_tokens) / tokens_per_step))

    warmup_ratio = training_config.get("warmup_ratio")
    if warmup_ratio is not None:
        configured = int(total_steps * float(warmup_ratio))
    else:
        configured = int(training_config.get("warmup_steps", 0))

    if configured <= 0 or total_steps <= 0:
        return 0
    if configured >= total_steps:
        return max(1, int(total_steps * 0.1))
    return configured


def resolve_warmup_tokens(
    training_config: dict[str, Any],
    total_tokens: int | None,
    tokens_per_step: int,
) -> int | None:
    warmup_tokens = training_config.get("warmup_tokens")
    if warmup_tokens is not None:
        return max(1, int(warmup_tokens))

    if total_tokens is None:
        return None

    warmup_ratio = training_config.get("warmup_ratio")
    if warmup_ratio is not None:
        configured = int(int(total_tokens) * float(warmup_ratio))
    else:
        warmup_steps = int(training_config.get("warmup_steps", 0) or 0)
        configured = warmup_steps * max(1, int(tokens_per_step))

    if configured <= 0:
        return 0
    if configured >= int(total_tokens):
        return max(1, int(int(total_tokens) * 0.1))
    return configured


def validate_warmup_config(training_config: dict[str, Any]) -> None:
    active: list[str] = []
    warmup_tokens = training_config.get("warmup_tokens")
    warmup_ratio = training_config.get("warmup_ratio")
    warmup_steps = training_config.get("warmup_steps")
    if warmup_tokens is not None:
        active.append("warmup_tokens")
    if warmup_ratio is not None:
        active.append("warmup_ratio")
    if warmup_steps is not None and int(warmup_steps) > 0:
        active.append("warmup_steps")
    if len(active) > 1:
        raise ValueError(
            "Set only one warmup schedule knob: "
            + ", ".join(active)
            + ". Use warmup_tokens, warmup_ratio, or warmup_steps."
        )
