from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .utils import resolve_device


@dataclass(frozen=True)
class DistributedContext:
    """Runtime process topology for single-process, DDP, and torchrun jobs."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    backend: str | None = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def init_distributed(device_config: str | None = "auto") -> DistributedContext:
    """Initialize torch.distributed when launched under torchrun.

    The function is intentionally environment-driven. A normal `python
    scripts/pretrain.py` launch remains single-process, while `torchrun
    --nproc_per_node=N scripts/pretrain.py` automatically enables DDP.
    """

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    if world_size <= 1:
        return DistributedContext(device=resolve_device(device_config))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = resolve_device(device_config)
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def barrier(context: DistributedContext | None = None) -> None:
    if context is not None and context.enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_gather_int(value: int, device: torch.device | str = "cpu") -> list[int]:
    """Gather one integer from each rank without padding or averaging."""
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    if not (dist.is_available() and dist.is_initialized()):
        return [int(tensor.item())]
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [int(item.item()) for item in gathered]


def all_gather_objects(value: Any) -> list[Any]:
    """Gather one picklable Python object from every rank."""

    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model
    while True:
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
            continue
        return unwrapped
