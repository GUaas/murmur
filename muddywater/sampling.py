from __future__ import annotations

import random
from collections.abc import Iterator, Sized
from itertools import islice
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler


class ResumeOffsetSampler(Sampler[int]):
    """Skip already-consumed sampler indices without loading their samples.

    The wrapped sampler still reconstructs the exact epoch order, but the
    DataLoader only sees indices after ``start_index``.  This keeps resumed
    training deterministic while avoiding expensive dataset reads for every
    batch that was completed in an earlier session.
    """

    def __init__(self, sampler: Sampler[int]) -> None:
        self.sampler = sampler
        self.start_index = 0

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.sampler, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))

    def set_start_index(self, start_index: int) -> None:
        self.start_index = min(max(0, int(start_index)), len(self.sampler))

    def __iter__(self) -> Iterator[int]:
        return islice(iter(self.sampler), self.start_index, None)

    def __len__(self) -> int:
        return max(0, len(self.sampler) - self.start_index)


def set_loader_start_batch(loader: Any, start_batch: int) -> bool:
    """Move a resumable DataLoader to a completed batch boundary.

    Returns ``False`` for legacy or third-party loaders whose sampler does not
    support offsets, allowing the trainer to retain its safe sequential-skip
    fallback.
    """

    sampler = getattr(loader, "sampler", None)
    setter = getattr(sampler, "set_start_index", None)
    batch_size = getattr(loader, "batch_size", None)
    if not callable(setter) or batch_size is None:
        return False
    setter(max(0, int(start_batch)) * int(batch_size))
    return True


class DeterministicEpochSampler(Sampler[int]):
    """Epoch-seeded sampler whose order can be reconstructed after resume."""

    def __init__(self, data_source: Sized, seed: int = 42, shuffle: bool = True) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if not self.shuffle:
            return iter(range(n))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(n, generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


class NonPaddingDistributedSampler(Sampler[int]):
    """Distributed sampler for evaluation that never duplicates samples.

    PyTorch's DistributedSampler pads indices so every rank has the same
    number of samples. That is useful for training throughput, but it pollutes
    validation metrics because some examples are counted more than once.
    """

    def __init__(
        self,
        data_source: Sized,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 42,
        max_samples_per_rank: int | None = None,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.data_source = data_source
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_samples_per_rank = (
            None if max_samples_per_rank is None else max(0, int(max_samples_per_rank))
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ordered_indices(self) -> list[int]:
        n = len(self.data_source)
        if not self.shuffle:
            if self.max_samples_per_rank is None:
                return list(range(n))
            limit = min(n, self.max_samples_per_rank * self.num_replicas)
            return list(range(limit))
        if self.max_samples_per_rank is not None:
            sample_size = min(n, self.max_samples_per_rank * self.num_replicas)
            rng = random.Random(self.seed + self.epoch)
            return rng.sample(range(n), sample_size)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(n, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        indices = self._ordered_indices()
        return iter(indices[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        n = len(self.data_source)
        if self.max_samples_per_rank is not None:
            n = min(n, self.max_samples_per_rank * self.num_replicas)
        if n <= self.rank:
            return 0
        return (n - 1 - self.rank) // self.num_replicas + 1


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
