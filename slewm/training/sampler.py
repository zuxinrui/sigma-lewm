"""Epoch-addressable sampler for bit-exact paper-run restarts."""

from __future__ import annotations

import torch


class EpochSeededRandomSampler(torch.utils.data.Sampler[int]):
    """Shuffle from ``seed + epoch`` instead of mutable generator state.

    A fresh Lightning Trainer can construct and prefetch a multi-worker
    DataLoader *before* it restores/calls ``set_epoch``.  Formal one-epoch
    invocations therefore pass the zero-based resumed epoch explicitly through
    ``initial_epoch``.  Lightning's later ``set_epoch`` call remains compatible
    and is still what advances uninterrupted multi-epoch runs.
    """

    def __init__(self, data_source, *, seed: int, initial_epoch: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        if int(initial_epoch) < 0:
            raise ValueError("initial sampler epoch must be non-negative")
        self.epoch = int(initial_epoch)

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self), generator=generator)
        yield from permutation.tolist()


__all__ = ["EpochSeededRandomSampler"]
