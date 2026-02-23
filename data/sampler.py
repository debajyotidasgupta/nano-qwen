"""Distributed data sampler with deterministic shuffling and checkpoint resume."""

from __future__ import annotations

import math
from typing import Optional, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset


class DistributedSampler(Sampler):
    """Deterministic sampler that shards data across distributed ranks.

    Supports:
      - Reproducible shuffling via seed
      - Resume from a specific sample offset (for checkpoint recovery)
      - Even distribution across ranks with optional padding
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        start_index: int = 0,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.start_index = start_index
        self.epoch = 0

        total = len(dataset)
        if drop_last:
            self.num_samples = total // num_replicas
        else:
            self.num_samples = math.ceil(total / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = rng.permutation(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Pad to total_size
        if len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            indices += indices[:padding]

        # Shard
        indices = indices[self.rank::self.num_replicas]

        # Resume from checkpoint offset
        if self.start_index > 0:
            indices = indices[self.start_index:]

        return iter(indices)

    def __len__(self) -> int:
        return max(0, self.num_samples - self.start_index)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def set_start_index(self, start_index: int):
        """Set resume offset for checkpoint recovery."""
        self.start_index = start_index

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "start_index": self.start_index}

    def load_state_dict(self, state: dict):
        self.epoch = state.get("epoch", 0)
        self.start_index = state.get("start_index", 0)
