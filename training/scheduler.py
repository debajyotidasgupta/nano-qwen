"""Learning rate schedulers — warmup + cosine decay, WSD."""

from __future__ import annotations

import math
from typing import Optional

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine decay to min_lr.

    lr(step) =
      - warmup: lr * step / warmup_steps
      - decay:  min_lr + 0.5 * (lr - min_lr) * (1 + cos(pi * progress))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 2000,
        total_steps: int = 100000,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio

            if step < self.warmup_steps:
                lr = base_lr * step / max(1, self.warmup_steps)
            elif step >= self.total_steps:
                lr = min_lr
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

            lrs.append(lr)
        return lrs


class WSDScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Warmup-Stable-Decay (WSD) scheduler.

    Three phases:
      1. Warmup: linear increase to max_lr
      2. Stable: constant at max_lr
      3. Decay:  linear decay to min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 2000,
        stable_steps: int = 80000,
        decay_steps: int = 18000,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.total_steps = warmup_steps + stable_steps + decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio

            if step < self.warmup_steps:
                lr = base_lr * step / max(1, self.warmup_steps)
            elif step < self.warmup_steps + self.stable_steps:
                lr = base_lr
            else:
                decay_progress = (step - self.warmup_steps - self.stable_steps) / max(1, self.decay_steps)
                decay_progress = min(decay_progress, 1.0)
                lr = base_lr - (base_lr - min_lr) * decay_progress

            lrs.append(lr)
        return lrs


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr_ratio)
    elif scheduler_type == "wsd":
        stable_steps = kwargs.get("stable_steps", int(total_steps * 0.8))
        decay_steps = total_steps - warmup_steps - stable_steps
        return WSDScheduler(optimizer, warmup_steps, stable_steps, decay_steps, min_lr_ratio)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
