"""Optimizer configuration — AdamW with per-parameter weight decay."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torch.optim import AdamW as _AdamW
    # Fused AdamW available in PyTorch >= 2.1
    _FUSED_AVAILABLE = "fused" in _AdamW.__init__.__code__.co_varnames
except Exception:
    _FUSED_AVAILABLE = False


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    fused: bool = True,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with per-parameter weight decay.

    Skips weight decay for:
      - bias parameters
      - LayerNorm / RMSNorm weights
      - Embedding weights
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or "norm" in name or "bias" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = fused and _FUSED_AVAILABLE and torch.cuda.is_available()

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        fused=use_fused,
    )

    return optimizer
