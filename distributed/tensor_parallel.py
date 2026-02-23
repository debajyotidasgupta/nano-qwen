"""Tensor Parallelism (TP) utilities.

Implements column-parallel and row-parallel linear layers for splitting
attention/FFN projections across TP ranks.  Uses PyTorch's DTensor APIs
when available, with manual all-reduce fallback.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    HAS_TP_API = True
except ImportError:
    HAS_TP_API = False


def apply_tensor_parallel(
    model: nn.Module,
    tp_mesh: object,
) -> nn.Module:
    """Apply tensor parallelism to attention and FFN projections.

    Column-parallel: Q, K, V, gate_proj, up_proj
    Row-parallel: o_proj, down_proj
    """
    if not HAS_TP_API:
        raise RuntimeError("Tensor parallel API not available — requires PyTorch >= 2.3")

    from model.transformer import TransformerBlock

    for module in model.modules():
        if isinstance(module, TransformerBlock):
            plan = {
                "attention.q_proj": ColwiseParallel(),
                "attention.k_proj": ColwiseParallel(),
                "attention.v_proj": ColwiseParallel(),
                "attention.o_proj": RowwiseParallel(),
            }

            if hasattr(module.feed_forward, "gate_proj"):
                plan.update({
                    "feed_forward.gate_proj": ColwiseParallel(),
                    "feed_forward.up_proj": ColwiseParallel(),
                    "feed_forward.down_proj": RowwiseParallel(),
                })

            parallelize_module(module, tp_mesh, plan)

    return model


class ColumnParallelLinear(nn.Module):
    """Linear layer split along the output dimension across TP ranks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group
        world_size = dist.get_world_size(process_group) if process_group else 1
        assert out_features % world_size == 0

        self.local_out_features = out_features // world_size
        self.linear = nn.Linear(in_features, self.local_out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """Linear layer split along the input dimension; all-reduce in forward."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group
        world_size = dist.get_world_size(process_group) if process_group else 1
        assert in_features % world_size == 0

        self.local_in_features = in_features // world_size
        self.linear = nn.Linear(self.local_in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)
        if self.process_group is not None and dist.get_world_size(self.process_group) > 1:
            dist.all_reduce(output, group=self.process_group)
        return output
