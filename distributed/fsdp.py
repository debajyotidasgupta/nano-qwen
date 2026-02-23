"""FSDP2 wrapping utilities.

Provides helper functions to apply PyTorch FSDP2 (fully_shard) to the model
with configurable sharding granularity, mixed precision, and CPU offloading.
"""

from __future__ import annotations

from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    HAS_FSDP2 = True
except ImportError:
    HAS_FSDP2 = False
    fully_shard = None
    MixedPrecisionPolicy = None

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (
        ShardingStrategy,
        MixedPrecision,
        CPUOffload,
    )
    HAS_FSDP1 = True
except ImportError:
    HAS_FSDP1 = False


def get_mixed_precision_policy(dtype: str = "bfloat16"):
    """Create a mixed precision policy for FSDP2."""
    dt = getattr(torch, dtype, torch.bfloat16)
    if HAS_FSDP2 and MixedPrecisionPolicy is not None:
        return MixedPrecisionPolicy(param_dtype=dt, reduce_dtype=torch.float32)
    return None


def apply_fsdp2(
    model: nn.Module,
    mesh: Optional[object] = None,
    mp_policy: Optional[object] = None,
    reshard_after_forward: bool = True,
) -> nn.Module:
    """Apply FSDP2 (fully_shard) to each transformer block and the full model.

    Shards at the granularity of individual TransformerBlock modules, then
    wraps the entire model for the root shard.
    """
    if not HAS_FSDP2:
        raise RuntimeError("FSDP2 not available — requires PyTorch >= 2.5")

    from model.transformer import TransformerBlock

    fsdp_kwargs = {}
    if mesh is not None:
        fsdp_kwargs["mesh"] = mesh
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
    fsdp_kwargs["reshard_after_forward"] = reshard_after_forward

    # Shard each transformer block individually
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, **fsdp_kwargs)

    # Root shard for the full model
    fully_shard(model, **fsdp_kwargs)

    return model


def apply_fsdp1(
    model: nn.Module,
    dtype: str = "bfloat16",
    cpu_offload: bool = False,
    sharding_strategy: str = "FULL_SHARD",
) -> nn.Module:
    """Fallback: apply FSDP1 wrapping."""
    if not HAS_FSDP1:
        raise RuntimeError("FSDP1 not available")

    dt = getattr(torch, dtype, torch.bfloat16)
    mp = MixedPrecision(param_dtype=dt, reduce_dtype=torch.float32, buffer_dtype=dt)

    strategy = getattr(ShardingStrategy, sharding_strategy, ShardingStrategy.FULL_SHARD)

    model = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        use_orig_params=True,
        limit_all_gathers=True,
    )
    return model
