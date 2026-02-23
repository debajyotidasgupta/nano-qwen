"""Parallelism orchestrator — compose N-D parallelism strategies.

Sets up a multi-dimensional DeviceMesh (DP, CP, TP, PP, EP) and applies
each parallelism strategy to the model in the correct order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
    HAS_DEVICE_MESH = True
except ImportError:
    HAS_DEVICE_MESH = False
    DeviceMesh = None

from distributed.comm import init_distributed, get_rank, get_world_size


@dataclass
class ParallelConfig:
    dp_size: int = -1       # -1 = auto (world_size / tp / pp / ep)
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    cp_size: int = 1
    fsdp_enabled: bool = True
    dtype: str = "bfloat16"
    reshard_after_forward: bool = True


class ParallelismManager:
    """Orchestrates N-D parallelism setup and model wrapping."""

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.rank, self.world_size = init_distributed()
        self.mesh: Optional[object] = None
        self._setup_mesh()

    def _setup_mesh(self):
        """Create the N-D DeviceMesh."""
        c = self.config

        if c.dp_size <= 0:
            c.dp_size = self.world_size // (c.tp_size * c.pp_size)

        assert c.dp_size * c.tp_size * c.pp_size == self.world_size, (
            f"dp({c.dp_size}) * tp({c.tp_size}) * pp({c.pp_size}) = "
            f"{c.dp_size * c.tp_size * c.pp_size} != world_size({self.world_size})"
        )

        if not HAS_DEVICE_MESH or self.world_size <= 1:
            return

        mesh_dims = []
        mesh_names = []

        if c.pp_size > 1:
            mesh_dims.append(c.pp_size)
            mesh_names.append("pp")
        if c.dp_size > 1:
            mesh_dims.append(c.dp_size)
            mesh_names.append("dp")
        if c.tp_size > 1:
            mesh_dims.append(c.tp_size)
            mesh_names.append("tp")

        if mesh_dims:
            self.mesh = init_device_mesh(
                "cuda",
                tuple(mesh_dims),
                mesh_dim_names=tuple(mesh_names),
            )

    def apply_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply all parallelism strategies to the model.

        Order: TP -> EP -> PP -> FSDP (outermost)
        """
        c = self.config

        # 1. Tensor Parallelism
        if c.tp_size > 1 and self.mesh is not None:
            from distributed.tensor_parallel import apply_tensor_parallel
            tp_mesh = self.mesh["tp"] if "tp" in self.mesh.mesh_dim_names else self.mesh
            model = apply_tensor_parallel(model, tp_mesh)

        # 2. Expert Parallelism
        if c.ep_size > 1:
            from distributed.expert_parallel import apply_expert_parallel
            ep_group = self._create_ep_group()
            model = apply_expert_parallel(model, ep_group)

        # 3. Pipeline Parallelism
        if c.pp_size > 1:
            from distributed.pipeline_parallel import apply_pipeline_parallel
            pp_mesh = self.mesh["pp"] if self.mesh and "pp" in self.mesh.mesh_dim_names else None
            split_spec, stage_layers = apply_pipeline_parallel(model, pp_mesh, c.pp_size)
            # Store for the training loop to set up the schedule
            self._pp_split_spec = split_spec
            self._pp_stage_layers = stage_layers

        # 4. FSDP2 (outermost)
        if c.fsdp_enabled and self.world_size > 1:
            from distributed.fsdp import apply_fsdp2, get_mixed_precision_policy
            dp_mesh = self.mesh["dp"] if self.mesh and "dp" in self.mesh.mesh_dim_names else None
            mp_policy = get_mixed_precision_policy(c.dtype)
            model = apply_fsdp2(
                model,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=c.reshard_after_forward,
            )

        return model

    def _create_ep_group(self) -> Optional[dist.ProcessGroup]:
        """Create process groups for expert parallelism."""
        if self.world_size <= 1:
            return None

        ep_size = self.config.ep_size
        num_ep_groups = self.world_size // ep_size
        ep_group = None

        for i in range(num_ep_groups):
            ranks = list(range(i * ep_size, (i + 1) * ep_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                ep_group = group

        return ep_group

    @property
    def dp_mesh(self):
        if self.mesh and "dp" in self.mesh.mesh_dim_names:
            return self.mesh["dp"]
        return None

    @property
    def tp_mesh(self):
        if self.mesh and "tp" in self.mesh.mesh_dim_names:
            return self.mesh["tp"]
        return None
