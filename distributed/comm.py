"""Communication primitives and overlap utilities for distributed training."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def init_distributed(
    backend: str = "nccl",
    timeout_minutes: int = 30,
) -> tuple[int, int]:
    """Initialize the distributed process group.

    Returns (rank, world_size).
    """
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.distributed_c10d._DEFAULT_PG_TIMEOUT
            if timeout_minutes <= 0
            else __import__("datetime").timedelta(minutes=timeout_minutes),
        )
    return rank, world_size


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor with mean reduction."""
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> list[torch.Tensor]:
    """All-gather a tensor from all ranks."""
    if not dist.is_initialized() or get_world_size() == 1:
        return [tensor]
    gathered = [torch.empty_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return gathered


class AsyncAllReduce:
    """Non-blocking all-reduce for overlapping communication with computation."""

    def __init__(self):
        self._handle: Optional[dist.Work] = None
        self._tensor: Optional[torch.Tensor] = None

    def start(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        if not dist.is_initialized() or get_world_size() == 1:
            self._tensor = tensor
            return
        self._tensor = tensor
        self._handle = dist.all_reduce(tensor, op=op, async_op=True)

    def wait(self) -> torch.Tensor:
        if self._handle is not None:
            self._handle.wait()
        return self._tensor


def all_to_all(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """All-to-all communication for expert parallelism."""
    if not dist.is_initialized() or get_world_size() == 1:
        for i, t in enumerate(input_tensors):
            output_tensors[i].copy_(t)
        return
    dist.all_to_all(output_tensors, input_tensors, group=group)
