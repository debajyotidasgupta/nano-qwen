"""Expert Parallelism (EP) for MoE models.

Distributes experts across EP ranks and handles all-to-all token dispatch
and combine operations.  When combined with data parallelism, each DP group
holds a full copy of non-expert parameters while expert parameters are
sharded across EP ranks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from distributed.comm import all_to_all


class ExpertParallelMoE(nn.Module):
    """Wraps an MoE layer to distribute experts across EP ranks.

    With EP_size=8 and 128 experts, each rank holds 16 experts.
    Token routing uses all-to-all communication to dispatch tokens to
    the rank that owns their assigned expert.
    """

    def __init__(
        self,
        moe_layer: nn.Module,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.moe_layer = moe_layer
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
        self.ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0

        if self.ep_size > 1:
            self._shard_experts()

    def _shard_experts(self):
        """Keep only the experts assigned to this EP rank."""
        total_experts = len(self.moe_layer.experts)
        experts_per_rank = total_experts // self.ep_size
        start = self.ep_rank * experts_per_rank
        end = start + experts_per_rank

        self.local_expert_start = start
        self.local_expert_end = end
        self.experts_per_rank = experts_per_rank

        # Replace expert list with only local experts
        local_experts = nn.ModuleList(
            [self.moe_layer.experts[i] for i in range(start, end)]
        )
        self.moe_layer.experts = local_experts

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ep_size <= 1:
            return self.moe_layer(hidden_states)

        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, orig_shape[-1])
        N, D = x.shape

        # Route tokens
        expert_weights, expert_indices, aux_loss = self.moe_layer.router(x)

        # Dispatch: send tokens to their expert's EP rank
        dispatched, recv_counts = self._dispatch_tokens(x, expert_indices, expert_weights)

        # Process with local experts
        output = self._process_local_experts(dispatched, recv_counts)

        # Combine: send expert outputs back to original ranks
        combined = self._combine_tokens(output, recv_counts, N, D)

        return combined.view(orig_shape), aux_loss

    def _dispatch_tokens(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-to-all dispatch of tokens to expert-owning ranks."""
        N, D = x.shape
        K = expert_indices.shape[1]

        # Count tokens going to each EP rank
        send_counts = torch.zeros(self.ep_size, dtype=torch.long, device=x.device)
        for k in range(K):
            for rank in range(self.ep_size):
                start = rank * self.experts_per_rank
                end = start + self.experts_per_rank
                mask = (expert_indices[:, k] >= start) & (expert_indices[:, k] < end)
                send_counts[rank] += mask.sum()

        # Gather receive counts from all ranks
        recv_counts = torch.zeros_like(send_counts)
        if dist.is_initialized():
            dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        else:
            recv_counts = send_counts.clone()

        max_recv = max(recv_counts.max().item(), 1)
        recv_buffer = torch.zeros(max_recv * self.ep_size, D, device=x.device, dtype=x.dtype)

        return recv_buffer, recv_counts

    def _process_local_experts(
        self,
        dispatched: torch.Tensor,
        recv_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Process received tokens through local experts."""
        output = torch.zeros_like(dispatched)
        offset = 0
        for expert_idx, expert in enumerate(self.moe_layer.experts):
            count = recv_counts[self.ep_rank].item() // max(len(self.moe_layer.experts), 1)
            if count > 0:
                expert_input = dispatched[offset : offset + count]
                expert_output = expert(expert_input)
                output[offset : offset + count] = expert_output
                offset += count
        return output

    def _combine_tokens(
        self,
        output: torch.Tensor,
        recv_counts: torch.Tensor,
        num_tokens: int,
        D: int,
    ) -> torch.Tensor:
        """All-to-all combine to return expert outputs to original ranks."""
        return output[:num_tokens]


def apply_expert_parallel(
    model: nn.Module,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> nn.Module:
    """Wrap all MoE layers in the model with ExpertParallelMoE."""
    from model.moe import MoELayer

    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            ep_moe = ExpertParallelMoE(module, ep_group)
            setattr(parent, child_name, ep_moe)

    return model
