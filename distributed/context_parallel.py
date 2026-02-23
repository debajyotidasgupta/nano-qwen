"""Context Parallelism (CP) — sequence splitting with ring attention.

Splits long sequences across CP ranks and uses ring attention for
cross-chunk communication.  Useful for training with 128K+ contexts
where a single GPU cannot hold the full sequence's activations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist


def split_sequence(
    hidden_states: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Split sequence dimension across CP ranks.

    Args:
        hidden_states: (B, S, D)
        cp_rank: this rank's index in the CP group
        cp_size: total CP ranks

    Returns:
        (B, S // cp_size, D) local chunk
    """
    B, S, D = hidden_states.shape
    assert S % cp_size == 0, f"seq_len={S} not divisible by cp_size={cp_size}"
    chunk_size = S // cp_size
    return hidden_states[:, cp_rank * chunk_size : (cp_rank + 1) * chunk_size].contiguous()


def gather_sequence(
    local_hidden: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-gather local chunks back to full sequence."""
    if cp_group is None or dist.get_world_size(cp_group) <= 1:
        return local_hidden

    world_size = dist.get_world_size(cp_group)
    gathered = [torch.empty_like(local_hidden) for _ in range(world_size)]
    dist.all_gather(gathered, local_hidden, group=cp_group)
    return torch.cat(gathered, dim=1)


class RingAttention(nn.Module):
    """Ring attention for context parallelism.

    Each CP rank holds a chunk of Q and iteratively receives K/V chunks
    from other ranks in a ring pattern, accumulating the attention output
    with online softmax rescaling.
    """

    def __init__(self, cp_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.cp_group = cp_group
        self.cp_size = dist.get_world_size(cp_group) if cp_group else 1
        self.cp_rank = dist.get_rank(cp_group) if cp_group else 0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Ring attention forward.

        Args:
            q, k, v: (B, H, S_local, D) — local sequence chunks.

        Returns:
            (B, H, S_local, D) attention output.
        """
        if self.cp_size <= 1:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

        B, H, S, D = q.shape
        device = q.device

        # Running max and sum for online softmax
        out = torch.zeros_like(q)
        max_score = torch.full((B, H, S, 1), float("-inf"), device=device)
        sum_exp = torch.zeros(B, H, S, 1, device=device)

        k_recv = torch.empty_like(k)
        v_recv = torch.empty_like(v)

        k_send = k.contiguous()
        v_send = v.contiguous()

        for step in range(self.cp_size):
            src_rank = (self.cp_rank + step) % self.cp_size

            if step < self.cp_size - 1:
                next_rank = (self.cp_rank + 1) % self.cp_size
                prev_rank = (self.cp_rank - 1) % self.cp_size

                send_k_op = dist.isend(k_send, next_rank, group=self.cp_group)
                send_v_op = dist.isend(v_send, next_rank, group=self.cp_group)
                recv_k_op = dist.irecv(k_recv, prev_rank, group=self.cp_group)
                recv_v_op = dist.irecv(v_recv, prev_rank, group=self.cp_group)

            # Compute attention for this chunk
            scale = D ** -0.5
            scores = torch.matmul(q, k_send.transpose(-2, -1)) * scale  # (B, H, S, S_kv)

            if causal and src_rank > self.cp_rank:
                scores.fill_(float("-inf"))
            elif causal and src_rank == self.cp_rank:
                mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
                scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            chunk_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(max_score, chunk_max)

            exp_scores = torch.exp(scores - new_max)
            exp_sum = exp_scores.sum(dim=-1, keepdim=True)

            correction = torch.exp(max_score - new_max)
            out = out * correction + torch.matmul(exp_scores, v_send)
            sum_exp = sum_exp * correction + exp_sum
            max_score = new_max

            if step < self.cp_size - 1:
                send_k_op.wait()
                send_v_op.wait()
                recv_k_op.wait()
                recv_v_op.wait()
                k_send = k_recv.clone()
                v_send = v_recv.clone()

        out = out / sum_exp
        return out
