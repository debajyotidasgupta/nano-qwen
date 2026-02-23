"""Mixture of Experts (MoE) layer — Qwen3 design.

Key design choices matching Qwen3-235B-A22B:
  - 128 fine-grained experts per layer, top-8 routing
  - No shared experts (departure from Qwen2.5-MoE / DeepSeek-MoE)
  - Global-batch load-balancing auxiliary loss
  - Softmax normalization of selected expert weights
  - SwiGLU expert FFNs with smaller intermediate size
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.ffn import SwiGLU


class TopKRouter(nn.Module):
    """Token-to-expert routing with top-k selection and load-balancing loss."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.aux_loss_coef = config.router_aux_loss_coef

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts.

        Args:
            hidden_states: (num_tokens, hidden_size)

        Returns:
            expert_weights: (num_tokens, top_k) — normalized routing weights
            expert_indices: (num_tokens, top_k) — selected expert IDs
            aux_loss: scalar load-balancing loss
        """
        router_logits = self.gate(hidden_states)  # (N, E)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)

        if self.norm_topk_prob:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        expert_weights = expert_weights.to(hidden_states.dtype)

        aux_loss = self._load_balancing_loss(router_probs, expert_indices)

        return expert_weights, expert_indices, aux_loss

    def _load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load-balancing auxiliary loss.

        Uses the standard Switch Transformer formulation:
            L_aux = N_experts * sum_i(f_i * p_i)
        where f_i is the fraction of tokens routed to expert i
        and p_i is the average routing probability for expert i.

        This is computed at the local batch level; for global-batch balancing,
        the distributed training code aggregates across DP ranks before
        computing the final loss.
        """
        num_tokens = router_probs.shape[0]

        # f_i: fraction of tokens routed to each expert
        one_hot = F.one_hot(expert_indices, self.num_experts).float()  # (N, K, E)
        tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)  # (E,)
        f = tokens_per_expert / (num_tokens * self.top_k)

        # p_i: average routing probability for each expert
        p = router_probs.mean(dim=0)  # (E,)

        return self.num_experts * (f * p).sum()


class ExpertFFN(nn.Module):
    """A single MoE expert: SwiGLU FFN with expert-specific intermediate size."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoELayer(nn.Module):
    """Full MoE layer: router + N experts + dispatch/combine logic."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.drop_tokens = config.moe_drop_tokens
        self.capacity_factor = config.moe_capacity_factor

        self.router = TopKRouter(config)
        self.experts = nn.ModuleList([
            ExpertFFN(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            output: (batch, seq_len, hidden_size)
            aux_loss: scalar
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, orig_shape[-1])  # (N, D)
        num_tokens, D = hidden_states.shape

        expert_weights, expert_indices, aux_loss = self.router(hidden_states)
        # expert_weights: (N, K), expert_indices: (N, K)

        if self.drop_tokens:
            output = self._forward_with_capacity(hidden_states, expert_weights, expert_indices)
        else:
            output = self._forward_dropless(hidden_states, expert_weights, expert_indices)

        return output.view(orig_shape), aux_loss

    def _forward_dropless(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Process all tokens without dropping — iterate over experts."""
        N, D = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert and their corresponding top-k slot
            mask = (expert_indices == expert_idx)  # (N, K) bool
            if not mask.any():
                continue

            token_ids, slot_ids = mask.nonzero(as_tuple=True)
            expert_input = hidden_states[token_ids]
            expert_output = self.experts[expert_idx](expert_input)
            weights = expert_weights[token_ids, slot_ids].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_output * weights)

        return output

    def _forward_with_capacity(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Process tokens with capacity factor — drop overflow tokens."""
        N, D = hidden_states.shape
        capacity = int(self.capacity_factor * N * self.top_k / self.num_experts)
        output = torch.zeros_like(hidden_states)

        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx)
            if not mask.any():
                continue

            token_ids, slot_ids = mask.nonzero(as_tuple=True)

            # Drop tokens exceeding capacity
            if token_ids.shape[0] > capacity:
                token_ids = token_ids[:capacity]
                slot_ids = slot_ids[:capacity]

            expert_input = hidden_states[token_ids]
            expert_output = self.experts[expert_idx](expert_input)
            weights = expert_weights[token_ids, slot_ids].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_output * weights)

        return output


class GlobalBatchLoadBalancer:
    """Accumulates routing statistics across micro-batches for global-batch
    load-balancing loss computation.

    Usage:
        balancer = GlobalBatchLoadBalancer(num_experts)
        for micro_batch in gradient_accumulation_steps:
            ...
            balancer.update(router_probs, expert_indices)
        global_aux_loss = balancer.compute_loss()
        balancer.reset()
    """

    def __init__(self, num_experts: int, top_k: int):
        self.num_experts = num_experts
        self.top_k = top_k
        self.total_tokens = 0
        self.tokens_per_expert = None
        self.prob_sum_per_expert = None
        self.num_updates = 0

    def update(self, router_probs: torch.Tensor, expert_indices: torch.Tensor):
        N = router_probs.shape[0]
        one_hot = F.one_hot(expert_indices, self.num_experts).float()
        tpe = one_hot.sum(dim=1).sum(dim=0)
        pse = router_probs.sum(dim=0)

        if self.tokens_per_expert is None:
            self.tokens_per_expert = tpe
            self.prob_sum_per_expert = pse
        else:
            self.tokens_per_expert += tpe
            self.prob_sum_per_expert += pse

        self.total_tokens += N
        self.num_updates += 1

    def compute_loss(self) -> torch.Tensor:
        if self.total_tokens == 0:
            return torch.tensor(0.0)
        f = self.tokens_per_expert / (self.total_tokens * self.top_k)
        p = self.prob_sum_per_expert / self.total_tokens
        return self.num_experts * (f * p).sum()

    def reset(self):
        self.total_tokens = 0
        self.tokens_per_expert = None
        self.prob_sum_per_expert = None
        self.num_updates = 0
