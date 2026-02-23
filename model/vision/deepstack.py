"""DeepStack — multi-level ViT feature fusion for Qwen3-VL.

Collects hidden states from multiple intermediate ViT layers and fuses them
via learned projections to produce enriched visual token representations
that capture both low-level details and high-level semantics.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DeepStack(nn.Module):
    """Fuse multi-level ViT features into the LLM's hidden dimension."""

    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        num_levels: int,
    ):
        super().__init__()
        self.num_levels = num_levels

        # Per-level projection from ViT dim -> LLM dim
        self.level_projs = nn.ModuleList([
            nn.Linear(vision_hidden_size, llm_hidden_size)
            for _ in range(num_levels)
        ])

        # Learned fusion weights (softmax-normalized)
        self.fusion_weights = nn.Parameter(torch.ones(num_levels))
        self.norm = nn.LayerNorm(llm_hidden_size)

    def forward(
        self,
        intermediate_states: List[torch.Tensor],
        last_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            intermediate_states: list of (B, N, vision_D) from selected ViT layers
            last_hidden_state: (B, N, vision_D) final ViT output

        Returns:
            (B, N, llm_D) fused representation
        """
        all_states = intermediate_states + [last_hidden_state]

        if len(all_states) != self.num_levels:
            # Fallback: just project the last hidden state
            return self.norm(self.level_projs[-1](last_hidden_state))

        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = torch.zeros_like(self.level_projs[0](all_states[0]))

        for i, (state, proj) in enumerate(zip(all_states, self.level_projs)):
            fused = fused + weights[i] * proj(state)

        return self.norm(fused)
