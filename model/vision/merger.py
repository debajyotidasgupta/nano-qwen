"""Vision-Language Merger — compress visual tokens before feeding to the LLM.

Performs spatial pooling (e.g., 2x2 adjacent patch merging) to reduce the
number of visual tokens, then projects to the LLM's hidden dimension.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class VisionMerger(nn.Module):
    """Merge adjacent visual patches and project to LLM hidden size."""

    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        spatial_factor: int = 2,
    ):
        super().__init__()
        self.spatial_factor = spatial_factor
        merged_dim = vision_hidden_size * (spatial_factor ** 2)

        self.mlp = nn.Sequential(
            nn.LayerNorm(merged_dim),
            nn.Linear(merged_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, visual_tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, H*W, D) patch tokens
            grid_h, grid_w: spatial grid dimensions

        Returns:
            (B, H'*W', llm_D) merged tokens where H'=H/factor, W'=W/factor
        """
        B, N, D = visual_tokens.shape
        f = self.spatial_factor

        assert grid_h % f == 0 and grid_w % f == 0, (
            f"Grid ({grid_h}, {grid_w}) not divisible by spatial_factor={f}"
        )

        # Reshape to spatial grid, then pool adjacent patches
        x = visual_tokens.view(B, grid_h, grid_w, D)
        x = x.view(B, grid_h // f, f, grid_w // f, f, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, (grid_h // f) * (grid_w // f), f * f * D)

        return self.mlp(x)
