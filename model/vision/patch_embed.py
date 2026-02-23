"""Patch embedding for Vision Transformer.

Converts (B, C, H, W) images into (B, num_patches, embed_dim) patch tokens
using a 2D convolution with stride = patch_size.
"""

import torch
import torch.nn as nn

from model.config import ModelConfig


class PatchEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_size = config.vision_patch_size
        self.proj = nn.Conv2d(
            config.vision_num_channels,
            config.vision_hidden_size,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
            bias=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim) where num_patches = (H/P) * (W/P)
        """
        x = self.proj(pixel_values)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        return x
