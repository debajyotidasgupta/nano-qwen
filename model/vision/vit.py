"""Vision Transformer (ViT) encoder for Qwen3-VL.

Standard ViT architecture with pre-norm (LayerNorm), multi-head self-attention,
and MLP blocks.  Supports extracting intermediate hidden states for DeepStack.
"""

from __future__ import annotations

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.vision.patch_embed import PatchEmbedding


class ViTMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ViTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        return self.proj(attn_out)


class ViTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = ViTAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = ViTMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT encoder that returns both final and intermediate hidden states."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)

        num_patches = config.vision_num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, config.vision_hidden_size)
        )

        self.blocks = nn.ModuleList([
            ViTBlock(
                config.vision_hidden_size,
                config.vision_num_attention_heads,
                config.vision_intermediate_size,
            )
            for _ in range(config.vision_num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.vision_hidden_size)

        # Which layers to extract for DeepStack
        self.deepstack_layers = set(config.vision_deepstack_layers) if config.vision_deepstack_layers else set()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> dict:
        """
        Args:
            pixel_values: (B, C, H, W)

        Returns:
            dict with:
                "last_hidden_state": (B, num_patches, D)
                "intermediate_states": list of (B, num_patches, D) for DeepStack layers
        """
        x = self.patch_embed(pixel_values)

        # Handle variable resolution: interpolate pos_embed if needed
        if x.shape[1] != self.pos_embed.shape[1]:
            x = x + self._interpolate_pos_embed(x)
        else:
            x = x + self.pos_embed

        intermediates = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.deepstack_layers:
                intermediates.append(x)

        x = self.norm(x)

        return {
            "last_hidden_state": x,
            "intermediate_states": intermediates,
        }

    def _interpolate_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of position embeddings for arbitrary resolutions."""
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N:
            return self.pos_embed

        dim = x.shape[-1]
        h0 = w0 = int(math.sqrt(N))
        h1 = w1 = int(math.sqrt(npatch))

        pos = self.pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(h1, w1), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return pos
