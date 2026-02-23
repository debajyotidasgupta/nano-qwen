"""Interleaved Multimodal RoPE (MRoPE) for Qwen3-VL.

Splits RoPE frequency bands across 3 dimensions (temporal, height, width)
in an interleaved layout.  Text tokens use standard 1D positions mapped
to all three components; image/video tokens use separate 3D coordinates.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.rope import RotaryEmbedding


class MultimodalRoPE(nn.Module):
    """Interleaved 3D rotary position embedding for vision-language models.

    The head_dim is split into 3 equal parts (temporal, height, width).
    For text tokens all three position components share the same 1D position.
    For image tokens the components hold (frame_idx, row_idx, col_idx).
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1_000_000.0,
    ):
        super().__init__()
        assert head_dim % 3 == 0, f"head_dim={head_dim} must be divisible by 3 for MRoPE"
        self.head_dim = head_dim
        self.section_dim = head_dim // 3

        self.rope_t = RotaryEmbedding(self.section_dim, max_position_embeddings, base)
        self.rope_h = RotaryEmbedding(self.section_dim, max_position_embeddings, base)
        self.rope_w = RotaryEmbedding(self.section_dim, max_position_embeddings, base)

    def forward(
        self,
        x: torch.Tensor,
        position_ids_t: Optional[torch.Tensor] = None,
        position_ids_h: Optional[torch.Tensor] = None,
        position_ids_w: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute interleaved 3D cos/sin tables.

        Args:
            x: reference tensor for dtype/device
            position_ids_t: (B, S) temporal positions
            position_ids_h: (B, S) height positions
            position_ids_w: (B, S) width positions

        Returns:
            cos, sin each (B, S, head_dim) or (S, head_dim)
        """
        cos_t, sin_t = self.rope_t(x, position_ids_t)
        cos_h, sin_h = self.rope_h(x, position_ids_h)
        cos_w, sin_w = self.rope_w(x, position_ids_w)

        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)
        return cos, sin


def build_mrope_position_ids(
    text_positions: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    image_token_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct 3D position IDs for a mixed text-image sequence.

    Args:
        text_positions: (B, S) standard 1D positions for the full sequence
        image_grid_thw: (num_images, 3) — T, H, W for each image
        image_token_mask: (B, S) bool mask indicating image token positions

    Returns:
        position_ids_t, position_ids_h, position_ids_w each (B, S)
    """
    B, S = text_positions.shape

    pos_t = text_positions.clone()
    pos_h = text_positions.clone()
    pos_w = text_positions.clone()

    if image_grid_thw is None or image_token_mask is None:
        return pos_t, pos_h, pos_w

    # For image tokens, overwrite with spatial coordinates
    for b in range(B):
        img_mask = image_token_mask[b]
        img_positions = img_mask.nonzero(as_tuple=True)[0]

        if img_positions.numel() == 0:
            continue

        offset = 0
        for img_idx in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[img_idx].tolist()
            num_tokens = int(t * h * w)

            if offset + num_tokens > img_positions.numel():
                break

            for ti in range(int(t)):
                for hi in range(int(h)):
                    for wi in range(int(w)):
                        flat_idx = ti * int(h) * int(w) + hi * int(w) + wi
                        if offset + flat_idx < img_positions.numel():
                            pos = img_positions[offset + flat_idx]
                            pos_t[b, pos] = ti
                            pos_h[b, pos] = hi
                            pos_w[b, pos] = wi

            offset += num_tokens

    return pos_t, pos_h, pos_w
