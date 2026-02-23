"""Rotary Position Embeddings (RoPE) with YARN extension.

Implements:
  - Standard RoPE (Su et al., 2021)
  - YARN extension for context-length scaling (Peng et al., 2023)
  - Adjustable Base Frequency (ABF) via rope_theta
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _compute_freqs(
    dim: int,
    end: int,
    theta: float = 1_000_000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute the complex rotation frequencies for standard RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(end, device=device, dtype=dtype)
    freqs = torch.outer(t, freqs)
    return freqs


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    """Find the RoPE dimension where a given number of rotations occurs at max_position_embeddings."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
) -> Tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(low: int, high: int, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return linear.clamp(0, 1)


def compute_yarn_freqs(
    dim: int,
    max_position_embeddings: int,
    base: float = 1_000_000.0,
    scale: float = 1.0,
    original_max_position_embeddings: int = 32768,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute RoPE frequencies with YARN context-length scaling."""
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    freq_inter = 1.0 / (scale * base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings
    )
    inv_freq_mask = 1.0 - _yarn_linear_ramp(low, high, dim // 2).to(device)
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embedding with optional YARN scaling."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1_000_000.0,
        rope_scaling: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling

        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            scale = rope_scaling.get("factor", 4.0)
            original_max = rope_scaling.get("original_max_position_embeddings", 32768)
            freqs = compute_yarn_freqs(
                dim, max_position_embeddings, base, scale, original_max, device=device,
            )
        else:
            freqs = _compute_freqs(dim, max_position_embeddings, base, device=device)

        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables for the given positions.

        Args:
            x: Input tensor, only used for dtype/device.
            position_ids: (batch, seq_len) position indices.

        Returns:
            cos, sin each of shape (batch, seq_len, dim) or broadcastable.
        """
        if position_ids is None:
            seq_len = x.shape[-2]
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)
        else:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to Q and K tensors.

    Expected shapes: q, k = (batch, n_heads, seq_len, head_dim)
                     cos, sin = (seq_len, head_dim//2) or (batch, seq_len, head_dim//2)
    """
    def _rotate(x: torch.Tensor, cos_: torch.Tensor, sin_: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]

        # Broadcast cos/sin to (batch, 1, seq_len, half)
        if cos_.dim() == 2:
            cos_ = cos_.unsqueeze(0).unsqueeze(0)
            sin_ = sin_.unsqueeze(0).unsqueeze(0)
        elif cos_.dim() == 3:
            cos_ = cos_.unsqueeze(1)
            sin_ = sin_.unsqueeze(1)

        return torch.cat([
            x1 * cos_ - x2 * sin_,
            x2 * cos_ + x1 * sin_,
        ], dim=-1)

    return _rotate(q, cos, sin), _rotate(k, cos, sin)
