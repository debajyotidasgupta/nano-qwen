"""SwiGLU Feed-Forward Network.

Implements the gated FFN used in Qwen3:
    FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

Reference: Shazeer, 2020 — "GLU Variants Improve Transformer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network with gate, up, and down projections."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def build_ffn(config: ModelConfig) -> SwiGLU:
    return SwiGLU(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        bias=False,
    )
