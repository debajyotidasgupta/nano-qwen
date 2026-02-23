"""Triton kernels for MoE token dispatch and combine.

Provides:
  - scatter_tokens: routes tokens to their assigned experts
  - gather_tokens: combines expert outputs weighted by routing probabilities
  - PyTorch fallbacks for non-CUDA environments
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _scatter_kernel(
        INPUT_ptr, OUTPUT_ptr,
        INDICES_ptr, WEIGHTS_ptr,
        stride_in_tok, stride_out_tok,
        D: tl.constexpr,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Scatter tokens to expert slots: output[flat_idx] = input[token] * weight."""
        tok = tl.program_id(0)
        k_idx = tl.program_id(1)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        expert_flat_idx = tl.load(INDICES_ptr + tok * K + k_idx)
        weight = tl.load(WEIGHTS_ptr + tok * K + k_idx)

        x = tl.load(INPUT_ptr + tok * stride_in_tok + cols, mask=mask, other=0.0)
        tl.store(OUTPUT_ptr + expert_flat_idx * stride_out_tok + cols, x * weight, mask=mask)

    @triton.jit
    def _gather_kernel(
        EXPERT_OUT_ptr, OUTPUT_ptr,
        INDICES_ptr, WEIGHTS_ptr,
        stride_eo_tok, stride_out_tok,
        D: tl.constexpr,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Gather expert outputs back to token positions with weighted sum."""
        tok = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for k in range(K):
            expert_flat_idx = tl.load(INDICES_ptr + tok * K + k)
            weight = tl.load(WEIGHTS_ptr + tok * K + k)
            eo = tl.load(EXPERT_OUT_ptr + expert_flat_idx * stride_eo_tok + cols, mask=mask, other=0.0)
            acc += eo.to(tl.float32) * weight

        tl.store(OUTPUT_ptr + tok * stride_out_tok + cols, acc, mask=mask)


def triton_scatter_tokens(
    x: torch.Tensor,
    flat_indices: torch.Tensor,
    weights: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    """Scatter tokens to expert slots.

    Args:
        x: (num_tokens, D) input tokens
        flat_indices: (num_tokens, K) flat expert-slot indices
        weights: (num_tokens, K) routing weights
        num_slots: total number of expert slots

    Returns:
        (num_slots, D) scattered expert inputs
    """
    N, D = x.shape
    K = flat_indices.shape[1]
    out = torch.zeros(num_slots, D, device=x.device, dtype=x.dtype)
    BLOCK_D = triton.next_power_of_2(D)

    _scatter_kernel[(N, K)](
        x, out, flat_indices, weights,
        x.stride(0), out.stride(0),
        D=D, K=K, BLOCK_D=BLOCK_D,
    )
    return out


def triton_gather_tokens(
    expert_out: torch.Tensor,
    flat_indices: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: int,
    D: int,
) -> torch.Tensor:
    """Gather expert outputs back to original token positions.

    Args:
        expert_out: (num_slots, D) expert outputs
        flat_indices: (num_tokens, K)
        weights: (num_tokens, K)
        num_tokens: original number of tokens
        D: hidden dimension

    Returns:
        (num_tokens, D) combined output
    """
    out = torch.zeros(num_tokens, D, device=expert_out.device, dtype=expert_out.dtype)
    K = flat_indices.shape[1]
    BLOCK_D = triton.next_power_of_2(D)

    _gather_kernel[(num_tokens,)](
        expert_out, out, flat_indices, weights,
        expert_out.stride(0), out.stride(0),
        D=D, K=K, BLOCK_D=BLOCK_D,
    )
    return out


# ---- PyTorch fallbacks ----

def pytorch_scatter_tokens(
    x: torch.Tensor,
    flat_indices: torch.Tensor,
    weights: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    N, D = x.shape
    K = flat_indices.shape[1]
    out = torch.zeros(num_slots, D, device=x.device, dtype=x.dtype)
    for k in range(K):
        idx = flat_indices[:, k]
        out[idx] += x * weights[:, k : k + 1]
    return out


def pytorch_gather_tokens(
    expert_out: torch.Tensor,
    flat_indices: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: int,
    D: int,
) -> torch.Tensor:
    K = flat_indices.shape[1]
    out = torch.zeros(num_tokens, D, device=expert_out.device, dtype=expert_out.dtype)
    for k in range(K):
        idx = flat_indices[:, k]
        out += expert_out[idx] * weights[:, k : k + 1]
    return out


def scatter_tokens(x, flat_indices, weights, num_slots, use_triton=True):
    if use_triton and HAS_TRITON and x.is_cuda:
        return triton_scatter_tokens(x, flat_indices, weights, num_slots)
    return pytorch_scatter_tokens(x, flat_indices, weights, num_slots)


def gather_tokens(expert_out, flat_indices, weights, num_tokens, D, use_triton=True):
    if use_triton and HAS_TRITON and expert_out.is_cuda:
        return triton_gather_tokens(expert_out, flat_indices, weights, num_tokens, D)
    return pytorch_gather_tokens(expert_out, flat_indices, weights, num_tokens, D)
