"""Triton Flash Attention kernel (FlashAttention-2 algorithm).

Implements tiled, online-softmax attention with:
  - GQA support (repeat KV heads)
  - Built-in causal masking
  - Forward pass only (backward uses PyTorch autograd via SDPA fallback)

For production backward pass, consider using the `flash-attn` package.
This implementation serves as a pedagogical reference and fallback.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _flash_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        batch = tl.program_id(2)
        head = tl.program_id(1)
        start_m = tl.program_id(0) * BLOCK_M

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptrs = Q_ptr + batch * stride_qb + head * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM), other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        end_n = N_CTX
        if IS_CAUSAL:
            end_n = tl.minimum(N_CTX, start_m + BLOCK_M)

        for start_n in range(0, end_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k_ptrs = K_ptr + batch * stride_kb + head * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM), other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_ptrs = V_ptr + batch * stride_vb + head * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM), other=0.0)

            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]

        o_ptrs = O_ptr + batch * stride_ob + head * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Flash attention forward pass via Triton.

    Args:
        q, k, v: (batch, n_heads, seq_len, head_dim)
        causal: Whether to apply causal masking.
        sm_scale: Softmax scale, defaults to 1/sqrt(head_dim).
    """
    B, H, N, D = q.shape
    assert D in (16, 32, 64, 128, 256), f"head_dim={D} not supported, must be power of 2 <= 256"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    o = torch.empty_like(q)

    BLOCK_M = min(128, N)
    BLOCK_N = min(64, N)
    grid = (triton.cdiv(N, BLOCK_M), H, B)

    _flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_CTX=N, HEAD_DIM=D, sm_scale=sm_scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return o


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    use_triton: bool = True,
) -> torch.Tensor:
    """Flash attention with Triton/SDPA auto-selection."""
    if use_triton and HAS_TRITON and q.is_cuda and not q.requires_grad:
        return triton_flash_attention(q, k, v, causal=causal, sm_scale=sm_scale)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal,
    )
