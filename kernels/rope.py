"""Triton-fused Rotary Position Embedding kernel.

Applies rotary embeddings in-place without materializing full cos/sin tables,
reducing HBM traffic.
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
    def _rope_fwd_kernel(
        Q_ptr, K_ptr,
        COS_ptr, SIN_ptr,
        stride_q_batch, stride_q_head, stride_q_seq, stride_q_dim,
        stride_k_batch, stride_k_head, stride_k_seq, stride_k_dim,
        stride_cs_seq,
        n_q_heads: tl.constexpr,
        n_k_heads: tl.constexpr,
        HALF_DIM: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        batch = tl.program_id(0)
        seq = tl.program_id(1)
        head = tl.program_id(2)

        cols = tl.arange(0, BLOCK_HD)
        mask = cols < HALF_DIM

        cos = tl.load(COS_ptr + seq * stride_cs_seq + cols, mask=mask, other=1.0)
        sin = tl.load(SIN_ptr + seq * stride_cs_seq + cols, mask=mask, other=0.0)

        # Q
        if head < n_q_heads:
            q_off = batch * stride_q_batch + head * stride_q_head + seq * stride_q_seq
            q1 = tl.load(Q_ptr + q_off + cols * stride_q_dim, mask=mask, other=0.0)
            q2 = tl.load(Q_ptr + q_off + (cols + HALF_DIM) * stride_q_dim, mask=mask, other=0.0)
            tl.store(Q_ptr + q_off + cols * stride_q_dim, q1 * cos - q2 * sin, mask=mask)
            tl.store(Q_ptr + q_off + (cols + HALF_DIM) * stride_q_dim, q2 * cos + q1 * sin, mask=mask)

        # K
        if head < n_k_heads:
            k_off = batch * stride_k_batch + head * stride_k_head + seq * stride_k_seq
            k1 = tl.load(K_ptr + k_off + cols * stride_k_dim, mask=mask, other=0.0)
            k2 = tl.load(K_ptr + k_off + (cols + HALF_DIM) * stride_k_dim, mask=mask, other=0.0)
            tl.store(K_ptr + k_off + cols * stride_k_dim, k1 * cos - k2 * sin, mask=mask)
            tl.store(K_ptr + k_off + (cols + HALF_DIM) * stride_k_dim, k2 * cos + k1 * sin, mask=mask)


def triton_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE in-place using Triton. q/k: (B, H, S, D), cos/sin: (S, D//2)."""
    B, Hq, S, D = q.shape
    Hk = k.shape[1]
    HALF = D // 2
    n_heads = max(Hq, Hk)
    BLOCK_HD = triton.next_power_of_2(HALF)

    if cos.dim() == 3:
        cos = cos[:, 0, :]  # collapse batch dim for kernel
        sin = sin[:, 0, :]

    grid = (B, S, n_heads)
    _rope_fwd_kernel[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0),
        n_q_heads=Hq, n_k_heads=Hk,
        HALF_DIM=HALF, BLOCK_HD=BLOCK_HD,
    )
    return q, k


def pytorch_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch RoPE application."""
    from model.rope import apply_rotary_pos_emb
    return apply_rotary_pos_emb(q, k, cos, sin)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_triton and HAS_TRITON and q.is_cuda:
        return triton_apply_rope(q, k, cos, sin)
    return pytorch_apply_rope(q, k, cos, sin)
