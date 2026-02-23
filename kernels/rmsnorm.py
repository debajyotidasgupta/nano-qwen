"""Triton-fused RMSNorm kernel with PyTorch fallback.

Fuses the variance computation, normalization, and scaling into a single kernel
to minimize HBM traffic.  Falls back to the pure-PyTorch implementation when
Triton is unavailable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr, W_ptr, Y_ptr,
        stride_x_row, stride_y_row,
        N: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N
        rrms = tl.rsqrt(var + eps)

        y = x * rrms * w
        tl.store(Y_ptr + row * stride_y_row + cols, y, mask=mask)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        DY_ptr, X_ptr, W_ptr, DX_ptr, DW_partial_ptr,
        stride_x_row, stride_dy_row, stride_dx_row,
        N: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N
        rrms = tl.rsqrt(var + eps)

        x_hat = x * rrms
        dy_w = dy * w
        dx = dy_w * rrms - x_hat * (tl.sum(dy_w * x_hat, axis=0) / N)
        dw_partial = dy * x_hat

        tl.store(DX_ptr + row * stride_dx_row + cols, dx, mask=mask)
        tl.store(DW_partial_ptr + row * N + cols, dw_partial, mask=mask)


class _TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        B = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
        x_2d = x.reshape(B, -1).contiguous()
        N = x_2d.shape[1]
        y = torch.empty_like(x_2d)

        BLOCK_N = triton.next_power_of_2(N)
        _rmsnorm_fwd_kernel[(B,)](
            x_2d, weight, y,
            x_2d.stride(0), y.stride(0),
            N=N, eps=eps, BLOCK_N=BLOCK_N,
        )
        ctx.save_for_backward(x_2d, weight)
        ctx.eps = eps
        ctx.orig_shape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight = ctx.saved_tensors
        B, N = x_2d.shape
        dy_2d = dy.reshape(B, N).contiguous()
        dx = torch.empty_like(x_2d)
        dw_partial = torch.empty(B, N, device=x_2d.device, dtype=torch.float32)

        BLOCK_N = triton.next_power_of_2(N)
        _rmsnorm_bwd_kernel[(B,)](
            dy_2d, x_2d, weight, dx, dw_partial,
            x_2d.stride(0), dy_2d.stride(0), dx.stride(0),
            N=N, eps=ctx.eps, BLOCK_N=BLOCK_N,
        )
        dw = dw_partial.sum(dim=0).to(weight.dtype)
        return dx.view(ctx.orig_shape), dw, None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm via Triton. Supports autograd."""
    return _TritonRMSNormFn.apply(x, weight, eps)


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pure PyTorch fallback."""
    x_fp32 = x.float()
    normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight).to(x.dtype)


class FusedRMSNorm(nn.Module):
    """Drop-in replacement for model.norm.RMSNorm with optional Triton acceleration."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, use_triton: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.use_triton = use_triton and HAS_TRITON

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            return triton_rmsnorm(x, self.weight, self.eps)
        return pytorch_rmsnorm(x, self.weight, self.eps)
