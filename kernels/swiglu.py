"""Triton-fused SwiGLU activation kernel.

Fuses SiLU(gate) * up into a single kernel, avoiding the intermediate
materialization of the gated output in HBM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        GATE_ptr, UP_ptr, OUT_ptr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        gate = tl.load(GATE_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(UP_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

        # SiLU(gate) = gate * sigmoid(gate)
        silu_gate = gate * tl.sigmoid(gate)
        out = silu_gate * up

        tl.store(OUT_ptr + row * N + cols, out, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        DOUT_ptr, GATE_ptr, UP_ptr,
        DGATE_ptr, DUP_ptr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        dout = tl.load(DOUT_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(GATE_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(UP_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

        sig_gate = tl.sigmoid(gate)
        silu_gate = gate * sig_gate

        # d(SiLU)/d(gate) = sig + gate * sig * (1 - sig)
        dsilu = sig_gate + gate * sig_gate * (1.0 - sig_gate)
        dgate = dout * up * dsilu
        dup = dout * silu_gate

        tl.store(DGATE_ptr + row * N + cols, dgate, mask=mask)
        tl.store(DUP_ptr + row * N + cols, dup, mask=mask)


class _TritonSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        B = gate.shape[0] if gate.dim() == 2 else gate.reshape(-1, gate.shape[-1]).shape[0]
        N = gate.shape[-1]
        gate_2d = gate.reshape(B, N).contiguous()
        up_2d = up.reshape(B, N).contiguous()
        out = torch.empty_like(gate_2d)

        BLOCK = triton.next_power_of_2(N)
        _swiglu_fwd_kernel[(B,)](gate_2d, up_2d, out, N=N, BLOCK=BLOCK)

        ctx.save_for_backward(gate_2d, up_2d)
        ctx.orig_shape = gate.shape
        return out.view(gate.shape)

    @staticmethod
    def backward(ctx, dout):
        gate_2d, up_2d = ctx.saved_tensors
        B, N = gate_2d.shape
        dout_2d = dout.reshape(B, N).contiguous()
        dgate = torch.empty_like(gate_2d)
        dup = torch.empty_like(up_2d)

        BLOCK = triton.next_power_of_2(N)
        _swiglu_bwd_kernel[(B,)](dout_2d, gate_2d, up_2d, dgate, dup, N=N, BLOCK=BLOCK)

        return dgate.view(ctx.orig_shape), dup.view(ctx.orig_shape)


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU: SiLU(gate) * up via Triton."""
    return _TritonSwiGLUFn.apply(gate, up)


def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    if use_triton and HAS_TRITON and gate.is_cuda:
        return triton_swiglu(gate, up)
    return pytorch_swiglu(gate, up)
