"""Triton-fused cross-entropy loss kernel.

Computes cross-entropy without materializing the full (B*T, V) logit tensor
into a softmax distribution.  Uses online log-sum-exp for numerical stability
and chunks the vocabulary to keep memory bounded.
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
    def _cross_entropy_fwd_kernel(
        LOGITS_ptr, LABELS_ptr, LOSS_ptr, LSE_ptr,
        stride_logit_row,
        V: tl.constexpr,
        IGNORE_IDX: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        label = tl.load(LABELS_ptr + row)

        if label == IGNORE_IDX:
            tl.store(LOSS_ptr + row, 0.0)
            tl.store(LSE_ptr + row, 0.0)
            return

        # Compute log-sum-exp over vocab in chunks
        m = tl.full([], float("-inf"), dtype=tl.float32)
        s = tl.zeros([], dtype=tl.float32)
        target_logit = tl.zeros([], dtype=tl.float32)

        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < V
            logits = tl.load(LOGITS_ptr + row * stride_logit_row + cols, mask=mask, other=float("-inf")).to(tl.float32)

            chunk_max = tl.max(logits, axis=0)
            new_m = tl.maximum(m, chunk_max)
            s = s * tl.exp(m - new_m) + tl.sum(tl.exp(logits - new_m), axis=0)
            m = new_m

            target_mask = cols == label
            target_logit = target_logit + tl.sum(tl.where(target_mask, logits, 0.0), axis=0)

        lse = m + tl.log(s)
        loss = lse - target_logit

        tl.store(LOSS_ptr + row, loss)
        tl.store(LSE_ptr + row, lse)


class _TritonCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100):
        BT, V = logits.shape
        losses = torch.empty(BT, device=logits.device, dtype=torch.float32)
        lse = torch.empty(BT, device=logits.device, dtype=torch.float32)

        BLOCK_V = min(triton.next_power_of_2(V), 32768)
        _cross_entropy_fwd_kernel[(BT,)](
            logits, labels, losses, lse,
            logits.stride(0),
            V=V, IGNORE_IDX=ignore_index, BLOCK_V=BLOCK_V,
        )

        valid = (labels != ignore_index).sum()
        loss = losses.sum() / valid.clamp(min=1)

        ctx.save_for_backward(logits, labels, lse)
        ctx.ignore_index = ignore_index
        ctx.valid_count = valid
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, lse = ctx.saved_tensors
        BT, V = logits.shape

        probs = torch.softmax(logits.float(), dim=-1)
        one_hot = F.one_hot(labels.clamp(min=0), V).float()
        grad = (probs - one_hot) * grad_output / ctx.valid_count.clamp(min=1)

        mask = (labels == ctx.ignore_index).unsqueeze(-1)
        grad = grad.masked_fill(mask, 0.0)

        return grad.to(logits.dtype), None, None


def triton_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Memory-efficient cross-entropy via Triton."""
    return _TritonCrossEntropyFn.apply(logits, labels, ignore_index)


def fused_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    use_triton: bool = True,
) -> torch.Tensor:
    if use_triton and HAS_TRITON and logits.is_cuda:
        return triton_cross_entropy(logits, labels, ignore_index)
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)
