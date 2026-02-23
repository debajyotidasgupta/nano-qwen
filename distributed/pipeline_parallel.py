"""Pipeline Parallelism (PP) utilities.

Splits transformer layers across PP stages and implements scheduling
for overlapping forward/backward micro-batches.  Uses PyTorch's
torch.distributed.pipelining APIs when available.
"""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from torch.distributed.pipelining import (
        SplitPoint,
        pipeline,
        ScheduleGPipe,
        Schedule1F1B,
    )
    HAS_PP_API = True
except ImportError:
    HAS_PP_API = False


def split_model_into_stages(
    model: nn.Module,
    pp_size: int,
) -> List[List[int]]:
    """Compute layer-to-stage assignment for even distribution.

    Returns a list of lists, where stage_layers[i] contains the layer
    indices assigned to PP stage i.
    """
    from model.transformer import Qwen3Model
    assert isinstance(model, Qwen3Model)

    num_layers = len(model.layers)
    layers_per_stage = num_layers // pp_size
    remainder = num_layers % pp_size

    stage_layers = []
    offset = 0
    for i in range(pp_size):
        count = layers_per_stage + (1 if i < remainder else 0)
        stage_layers.append(list(range(offset, offset + count)))
        offset += count

    return stage_layers


def apply_pipeline_parallel(
    model: nn.Module,
    pp_mesh: object,
    pp_size: int,
    n_microbatches: int = 4,
) -> object:
    """Apply pipeline parallelism using torch.distributed.pipelining.

    Returns a pipeline schedule object.
    """
    if not HAS_PP_API:
        raise RuntimeError("Pipeline parallelism API not available")

    from model.transformer import Qwen3Model
    assert isinstance(model, Qwen3Model)

    stage_layers = split_model_into_stages(model, pp_size)

    # Define split points at the boundaries between stages
    split_spec = {}
    offset = 0
    for stage_idx in range(pp_size - 1):
        offset += len(stage_layers[stage_idx])
        split_spec[f"layers.{offset}"] = SplitPoint.BEGINNING

    return split_spec, stage_layers


class ManualPipelineStage(nn.Module):
    """Manual pipeline stage that holds a subset of transformer layers.

    Used when the native pipelining API is unavailable.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        stage_idx: int,
        is_first: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        from model.transformer import Qwen3Model
        assert isinstance(model, Qwen3Model)

        self.stage_idx = stage_idx
        self.is_first = is_first
        self.is_last = is_last

        self.layers = nn.ModuleList([model.layers[i] for i in layer_indices])

        if is_first:
            self.embed = model.embed
        if is_last:
            self.norm = model.norm
            self.lm_head = model.lm_head

    def forward(self, hidden_states, position_ids=None, attention_mask=None):
        if self.is_first and hidden_states.dtype == torch.long:
            hidden_states = self.embed(hidden_states)

        aux_losses = []
        for layer in self.layers:
            hidden_states, aux_loss = layer(hidden_states, position_ids, attention_mask)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        if self.is_last:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, sum(aux_losses) if aux_losses else None

        return hidden_states, sum(aux_losses) if aux_losses else None
