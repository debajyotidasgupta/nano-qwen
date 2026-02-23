"""Transformer block and full language model.

Architecture follows Qwen3 pre-norm design:
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))         # or MoE(RMSNorm(x)) when MoE is enabled

The full model stacks: Embedding -> N blocks -> RMSNorm -> LM Head
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils

from model.config import ModelConfig
from model.norm import RMSNorm
from model.attention import Attention, KVCache
from model.ffn import SwiGLU, build_ffn
from model.embedding import TokenEmbedding


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm attention + FFN/MoE."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._use_moe = self._should_use_moe(config, layer_idx)
        if self._use_moe:
            from model.moe import MoELayer
            self.feed_forward = MoELayer(config)
        else:
            self.feed_forward = build_ffn(config)

    @staticmethod
    def _should_use_moe(config: ModelConfig, layer_idx: int) -> bool:
        if not config.is_moe:
            return False
        if config.decoder_sparse_step <= 0:
            return False
        return (layer_idx % config.decoder_sparse_step) == 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, position_ids, attention_mask, kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        aux_loss = None
        if self._use_moe:
            hidden_states, aux_loss = self.feed_forward(hidden_states)
        else:
            hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states, aux_loss


class Qwen3Model(nn.Module):
    """Full Qwen3 language model: Embedding -> Transformer blocks -> LM Head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed = TokenEmbedding(config)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.embed_tokens.weight

        self.gradient_checkpointing = config.gradient_checkpointing
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        hidden_states = self.embed(input_ids)

        total_aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        moe_layers_count = 0

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, aux_loss = checkpoint_utils.checkpoint(
                    layer, hidden_states, position_ids, attention_mask, kv_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, aux_loss = layer(hidden_states, position_ids, attention_mask, kv_cache)

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
                moe_layers_count += 1

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if moe_layers_count > 0:
                avg_aux = total_aux_loss / moe_layers_count
                loss = loss + self.config.router_aux_loss_coef * avg_aux

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss if moe_layers_count > 0 else None,
            "hidden_states": hidden_states,
        }

    @torch.no_grad()
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
