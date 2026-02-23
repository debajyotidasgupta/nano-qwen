"""Full multimodal model — ViT encoder + DeepStack + Merger + LLM backbone.

Combines the vision pipeline with the Qwen3 language model to handle
mixed image-text input sequences.
"""

from __future__ import annotations

import math
from typing import Optional, List, Dict

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.transformer import Qwen3Model
from model.vision.vit import VisionTransformer
from model.vision.deepstack import DeepStack
from model.vision.merger import VisionMerger


class Qwen3VLModel(nn.Module):
    """Multimodal Qwen3 model with vision and language capabilities."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        assert config.vision_enabled, "vision_enabled must be True for Qwen3VLModel"

        self.vision_encoder = VisionTransformer(config)

        num_deepstack_levels = len(config.vision_deepstack_layers) + 1  # intermediates + final
        self.deepstack = DeepStack(
            vision_hidden_size=config.vision_hidden_size,
            llm_hidden_size=config.hidden_size,
            num_levels=num_deepstack_levels,
        )

        self.merger = VisionMerger(
            vision_hidden_size=config.hidden_size,  # after deepstack, it's in LLM dim
            llm_hidden_size=config.hidden_size,
            spatial_factor=config.vision_merger_spatial_factor,
        )

        self.language_model = Qwen3Model(config)

    def encode_images(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Encode images through ViT + DeepStack + Merger.

        Args:
            pixel_values: (B, C, H, W) or (num_images, C, H, W)

        Returns:
            (num_images, num_merged_tokens, hidden_size) visual tokens
        """
        vit_output = self.vision_encoder(pixel_values)

        fused = self.deepstack(
            vit_output["intermediate_states"],
            vit_output["last_hidden_state"],
        )

        _, H_W, _ = fused.shape
        grid_h = grid_w = int(math.sqrt(H_W))
        merged = self.merger(fused, grid_h, grid_w)

        return merged

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: (B, S) token IDs
            pixel_values: (num_images, C, H, W) images
            image_token_mask: (B, S) bool mask where True = image placeholder token
            position_ids: (B, S) position IDs
            attention_mask: optional attention mask
            labels: (B, S) for language modeling loss

        Returns:
            dict with logits, loss, etc.
        """
        # Get text embeddings
        hidden_states = self.language_model.embed(input_ids)

        # If images are provided, encode and insert visual tokens
        if pixel_values is not None and image_token_mask is not None:
            visual_tokens = self.encode_images(pixel_values)

            # Flatten visual tokens across all images
            visual_tokens_flat = visual_tokens.reshape(-1, visual_tokens.shape[-1])

            # Replace image placeholder positions with visual tokens
            B, S, D = hidden_states.shape
            img_positions = image_token_mask.nonzero(as_tuple=False)

            num_visual = min(visual_tokens_flat.shape[0], img_positions.shape[0])
            for i in range(num_visual):
                b_idx, s_idx = img_positions[i]
                hidden_states[b_idx, s_idx] = visual_tokens_flat[i]

        # Pass through transformer layers
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        moe_count = 0

        for layer in self.language_model.layers:
            hidden_states, aux_loss = layer(hidden_states, position_ids, attention_mask)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
                moe_count += 1

        hidden_states = self.language_model.norm(hidden_states)
        logits = self.language_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if moe_count > 0:
                avg_aux = total_aux_loss / moe_count
                loss = loss + self.config.router_aux_loss_coef * avg_aux

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss if moe_count > 0 else None,
            "hidden_states": hidden_states,
        }
