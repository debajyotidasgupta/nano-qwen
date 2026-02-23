from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    # --- core dimensions ---
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None
    intermediate_size: int = 11008
    vocab_size: int = 151936
    max_position_embeddings: int = 32768

    # --- normalization ---
    rms_norm_eps: float = 1e-6

    # --- activation ---
    hidden_act: str = "silu"

    # --- attention ---
    attention_bias: bool = False
    attention_dropout: float = 0.0
    qk_norm: bool = True

    # --- RoPE ---
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[dict] = None

    # --- embedding ---
    tie_word_embeddings: bool = False

    # --- MoE ---
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    decoder_sparse_step: int = 1
    router_aux_loss_coef: float = 0.001
    norm_topk_prob: bool = True
    moe_drop_tokens: bool = False
    moe_capacity_factor: float = 1.25

    # --- Vision (ViT) ---
    vision_enabled: bool = False
    vision_hidden_size: int = 1024
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_intermediate_size: int = 4096
    vision_patch_size: int = 14
    vision_image_size: int = 448
    vision_num_channels: int = 3
    vision_deepstack_layers: list[int] = field(default_factory=list)
    vision_merger_spatial_factor: int = 2

    # --- training ---
    gradient_checkpointing: bool = False
    initializer_range: float = 0.02

    # --- special tokens ---
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0 and self.num_experts_per_tok > 0

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def vision_num_patches(self) -> int:
        p = self.vision_image_size // self.vision_patch_size
        return p * p

    def total_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        emb = self.vocab_size * self.hidden_size
        attn_per_layer = (
            self.hidden_size * self.num_attention_heads * self.head_dim  # Q
            + self.hidden_size * self.num_key_value_heads * self.head_dim * 2  # KV
            + self.num_attention_heads * self.head_dim * self.hidden_size  # O
        )
        if self.is_moe:
            ffn_per_layer = self.num_experts * 3 * self.hidden_size * self.moe_intermediate_size
            ffn_per_layer += self.hidden_size * self.num_experts  # router
        else:
            ffn_per_layer = 3 * self.hidden_size * self.intermediate_size
        norm_per_layer = 2 * self.hidden_size
        total = (
            emb
            + self.num_hidden_layers * (attn_per_layer + ffn_per_layer + norm_per_layer)
            + self.hidden_size  # final norm
        )
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.hidden_size
        return total

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str | Path):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_json(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def debug_config() -> ModelConfig:
    """~25M param model for fast local iteration."""
    return ModelConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=688,
        vocab_size=32000,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        rope_theta=10_000.0,
    )


def debug_moe_config() -> ModelConfig:
    """~50M param MoE model for local MoE testing."""
    return ModelConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=688,
        vocab_size=32000,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        rope_theta=10_000.0,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        decoder_sparse_step=1,
    )


def small_config() -> ModelConfig:
    """~1.7B dense model (Qwen3-1.7B-like)."""
    return ModelConfig(
        hidden_size=2048,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=5504,
        vocab_size=151936,
        max_position_embeddings=32768,
        tie_word_embeddings=True,
    )


def medium_config() -> ModelConfig:
    """~8B dense model (Qwen3-8B-like)."""
    return ModelConfig(
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=12288,
        vocab_size=151936,
        max_position_embeddings=131072,
    )


def large_moe_config() -> ModelConfig:
    """~235B-A22B MoE model (Qwen3-235B-A22B)."""
    return ModelConfig(
        hidden_size=4096,
        num_hidden_layers=94,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        intermediate_size=12288,
        vocab_size=151936,
        max_position_embeddings=131072,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=1536,
        decoder_sparse_step=1,
    )
