"""Token embedding layer with optional weight tying to the LM head."""

import torch
import torch.nn as nn

from model.config import ModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.hidden_size = config.hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
