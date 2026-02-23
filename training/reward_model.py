"""Reward model for RLHF — same architecture as policy model with a scalar value head.

Supports:
  - Bradley-Terry preference modeling
  - Verifiable rewards (math correctness, code execution)
  - Training on human preference data
"""

from __future__ import annotations

from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.transformer import Qwen3Model


class RewardModel(nn.Module):
    """Reward model: LLM backbone + scalar value head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = Qwen3Model(config)
        # Replace LM head with scalar reward head
        self.backbone.lm_head = nn.Identity()
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scalar reward for each sequence.

        Args:
            input_ids: (B, S)
            attention_mask: (B, S) optional

        Returns:
            rewards: (B,) scalar reward per sequence
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["hidden_states"]  # (B, S, D)

        # Pool to get sequence-level representation (last non-pad token)
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=-1) - 1
            batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            pooled = hidden_states[batch_idx, seq_lens.long()]
        else:
            pooled = hidden_states[:, -1, :]

        rewards = self.value_head(pooled).squeeze(-1)  # (B,)
        return rewards

    def compute_preference_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Bradley-Terry preference loss.

        Loss = -log(sigmoid(r_chosen - r_rejected))
        """
        r_chosen = self.forward(chosen_ids, chosen_mask)
        r_rejected = self.forward(rejected_ids, rejected_mask)

        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        accuracy = (r_chosen > r_rejected).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "chosen_reward": r_chosen.mean(),
            "rejected_reward": r_rejected.mean(),
        }


class VerifiableReward:
    """Compute verifiable rewards for math and code tasks."""

    def __init__(self):
        self.reward_fns = {
            "math": self._math_reward,
            "code": self._code_reward,
        }

    def compute(
        self,
        responses: List[str],
        ground_truths: List[str],
        task_type: str = "math",
    ) -> torch.Tensor:
        """Compute binary rewards for verifiable tasks.

        Returns: (N,) tensor of rewards in {0, 1}.
        """
        fn = self.reward_fns.get(task_type, self._default_reward)
        rewards = [fn(resp, gt) for resp, gt in zip(responses, ground_truths)]
        return torch.tensor(rewards, dtype=torch.float32)

    @staticmethod
    def _math_reward(response: str, ground_truth: str) -> float:
        """Check if the final numerical answer matches."""
        def extract_answer(text: str) -> str:
            # Look for \\boxed{...} or last number
            import re
            boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
            if boxed:
                return boxed[-1].strip()
            numbers = re.findall(r'-?\d+\.?\d*', text)
            return numbers[-1] if numbers else ""

        pred = extract_answer(response)
        gold = extract_answer(ground_truth)
        try:
            return 1.0 if abs(float(pred) - float(gold)) < 1e-6 else 0.0
        except (ValueError, ZeroDivisionError):
            return 1.0 if pred == gold else 0.0

    @staticmethod
    def _code_reward(response: str, ground_truth: str) -> float:
        """Placeholder for code execution reward.
        In production, this would execute the code in a sandbox.
        """
        return 1.0 if ground_truth.strip() in response else 0.0

    @staticmethod
    def _default_reward(response: str, ground_truth: str) -> float:
        return 1.0 if ground_truth.strip() in response else 0.0
