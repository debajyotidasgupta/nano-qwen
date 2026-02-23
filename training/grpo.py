"""Group Relative Policy Optimization (GRPO).

Implements the RL algorithm used in Qwen3's Reasoning RL (Stage 2) and
General RL (Stage 4) phases:

  1. For each query, sample G responses from the policy model
  2. Score responses with reward (verifiable or model-based)
  3. Compute group-relative advantage: A_i = (R_i - mean(R)) / std(R)
  4. Update policy with clipped surrogate objective + KL penalty

Reference: DeepSeek-R1 (arXiv:2501.12948)
"""

from __future__ import annotations

import math
from typing import Optional, List, Dict, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import Qwen3Model
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.logger import TrainingLogger
from training.checkpoint import CheckpointManager
from distributed.comm import is_main_process, get_rank


class GRPO:
    """Group Relative Policy Optimization trainer."""

    def __init__(
        self,
        policy_model: Qwen3Model,
        ref_model: Optional[Qwen3Model],
        tokenizer,
        config: dict,
        device: torch.device,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.device = device

        grpo_cfg = config.get("grpo", {})
        self.group_size = grpo_cfg.get("group_size", 16)
        self.clip_ratio = grpo_cfg.get("clip_ratio", 0.2)
        self.kl_coef = grpo_cfg.get("kl_coef", 0.01)
        self.temperature = grpo_cfg.get("temperature", 0.7)
        self.max_gen_len = grpo_cfg.get("max_gen_len", 2048)

        opt_cfg = config.get("optimizer", {})
        self.optimizer = build_optimizer(
            policy_model,
            lr=opt_cfg.get("lr", 5e-7),
            weight_decay=opt_cfg.get("weight_decay", 0.1),
        )

        sched_cfg = config.get("scheduler", {})
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_type=sched_cfg.get("type", "cosine"),
            warmup_steps=sched_cfg.get("warmup_steps", 50),
            total_steps=sched_cfg.get("total_steps", 2000),
        )

        self.grad_clip = opt_cfg.get("grad_clip", 1.0)

        if self.ref is not None:
            self.ref.eval()
            for p in self.ref.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def generate_responses(
        self,
        prompt_ids: torch.Tensor,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate G responses per prompt using nucleus sampling.

        Args:
            prompt_ids: (1, prompt_len) tokenized prompt
            num_samples: number of responses to generate

        Returns:
            response_ids: (G, max_len) generated token IDs
            response_log_probs: (G, max_len) log probabilities under policy
        """
        self.policy.eval()
        prompt = prompt_ids.expand(num_samples, -1).to(self.device)

        all_ids = prompt.clone()
        all_log_probs = []

        for _ in range(self.max_gen_len):
            outputs = self.policy(input_ids=all_ids)
            logits = outputs["logits"][:, -1, :] / self.temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            log_prob = F.log_softmax(logits, dim=-1).gather(1, next_token)

            all_ids = torch.cat([all_ids, next_token], dim=1)
            all_log_probs.append(log_prob)

            # Stop if all sequences have generated EOS
            if hasattr(self.tokenizer, 'eos_id'):
                if (next_token == self.tokenizer.eos_id).all():
                    break

        self.policy.train()
        response_log_probs = torch.cat(all_log_probs, dim=1)
        return all_ids, response_log_probs

    def compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute group-relative advantages.

        A_i = (R_i - mean(R)) / std(R)

        Args:
            rewards: (G,) rewards for each response in the group

        Returns:
            advantages: (G,) normalized advantages
        """
        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        return (rewards - mean_r) / std_r

    def compute_kl_penalty(
        self,
        response_ids: torch.Tensor,
        policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference model.

        KL(pi || ref) estimated per-token.
        """
        if self.ref is None:
            return torch.zeros(response_ids.shape[0], device=self.device)

        with torch.no_grad():
            ref_outputs = self.ref(input_ids=response_ids)
            ref_logits = ref_outputs["logits"][:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            tokens = response_ids[:, 1:]
            ref_token_log_probs = ref_log_probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)

        # policy_log_probs aligned to generated tokens
        gen_len = min(policy_log_probs.shape[1], ref_token_log_probs.shape[1])
        kl = (policy_log_probs[:, :gen_len] - ref_token_log_probs[:, :gen_len]).mean(dim=1)
        return kl

    def policy_loss(
        self,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute clipped surrogate policy gradient loss.

        L = -min(r * A, clip(r, 1-eps, 1+eps) * A) + beta * KL
        """
        outputs = self.policy(input_ids=response_ids)
        logits = outputs["logits"][:, :-1, :]
        new_log_probs = F.log_softmax(logits, dim=-1)

        tokens = response_ids[:, 1:]
        gen_len = min(new_log_probs.shape[1], tokens.shape[1], old_log_probs.shape[1])

        new_token_log_probs = new_log_probs[:, :gen_len].gather(
            2, tokens[:, :gen_len].unsqueeze(-1)
        ).squeeze(-1)
        old_token_log_probs = old_log_probs[:, :gen_len]

        # Per-token importance ratio
        ratio = torch.exp(new_token_log_probs - old_token_log_probs)
        adv = advantages.unsqueeze(1).expand_as(ratio)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty
        kl = self.compute_kl_penalty(response_ids, new_token_log_probs)
        total_loss = policy_loss + self.kl_coef * kl.mean()

        metrics = {
            "policy_loss": policy_loss.item(),
            "kl": kl.mean().item(),
            "ratio_mean": ratio.mean().item(),
            "advantages_mean": advantages.mean().item(),
        }

        return total_loss, metrics

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        reward_fn: Callable,
        ground_truth: Optional[str] = None,
    ) -> dict:
        """Single GRPO training step for one query.

        1. Generate G responses
        2. Score with reward function
        3. Compute advantages
        4. Update policy
        """
        # Generate
        response_ids, old_log_probs = self.generate_responses(prompt_ids, self.group_size)

        # Decode and compute rewards
        decoded = [
            self.tokenizer.decode(ids.tolist()) for ids in response_ids
        ]
        rewards = reward_fn(decoded, ground_truth)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = rewards.to(self.device)

        # Advantages
        advantages = self.compute_advantages(rewards)

        # Policy update
        self.optimizer.zero_grad()
        loss, metrics = self.policy_loss(response_ids, old_log_probs, advantages)
        loss.backward()

        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        metrics.update({
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "total_loss": loss.item(),
        })

        return metrics

    def train(
        self,
        queries: List[dict],
        reward_fn: Callable,
        total_steps: int,
    ):
        """Full GRPO training loop over a set of queries."""
        logger = TrainingLogger(log_interval=1)

        for step in range(total_steps):
            query = queries[step % len(queries)]
            prompt_ids = torch.tensor(
                self.tokenizer.encode(query["prompt"]),
                dtype=torch.long,
            ).unsqueeze(0)

            metrics = self.train_step(
                prompt_ids,
                reward_fn,
                ground_truth=query.get("answer"),
            )

            lr = self.scheduler.get_last_lr()[0]
            logger.log_step(
                step=step + 1,
                loss=metrics["total_loss"],
                lr=lr,
                extra={f"rl/{k}": v for k, v in metrics.items()},
            )

        logger.finish()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _math_reward_fn(responses: list[str], ground_truth: str | None) -> list[float]:
    """Simple verifiable reward: check if the final number matches."""
    import re

    def _extract(text: str) -> str:
        boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed[-1].strip()
        hashes = re.findall(r'####\s*(.+)', text)
        if hashes:
            return hashes[-1].strip()
        nums = re.findall(r'-?\d+\.?\d*', text)
        return nums[-1] if nums else ""

    if ground_truth is None:
        return [0.0] * len(responses)
    gt = _extract(ground_truth)
    rewards = []
    for resp in responses:
        pred = _extract(resp)
        try:
            rewards.append(1.0 if abs(float(pred) - float(gt)) < 1e-6 else 0.0)
        except (ValueError, ZeroDivisionError):
            rewards.append(1.0 if pred == gt else 0.0)
    return rewards


def run_grpo(
    model_config_path: str,
    training_config_path: str,
    resume_from: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """End-to-end GRPO pipeline: load config, build model, load HF data, train."""
    import yaml
    from model.config import ModelConfig
    from model.transformer import Qwen3Model
    from distributed.comm import (
        init_distributed, get_local_rank, get_world_size, is_main_process,
    )

    rank, world_size = init_distributed()
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig.from_yaml(model_config_path)
    with open(training_config_path) as f:
        training_config = yaml.safe_load(f)

    # Policy model
    policy = Qwen3Model(model_config).to(device)
    if is_main_process():
        p = policy.count_parameters()
        print(f"GRPO | params: {p['total']:,}")

    # Reference model (frozen copy)
    import copy
    ref = copy.deepcopy(policy)

    if world_size > 1:
        from distributed.parallel import ParallelConfig, ParallelismManager
        p_cfg = training_config.get("distributed", {})
        pm = ParallelismManager(ParallelConfig(
            dp_size=p_cfg.get("dp_size", -1),
            tp_size=p_cfg.get("tp_size", 1),
            pp_size=p_cfg.get("pp_size", 1),
            fsdp_enabled=p_cfg.get("fsdp_enabled", True),
            dtype=training_config.get("precision", {}).get("dtype", "bfloat16"),
        ))
        policy = pm.apply_parallelism(policy)

    # Tokenizer
    from data.tokenizer import Tokenizer
    tokenizer = Tokenizer(backend="tiktoken")

    # Load GRPO queries from HuggingFace
    try:
        from data.hf_datasets import load_grpo_dataset
        queries = load_grpo_dataset(max_samples=max_samples)
    except Exception as exc:
        if is_main_process():
            print(f"HF dataset load failed ({exc}), using dummy queries")
        queries = [
            {"prompt": "What is 15 + 27?", "answer": "42"},
            {"prompt": "What is 100 - 37?", "answer": "63"},
        ] * 50

    if is_main_process():
        print(f"Loaded {len(queries)} GRPO queries")

    trainer = GRPO(
        policy_model=policy,
        ref_model=ref,
        tokenizer=tokenizer,
        config=training_config,
        device=device,
    )

    total_steps = training_config.get("scheduler", {}).get("total_steps", 2000)
    trainer.train(queries, _math_reward_fn, total_steps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO training for Qwen3")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_grpo(args.model_config, args.training_config, args.resume, args.max_samples)
