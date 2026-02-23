"""Direct Preference Optimization (DPO) — online and offline variants.

Implements:
  - Standard offline DPO (Rafailov et al., 2023)
  - Online DPO with on-policy generation
  - Combined verifiable + model-based reward signals (Qwen3 Stage 4)
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
from distributed.comm import is_main_process


class DPOTrainer:
    """Direct Preference Optimization trainer."""

    def __init__(
        self,
        policy_model: Qwen3Model,
        ref_model: Qwen3Model,
        config: dict,
        device: torch.device,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.device = device

        dpo_cfg = config.get("dpo", {})
        self.beta = dpo_cfg.get("beta", 0.1)
        self.label_smoothing = dpo_cfg.get("label_smoothing", 0.0)
        self.reference_free = dpo_cfg.get("reference_free", False)

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

        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False

    def _get_per_token_log_probs(
        self,
        model: Qwen3Model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for the label tokens."""
        outputs = model(input_ids=input_ids)
        logits = outputs["logits"][:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)

        label_tokens = labels[:, 1:]
        per_token = log_probs.gather(2, label_tokens.unsqueeze(-1)).squeeze(-1)

        # Mask padding
        mask = (label_tokens != -100).float()
        return (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    def dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute DPO loss.

        L = -log(sigma(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))))
        """
        # Policy log probs
        pi_chosen = self._get_per_token_log_probs(self.policy, chosen_ids, chosen_labels)
        pi_rejected = self._get_per_token_log_probs(self.policy, rejected_ids, rejected_labels)

        # Reference log probs
        with torch.no_grad():
            if self.reference_free:
                ref_chosen = torch.zeros_like(pi_chosen)
                ref_rejected = torch.zeros_like(pi_rejected)
            else:
                ref_chosen = self._get_per_token_log_probs(self.ref, chosen_ids, chosen_labels)
                ref_rejected = self._get_per_token_log_probs(self.ref, rejected_ids, rejected_labels)

        # Log ratios
        chosen_logratios = pi_chosen - ref_chosen
        rejected_logratios = pi_rejected - ref_rejected
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # Loss with optional label smoothing
        if self.label_smoothing > 0:
            loss = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # Accuracy
        accuracy = (logits > 0).float().mean()

        # Implicit rewards
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }

        return loss, metrics

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> dict:
        """Single DPO training step."""
        self.optimizer.zero_grad()

        loss, metrics = self.dpo_loss(
            chosen_ids.to(self.device),
            rejected_ids.to(self.device),
            chosen_labels.to(self.device),
            rejected_labels.to(self.device),
        )

        loss.backward()

        if self.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            metrics["grad_norm"] = grad_norm.item()

        self.optimizer.step()
        self.scheduler.step()

        return metrics


class OnlineDPO(DPOTrainer):
    """Online DPO: generate preference pairs from the current policy.

    Uses a reward model (or verifiable rewards) to rank on-policy generations,
    then applies DPO loss on the resulting preference pairs.
    """

    def __init__(
        self,
        policy_model: Qwen3Model,
        ref_model: Qwen3Model,
        reward_fn: Callable,
        tokenizer,
        config: dict,
        device: torch.device,
    ):
        super().__init__(policy_model, ref_model, config, device)
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.num_candidates = config.get("grpo", {}).get("group_size", 4)

    @torch.no_grad()
    def generate_preference_pair(
        self,
        prompt_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate responses and select best/worst as chosen/rejected."""
        self.policy.eval()
        prompt = prompt_ids.expand(self.num_candidates, -1).to(self.device)

        all_ids = prompt.clone()
        max_gen = 512

        for _ in range(max_gen):
            outputs = self.policy(input_ids=all_ids)
            logits = outputs["logits"][:, -1, :]
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            all_ids = torch.cat([all_ids, next_token], dim=1)

        self.policy.train()

        # Score and select best/worst
        decoded = [self.tokenizer.decode(ids.tolist()) for ids in all_ids]
        rewards = self.reward_fn(decoded)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        best_idx = rewards.argmax()
        worst_idx = rewards.argmin()

        return all_ids[best_idx:best_idx + 1], all_ids[worst_idx:worst_idx + 1]

    def online_train_step(self, prompt_ids: torch.Tensor) -> dict:
        """Generate pair online, then do DPO update."""
        chosen, rejected = self.generate_preference_pair(prompt_ids)
        return self.train_step(chosen, rejected, chosen, rejected)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_dpo(
    model_config_path: str,
    training_config_path: str,
    resume_from: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """End-to-end DPO pipeline: load config, build model, load HF preference data, train."""
    import yaml
    import copy
    from model.config import ModelConfig
    from distributed.comm import (
        init_distributed, get_local_rank, get_world_size, is_main_process,
    )

    rank, world_size = init_distributed()
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig.from_yaml(model_config_path)
    with open(training_config_path) as f:
        training_config = yaml.safe_load(f)

    policy = Qwen3Model(model_config).to(device)
    if is_main_process():
        p = policy.count_parameters()
        print(f"DPO | params: {p['total']:,}")

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

    # Load DPO preference pairs from HuggingFace
    from data.tokenizer import Tokenizer
    tokenizer = Tokenizer(backend="tiktoken")
    batch_cfg = training_config.get("batch", {})
    max_seq_len = batch_cfg.get("max_seq_len", 2048)

    try:
        from data.hf_datasets import load_dpo_dataset
        dataset = load_dpo_dataset(tokenizer, max_seq_len, max_samples=max_samples)
    except Exception as exc:
        if is_main_process():
            print(f"HF dataset load failed ({exc}), using dummy pairs")
        from data.hf_datasets import DPODataset
        dummy_pairs = [
            {"chosen": "User: Hi\nAssistant: Hello! How can I help?",
             "rejected": "User: Hi\nAssistant: What do you want?"},
        ] * 100
        dataset = DPODataset(dummy_pairs, tokenizer, max_seq_len)

    if is_main_process():
        print(f"Loaded {len(dataset)} DPO preference pairs")

    from data.sampler import DistributedSampler as DSampler
    sampler = DSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_cfg.get("micro_batch_size", 2),
        sampler=sampler,
        num_workers=training_config.get("data", {}).get("num_workers", 4),
        pin_memory=True,
    )

    trainer = DPOTrainer(policy, ref, training_config, device)
    logger = TrainingLogger(log_interval=1)

    total_steps = training_config.get("scheduler", {}).get("total_steps", 2000)
    data_iter = iter(dataloader)
    for step in range(1, total_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        metrics = trainer.train_step(
            batch["chosen_ids"], batch["rejected_ids"],
            batch["chosen_labels"], batch["rejected_labels"],
        )

        lr = trainer.scheduler.get_last_lr()[0]
        logger.log_step(
            step=step,
            loss=metrics["loss"],
            lr=lr,
            extra={f"dpo/{k}": v for k, v in metrics.items()},
        )

    logger.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DPO training for Qwen3")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_dpo(args.model_config, args.training_config, args.resume, args.max_samples)
