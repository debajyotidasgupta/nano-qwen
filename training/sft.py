"""Supervised Fine-Tuning (SFT) — Long-CoT Cold Start and Thinking Mode Fusion.

Implements the Qwen3 post-training stages:
  Stage 1 (Long-CoT Cold Start):   Train on verified chain-of-thought solutions
  Stage 3 (Thinking Mode Fusion):  Mixed training on thinking + non-thinking data

Chat template format:
  <|im_start|>system\nYou are a helpful assistant.<|im_end|>
  <|im_start|>user\n{question}<|im_end|>
  <|im_start|>assistant\n<think>{reasoning}</think>\n{answer}<|im_end|>
"""

from __future__ import annotations

import os
import time
from typing import Optional, Dict, List, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.config import ModelConfig
from model.transformer import Qwen3Model
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.checkpoint import CheckpointManager
from training.logger import TrainingLogger
from distributed.comm import (
    init_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
)


CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>"
)

THINK_TEMPLATE = "<think>\n{reasoning}\n</think>\n{answer}"
NO_THINK_TEMPLATE = "<think>\n</think>\n{answer}"


def format_conversation(
    system: str,
    user: str,
    assistant: str,
    reasoning: Optional[str] = None,
    enable_thinking: bool = True,
) -> str:
    """Format a single conversation turn with optional thinking."""
    if enable_thinking and reasoning:
        assistant_text = THINK_TEMPLATE.format(reasoning=reasoning, answer=assistant)
    elif enable_thinking:
        assistant_text = NO_THINK_TEMPLATE.format(answer=assistant)
    else:
        assistant_text = assistant

    return CHAT_TEMPLATE.format(
        system=system,
        user=user,
        assistant=assistant_text,
    )


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning with chat templates.

    Expects samples as dicts with:
        - "system": str (system prompt)
        - "conversations": list of {"user": str, "assistant": str, "reasoning": str|None}
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_seq_len: int = 8192,
        mask_input: bool = True,
        thinking_enabled: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_input = mask_input
        self.thinking_enabled = thinking_enabled
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        system = sample.get("system", self.system_prompt)
        conversations = sample.get("conversations", [])

        full_text = ""
        assistant_spans = []

        for turn in conversations:
            user = turn["user"]
            assistant = turn["assistant"]
            reasoning = turn.get("reasoning", None)

            formatted = format_conversation(
                system=system,
                user=user,
                assistant=assistant,
                reasoning=reasoning,
                enable_thinking=self.thinking_enabled,
            )

            # Track where the assistant response starts (for masking)
            prefix = CHAT_TEMPLATE.split("{assistant}")[0].format(system=system, user=user)
            start = len(full_text) + len(prefix)
            full_text += formatted + "\n"
            end = len(full_text) - 1
            assistant_spans.append((start, end))

        # Tokenize
        token_ids = self.tokenizer.encode(full_text, allowed_special="all")
        token_ids = token_ids[: self.max_seq_len + 1]

        if len(token_ids) < self.max_seq_len + 1:
            token_ids = token_ids + [0] * (self.max_seq_len + 1 - len(token_ids))

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        # Mask input tokens in labels (only train on assistant responses)
        if self.mask_input:
            labels[:] = -100
            for start_char, end_char in assistant_spans:
                # Approximate token positions from character positions
                ratio = len(token_ids) / max(len(full_text), 1)
                start_tok = max(0, int(start_char * ratio))
                end_tok = min(len(labels), int(end_char * ratio))
                labels[start_tok:end_tok] = torch.tensor(
                    token_ids[start_tok + 1 : end_tok + 1], dtype=torch.long
                )

        return {"input_ids": input_ids, "labels": labels}


class SFTTrainer:
    """Trainer for supervised fine-tuning stages."""

    def __init__(
        self,
        model: Qwen3Model,
        training_config: dict,
        device: torch.device,
    ):
        self.model = model
        self.cfg = training_config
        self.device = device
        self.rank = get_rank()

        opt_cfg = self.cfg.get("optimizer", {})
        self.optimizer = build_optimizer(
            model,
            lr=opt_cfg.get("lr", 1e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.1),
            beta1=opt_cfg.get("beta1", 0.9),
            beta2=opt_cfg.get("beta2", 0.95),
        )

        sched_cfg = self.cfg.get("scheduler", {})
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_type=sched_cfg.get("type", "cosine"),
            warmup_steps=sched_cfg.get("warmup_steps", 100),
            total_steps=sched_cfg.get("total_steps", 5000),
        )

        ckpt_cfg = self.cfg.get("checkpoint", {})
        self.ckpt_mgr = CheckpointManager(
            save_dir=ckpt_cfg.get("save_dir", "checkpoints/sft"),
            save_interval=ckpt_cfg.get("save_interval", 500),
        )

        log_cfg = self.cfg.get("logging", {})
        self.logger = TrainingLogger(
            wandb_project=log_cfg.get("wandb_project"),
            wandb_enabled=log_cfg.get("wandb_enabled", False),
            tensorboard_dir=log_cfg.get("tensorboard_dir"),
            tensorboard_enabled=log_cfg.get("tensorboard_enabled", True),
            log_interval=log_cfg.get("log_interval", 10),
        )

        self.grad_clip = opt_cfg.get("grad_clip", 1.0)
        self.grad_accum_steps = self.cfg.get("batch", {}).get("gradient_accumulation_steps", 1)

    def train(
        self,
        dataloader: DataLoader,
        total_steps: int,
        start_step: int = 0,
    ):
        """SFT training loop."""
        self.model.train()
        dtype = getattr(torch, self.cfg.get("precision", {}).get("dtype", "bfloat16"))

        step = start_step
        data_iter = iter(dataloader)

        while step < total_steps:
            self.optimizer.zero_grad()
            accum_loss = 0.0

            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"] / self.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()

            if self.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            else:
                grad_norm = None

            self.optimizer.step()
            self.scheduler.step()
            step += 1

            lr = self.scheduler.get_last_lr()[0]
            self.logger.log_step(
                step=step, loss=accum_loss, lr=lr,
                grad_norm=grad_norm.item() if grad_norm is not None else None,
            )

            if self.ckpt_mgr.should_save(step):
                self.ckpt_mgr.save(step, self.model, self.optimizer, self.scheduler)

        self.logger.finish()


class ThinkingModeFusion:
    """Stage 3: Thinking Mode Fusion — mix thinking and non-thinking samples.

    Combines chain-of-thought (thinking) data with standard (non-thinking) data,
    where non-thinking samples include an empty <think></think> block for
    format consistency.  This enables the model to serve as both a reasoning
    model and a standard chat model.
    """

    def __init__(
        self,
        thinking_ratio: float = 0.5,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.thinking_ratio = thinking_ratio
        self.system_prompt = system_prompt

    def prepare_dataset(
        self,
        thinking_samples: List[Dict],
        non_thinking_samples: List[Dict],
    ) -> List[Dict]:
        """Merge thinking and non-thinking samples with appropriate formatting.

        Thinking samples: have "reasoning" field in conversations.
        Non-thinking samples: assistant responses wrapped in empty <think></think>.
        """
        # Ensure non-thinking samples have empty thinking blocks
        processed_non_thinking = []
        for sample in non_thinking_samples:
            new_conversations = []
            for turn in sample.get("conversations", []):
                new_turn = dict(turn)
                if "reasoning" not in new_turn:
                    new_turn["reasoning"] = ""  # empty thinking block
                new_conversations.append(new_turn)
            processed_non_thinking.append({
                **sample,
                "conversations": new_conversations,
            })

        # Mix according to ratio
        import random
        combined = []
        n_thinking = int(len(thinking_samples) * self.thinking_ratio / (1 - self.thinking_ratio))
        n_thinking = min(n_thinking, len(thinking_samples))

        combined.extend(thinking_samples[:n_thinking])
        combined.extend(processed_non_thinking)
        random.shuffle(combined)

        return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_sft(
    model_config_path: str,
    training_config_path: str,
    variant: str = "cold_start",
    resume_from: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """End-to-end SFT pipeline: load config, build model, load HF data, train."""
    rank, world_size = init_distributed()
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig.from_yaml(model_config_path)
    with open(training_config_path) as f:
        training_config = yaml.safe_load(f)

    model = Qwen3Model(model_config).to(device)
    if is_main_process():
        p = model.count_parameters()
        print(f"SFT | params: {p['total']:,} | variant: {variant}")

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
        model = pm.apply_parallelism(model)

    # Load dataset from HuggingFace
    batch_cfg = training_config.get("batch", {})
    sft_cfg = training_config.get("sft", {})
    max_seq_len = batch_cfg.get("max_seq_len", 8192)

    try:
        from data.hf_datasets import load_sft_dataset
        from data.tokenizer import Tokenizer
        tokenizer = Tokenizer(backend="tiktoken")
        dataset = load_sft_dataset(
            variant=variant,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
            thinking_enabled=sft_cfg.get("thinking_enabled", True),
        )
    except Exception as exc:
        if is_main_process():
            print(f"HF dataset load failed ({exc}), falling back to dummy data")
        dummy = [{"conversations": [{"user": "What is 2+2?", "assistant": "4", "reasoning": "2+2=4"}]}]
        from data.tokenizer import Tokenizer
        tokenizer = Tokenizer(backend="tiktoken")
        dataset = SFTDataset(dummy * 100, tokenizer, max_seq_len)

    from data.sampler import DistributedSampler as DSampler
    from data.collator import TextCollator
    sampler = DSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_cfg.get("micro_batch_size", 2),
        sampler=sampler,
        collate_fn=TextCollator(),
        num_workers=training_config.get("data", {}).get("num_workers", 4),
        pin_memory=True,
    )

    trainer = SFTTrainer(model, training_config, device)

    start_step = 0
    if resume_from:
        extra = trainer.ckpt_mgr.load(model, trainer.optimizer, trainer.scheduler, resume_from)
        start_step = extra.get("step", 0)

    total_steps = training_config.get("scheduler", {}).get("total_steps", 5000)
    trainer.train(dataloader, total_steps, start_step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT training for Qwen3")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--variant", default="cold_start", choices=["cold_start", "fusion"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_sft(args.model_config, args.training_config, args.variant, args.resume, args.max_samples)
