"""Pretraining loop — 3-stage pretraining pipeline.

Stage 1: General pretraining (ctx=4096)
Stage 2: Reasoning-focused data (ctx=4096)
Stage 3: Long-context extension (ctx=32768, YARN + ABF)
"""

from __future__ import annotations

import os
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.transformer import Qwen3Model
from data.dataset import PackedTextDataset
from data.sampler import DistributedSampler
from data.collator import TextCollator
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
    barrier,
    all_reduce_mean,
)
from distributed.parallel import ParallelConfig, ParallelismManager


def load_training_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_mfu(
    model_params: int,
    tokens_per_sec: float,
    num_gpus: int,
    gpu_flops: float = 312e12,  # H100 BF16 peak
) -> float:
    """Estimate Model FLOPs Utilization (MFU)."""
    flops_per_token = 6 * model_params
    achieved_flops = flops_per_token * tokens_per_sec
    peak_flops = gpu_flops * num_gpus
    return achieved_flops / peak_flops if peak_flops > 0 else 0.0


class PreTrainer:
    """Orchestrates the 3-stage pretraining pipeline."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: dict,
    ):
        self.model_config = model_config
        self.cfg = training_config

        self.rank, self.world_size = init_distributed()
        self.local_rank = get_local_rank()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = Qwen3Model(model_config).to(self.device)
        param_count = self.model.count_parameters()
        if is_main_process():
            print(f"Model parameters: {param_count['total']:,} total, {param_count['trainable']:,} trainable")

        # Apply distributed parallelism
        p_cfg = self.cfg.get("distributed", {})
        self.parallel_config = ParallelConfig(
            dp_size=p_cfg.get("dp_size", -1),
            tp_size=p_cfg.get("tp_size", 1),
            pp_size=p_cfg.get("pp_size", 1),
            ep_size=p_cfg.get("ep_size", 1),
            cp_size=p_cfg.get("cp_size", 1),
            fsdp_enabled=p_cfg.get("fsdp_enabled", True),
            dtype=self.cfg.get("precision", {}).get("dtype", "bfloat16"),
        )

        if self.world_size > 1:
            self.parallel_mgr = ParallelismManager(self.parallel_config)
            self.model = self.parallel_mgr.apply_parallelism(self.model)
        else:
            self.parallel_mgr = None

        # Build optimizer and scheduler
        opt_cfg = self.cfg.get("optimizer", {})
        self.optimizer = build_optimizer(
            self.model,
            lr=opt_cfg.get("lr", 3e-4),
            weight_decay=opt_cfg.get("weight_decay", 0.1),
            beta1=opt_cfg.get("beta1", 0.9),
            beta2=opt_cfg.get("beta2", 0.95),
            eps=opt_cfg.get("eps", 1e-8),
            fused=opt_cfg.get("fused", True),
        )

        sched_cfg = self.cfg.get("scheduler", {})
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_type=sched_cfg.get("type", "cosine"),
            warmup_steps=sched_cfg.get("warmup_steps", 2000),
            total_steps=sched_cfg.get("total_steps", 100000),
        )

        # Checkpointing
        ckpt_cfg = self.cfg.get("checkpoint", {})
        self.ckpt_mgr = CheckpointManager(
            save_dir=ckpt_cfg.get("save_dir", "checkpoints"),
            save_interval=ckpt_cfg.get("save_interval", 1000),
            async_save=ckpt_cfg.get("async_save", True),
        )

        # Logging
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
        train_dataloader: DataLoader,
        total_steps: int,
        start_step: int = 0,
    ):
        """Main training loop."""
        self.model.train()
        dtype = getattr(torch, self.cfg.get("precision", {}).get("dtype", "bfloat16"))

        step = start_step
        accum_loss = 0.0
        accum_aux = 0.0
        tokens_seen = 0
        t0 = time.time()

        data_iter = iter(train_dataloader)

        while step < total_steps:
            self.optimizer.zero_grad()

            for micro_step in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                batch_tokens = input_ids.numel()

                with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"] / self.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()
                if outputs["aux_loss"] is not None:
                    accum_aux += outputs["aux_loss"].item() / self.grad_accum_steps
                tokens_seen += batch_tokens

            # Gradient clipping
            if self.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            else:
                grad_norm = None

            self.optimizer.step()
            self.scheduler.step()

            step += 1

            # Logging
            dt = time.time() - t0
            tokens_per_sec = tokens_seen / dt if dt > 0 else 0
            current_lr = self.scheduler.get_last_lr()[0]

            self.logger.log_step(
                step=step,
                loss=accum_loss,
                lr=current_lr,
                grad_norm=grad_norm.item() if grad_norm is not None else None,
                tokens_per_sec=tokens_per_sec,
                aux_loss=accum_aux if accum_aux > 0 else None,
            )

            accum_loss = 0.0
            accum_aux = 0.0

            # Checkpointing
            if self.ckpt_mgr.should_save(step):
                self.ckpt_mgr.save(
                    step, self.model, self.optimizer, self.scheduler,
                    extra_state={"tokens_seen": tokens_seen},
                )

        if is_main_process():
            print(f"Training complete. Total tokens: {tokens_seen:,}")
        self.logger.finish()


def run_pretraining(
    model_config_path: str,
    training_config_path: str,
    resume_from: Optional[str] = None,
):
    """Entry point for pretraining."""
    model_config = ModelConfig.from_yaml(model_config_path)
    training_config = load_training_config(training_config_path)

    trainer = PreTrainer(model_config, training_config)

    # Build dataloader
    data_cfg = training_config.get("data", {})
    batch_cfg = training_config.get("batch", {})

    dataset = PackedTextDataset(
        data_dir=data_cfg.get("train_data", "data/train"),
        max_seq_len=batch_cfg.get("max_seq_len", 4096),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=trainer.world_size,
        rank=trainer.rank,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_cfg.get("micro_batch_size", 4),
        sampler=sampler,
        collate_fn=TextCollator(),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Resume
    start_step = 0
    if resume_from:
        extra = trainer.ckpt_mgr.load(
            trainer.model, trainer.optimizer, trainer.scheduler, resume_from,
        )
        start_step = extra.get("step", 0)

    total_steps = training_config.get("scheduler", {}).get("total_steps", 100000)
    trainer.train(dataloader, total_steps, start_step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain Qwen3 model")
    parser.add_argument("--model-config", required=True, help="Path to model config YAML")
    parser.add_argument("--training-config", required=True, help="Path to training config YAML")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    run_pretraining(args.model_config, args.training_config, args.resume)
