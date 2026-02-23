"""Training logger with WandB and TensorBoard support."""

from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

from distributed.comm import is_main_process


class TrainingLogger:
    """Unified logging interface for training metrics."""

    def __init__(
        self,
        wandb_project: Optional[str] = None,
        wandb_enabled: bool = False,
        tensorboard_dir: Optional[str] = None,
        tensorboard_enabled: bool = True,
        log_interval: int = 10,
    ):
        self.log_interval = log_interval
        self._wandb = None
        self._tb_writer = None
        self._start_time = time.time()

        if is_main_process():
            if wandb_enabled and wandb_project:
                try:
                    import wandb
                    wandb.init(project=wandb_project, config={})
                    self._wandb = wandb
                except ImportError:
                    pass

            if tensorboard_enabled and tensorboard_dir:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    os.makedirs(tensorboard_dir, exist_ok=True)
                    self._tb_writer = SummaryWriter(tensorboard_dir)
                except ImportError:
                    pass

    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to all enabled backends."""
        if not is_main_process():
            return

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        aux_loss: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Convenience method for logging a training step."""
        if step % self.log_interval != 0:
            return

        elapsed = time.time() - self._start_time
        metrics = {
            "train/loss": loss,
            "train/lr": lr,
            "train/elapsed_sec": elapsed,
            "train/step": step,
        }

        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm
        if tokens_per_sec is not None:
            metrics["train/tokens_per_sec"] = tokens_per_sec
        if aux_loss is not None:
            metrics["train/aux_loss"] = aux_loss
        if extra:
            metrics.update(extra)

        self.log(metrics, step)

        # Print to stdout
        parts = [f"step={step}"]
        parts.append(f"loss={loss:.4f}")
        parts.append(f"lr={lr:.2e}")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.4f}")
        if tokens_per_sec is not None:
            parts.append(f"tok/s={tokens_per_sec:.0f}")
        if aux_loss is not None:
            parts.append(f"aux_loss={aux_loss:.4f}")
        print(f"[{elapsed:.1f}s] {' | '.join(parts)}")

    def finish(self):
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()
