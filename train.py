#!/usr/bin/env python3
"""Unified training orchestrator for all Qwen3-MoE phases.

Usage
-----
::

    # Pretraining (single GPU)
    python train.py pretrain --model-config configs/model/debug.yaml --stage 1

    # Pretraining (multi-GPU)
    torchrun --nproc_per_node=8 train.py pretrain --model-config configs/model/large.yaml --stage 1

    # SFT — cold-start (chain-of-thought)
    python train.py sft --model-config configs/model/debug.yaml --variant cold_start

    # SFT — thinking-mode fusion
    python train.py sft --model-config configs/model/debug.yaml --variant fusion

    # GRPO reinforcement learning
    python train.py grpo --model-config configs/model/debug.yaml

    # DPO preference optimisation
    python train.py dpo --model-config configs/model/debug.yaml

    # Evaluation
    python train.py eval --model-config configs/model/debug.yaml --checkpoint checkpoints/step_1000
"""

from __future__ import annotations

import argparse
import sys


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--model-config", required=True, help="Path to model config YAML")
    sub.add_argument(
        "--training-config",
        default=None,
        help="Path to training config YAML (uses sensible defaults per phase if omitted)",
    )
    sub.add_argument("--resume", default=None, help="Checkpoint directory to resume from")
    sub.add_argument("--max-samples", type=int, default=None, help="Cap dataset size (for debugging)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-MoE From Scratch — Unified Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # ── pretrain ──────────────────────────────────────────────────────────
    p_pre = subs.add_parser("pretrain", help="3-stage pretraining (general → reasoning → long-ctx)")
    _add_common_args(p_pre)
    p_pre.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Pretraining stage")

    # ── sft ───────────────────────────────────────────────────────────────
    p_sft = subs.add_parser("sft", help="Supervised fine-tuning (cold-start or fusion)")
    _add_common_args(p_sft)
    p_sft.add_argument("--variant", default="cold_start", choices=["cold_start", "fusion"])

    # ── grpo ──────────────────────────────────────────────────────────────
    p_grpo = subs.add_parser("grpo", help="Group Relative Policy Optimization")
    _add_common_args(p_grpo)

    # ── dpo ───────────────────────────────────────────────────────────────
    p_dpo = subs.add_parser("dpo", help="Direct Preference Optimization")
    _add_common_args(p_dpo)

    # ── eval ──────────────────────────────────────────────────────────────
    p_eval = subs.add_parser("eval", help="Run evaluation benchmarks")
    p_eval.add_argument("--model-config", required=True)
    p_eval.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    p_eval.add_argument("--benchmarks", nargs="+", default=["hellaswag", "mmlu"])

    args = parser.parse_args()

    # Resolve default training configs when not provided
    _DEFAULT_TRAIN_CONFIGS = {
        "pretrain": "configs/training/pretrain.yaml",
        "sft": "configs/training/sft.yaml",
        "grpo": "configs/training/rl.yaml",
        "dpo": "configs/training/rl.yaml",
    }
    if args.command != "eval" and args.training_config is None:
        args.training_config = _DEFAULT_TRAIN_CONFIGS.get(args.command)

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.command == "pretrain":
        from training.pretrain import run_pretraining
        run_pretraining(
            model_config_path=args.model_config,
            training_config_path=args.training_config,
            resume_from=args.resume,
        )

    elif args.command == "sft":
        from training.sft import run_sft
        run_sft(
            model_config_path=args.model_config,
            training_config_path=args.training_config,
            variant=args.variant,
            resume_from=args.resume,
            max_samples=args.max_samples,
        )

    elif args.command == "grpo":
        from training.grpo import run_grpo
        run_grpo(
            model_config_path=args.model_config,
            training_config_path=args.training_config,
            resume_from=args.resume,
            max_samples=args.max_samples,
        )

    elif args.command == "dpo":
        from training.dpo import run_dpo
        run_dpo(
            model_config_path=args.model_config,
            training_config_path=args.training_config,
            resume_from=args.resume,
            max_samples=args.max_samples,
        )

    elif args.command == "eval":
        from model.config import ModelConfig
        from model.transformer import Qwen3Model
        from eval.evaluate import Evaluator
        import torch

        config = ModelConfig.from_yaml(args.model_config)
        model = Qwen3Model(config)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        evaluator = Evaluator(model, device=device)
        print(f"Running benchmarks: {args.benchmarks}")
        for name in args.benchmarks:
            print(f"\n{'='*40}\n{name}\n{'='*40}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
