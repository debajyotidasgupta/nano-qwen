"""Evaluation harness — benchmark running and metric computation.

Provides integration points for lm-eval-harness and custom benchmarks.
"""

from __future__ import annotations

import time
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn

from model.transformer import Qwen3Model
from eval.generate import TextGenerator


class Evaluator:
    """Evaluate model on standard benchmarks."""

    def __init__(
        self,
        model: Qwen3Model,
        tokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generator = TextGenerator(model, tokenizer, device)

    @torch.no_grad()
    def compute_perplexity(
        self,
        eval_dataloader,
        max_batches: Optional[int] = None,
    ) -> float:
        """Compute perplexity on an evaluation dataset."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            num_batches += 1

            if max_batches and num_batches >= max_batches:
                break

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def eval_multiple_choice(
        self,
        questions: List[Dict],
        num_choices: int = 4,
    ) -> dict:
        """Evaluate on multiple-choice questions (e.g., MMLU, ARC).

        Each question dict should have:
            - "prompt": str (the question text)
            - "choices": list[str] (answer options)
            - "answer": int (correct answer index)
        """
        self.model.eval()
        correct = 0
        total = 0

        for q in questions:
            prompt = q["prompt"]
            choices = q["choices"]
            answer = q["answer"]

            choice_log_probs = []
            for choice in choices:
                full = prompt + " " + choice
                ids = torch.tensor(
                    self.tokenizer.encode(full, allowed_special="all"),
                    dtype=torch.long,
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=ids)
                    logits = outputs["logits"]

                # Average log prob of choice tokens
                prompt_len = len(self.tokenizer.encode(prompt, allowed_special="all"))
                choice_logits = logits[0, prompt_len - 1:-1]
                choice_ids = ids[0, prompt_len:]

                log_probs = torch.nn.functional.log_softmax(choice_logits, dim=-1)
                token_log_probs = log_probs.gather(1, choice_ids.unsqueeze(-1)).squeeze(-1)
                avg_log_prob = token_log_probs.mean().item()
                choice_log_probs.append(avg_log_prob)

            predicted = max(range(len(choice_log_probs)), key=lambda i: choice_log_probs[i])
            if predicted == answer:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        return {"accuracy": accuracy, "correct": correct, "total": total}

    def eval_generative(
        self,
        questions: List[Dict],
        metric_fn=None,
        max_new_tokens: int = 512,
    ) -> dict:
        """Evaluate on generative benchmarks (e.g., GSM8K, HumanEval).

        Each question dict should have:
            - "prompt": str
            - "answer": str (ground truth)
        """
        predictions = []
        references = []

        for q in questions:
            response = self.generator.generate(
                q["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # greedy for evaluation
            )
            predictions.append(response)
            references.append(q["answer"])

        if metric_fn:
            return metric_fn(predictions, references)

        # Default: exact match
        correct = sum(1 for p, r in zip(predictions, references) if r.strip() in p)
        return {
            "accuracy": correct / max(len(predictions), 1),
            "correct": correct,
            "total": len(predictions),
        }
