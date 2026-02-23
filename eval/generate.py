"""Text generation with KV-cache, sampling strategies, and thinking mode.

Supports:
  - Autoregressive generation with KV-cache for O(1) per-token cost
  - Top-k, top-p (nucleus), temperature sampling
  - Thinking mode with configurable budget cap
  - Streaming output
"""

from __future__ import annotations

from typing import Optional, List, Callable

import torch
import torch.nn.functional as F

from model.transformer import Qwen3Model
from model.attention import KVCache


class TextGenerator:
    """Autoregressive text generator with KV-cache."""

    def __init__(
        self,
        model: Qwen3Model,
        tokenizer,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            enable_thinking: If True, prepend <think> to assistant response
            thinking_budget: Max tokens for the thinking phase
            stop_tokens: Token IDs that trigger generation stop
            stream_callback: Called with each new decoded token

        Returns:
            Generated text string
        """
        if enable_thinking:
            prompt = prompt + "<think>\n"

        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, allowed_special="all"),
            dtype=torch.long,
        ).unsqueeze(0).to(self.device)

        # Initialize KV caches
        kv_caches = [KVCache() for _ in range(len(self.model.layers))]

        # Prefill: process the entire prompt
        outputs = self.model(input_ids=input_ids, kv_caches=kv_caches)
        logits = outputs["logits"][:, -1, :]

        generated_ids = []
        thinking_tokens = 0
        thinking_ended = False

        eos_id = self.tokenizer.eos_id if hasattr(self.tokenizer, 'eos_id') else None
        stop_set = set(stop_tokens or [])
        if eos_id is not None:
            stop_set.add(eos_id)

        for _ in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_ids:
                for token_id in set(generated_ids):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Sample next token
            next_token = self._sample(logits, temperature, top_k, top_p)
            token_id = next_token.item()

            # Check thinking budget
            if enable_thinking and not thinking_ended:
                thinking_tokens += 1
                if thinking_budget and thinking_tokens >= thinking_budget:
                    # Force end thinking
                    think_end = self.tokenizer.encode("</think>\n", allowed_special="all")
                    generated_ids.extend(think_end)
                    thinking_ended = True
                    next_input = torch.tensor([[think_end[-1]]], device=self.device)
                    outputs = self.model(input_ids=next_input, kv_caches=kv_caches)
                    logits = outputs["logits"][:, -1, :]
                    continue

            # Check stop condition
            if token_id in stop_set:
                break

            generated_ids.append(token_id)

            # Stream callback
            if stream_callback:
                decoded = self.tokenizer.decode([token_id])
                stream_callback(decoded)

            # Check if thinking ended naturally
            if enable_thinking and not thinking_ended:
                recent = self.tokenizer.decode(generated_ids[-10:])
                if "</think>" in recent:
                    thinking_ended = True

            # Next step with KV-cache
            next_input = torch.tensor([[token_id]], device=self.device)
            outputs = self.model(input_ids=next_input, kv_caches=kv_caches)
            logits = outputs["logits"][:, -1, :]

        return self.tokenizer.decode(generated_ids)

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample from logits with temperature, top-k, and top-p filtering."""
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")

            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> List[str]:
        """Generate text for multiple prompts (no KV-cache, simpler path)."""
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt, max_new_tokens, temperature, top_k, top_p,
            )
            results.append(result)
        return results
