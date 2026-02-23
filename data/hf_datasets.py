"""HuggingFace dataset loaders for every training phase.

Provides factory functions that return PyTorch-compatible datasets backed by
the ``datasets`` library.  Streaming mode is used by default so that
terabyte-scale corpora can be consumed without downloading everything first.

Supported phases
----------------
- Pretrain S1 (general):   FineWeb, SlimPajama, Dolma
- Pretrain S2 (reasoning): The Stack v2, Proof-Pile-2, OpenWebMath
- Pretrain S3 (long ctx):  ProLong-64K, PG-19
- SFT cold-start:          OpenMathInstruct-2, NuminaMath-CoT, CodeFeedback
- SFT fusion:              Infinity-Instruct, UltraChat-200K, WildChat
- GRPO / RL:               Big-Math-RL-Verified, GSM8K, MATH
- DPO / Preference:        UltraFeedback, Skywork-Preference-80K
- Multimodal:              LLaVA-Instruct-150K
"""

from __future__ import annotations

import re
from typing import Optional, List, Dict, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

try:
    from datasets import load_dataset, interleave_datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ---------------------------------------------------------------------------
# Registry of HuggingFace dataset identifiers per phase
# ---------------------------------------------------------------------------

PRETRAIN_DATASETS = {
    1: [
        {"path": "HuggingFaceFW/fineweb", "name": "sample-10BT", "text_col": "text"},
        {"path": "cerebras/SlimPajama-627B", "text_col": "text"},
    ],
    2: [
        {"path": "open-web-math/open-web-math", "text_col": "text"},
        {"path": "EleutherAI/proof-pile-2", "name": "default", "text_col": "text"},
    ],
    3: [
        {"path": "emozilla/pg19", "text_col": "text"},
    ],
}

SFT_DATASETS = {
    "cold_start": [
        {"path": "nvidia/OpenMathInstruct-2", "question_col": "problem", "answer_col": "generated_solution"},
        {"path": "AI-MO/NuminaMath-CoT", "question_col": "problem", "answer_col": "solution"},
    ],
    "fusion": [
        {"path": "HuggingFaceH4/ultrachat_200k", "messages_col": "messages"},
        {"path": "teknium/OpenHermes-2.5", "messages_col": "conversations"},
    ],
}

GRPO_DATASETS = [
    {"path": "openai/gsm8k", "name": "main", "question_col": "question", "answer_col": "answer"},
    {"path": "lighteval/MATH", "name": "all", "question_col": "problem", "answer_col": "solution"},
]

DPO_DATASETS = [
    {"path": "openbmb/UltraFeedback", "name": "default"},
]

MULTIMODAL_DATASETS = [
    {"path": "liuhaotian/LLaVA-Instruct-150K"},
]


# ---------------------------------------------------------------------------
# Pretrain loaders
# ---------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """Wraps one or more HF streaming datasets, tokenises on the fly, and
    packs tokens into fixed-length training sequences."""

    def __init__(
        self,
        hf_configs: List[dict],
        tokenizer,
        max_seq_len: int = 4096,
        seed: int = 42,
    ):
        self.hf_configs = hf_configs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        streams = []
        for cfg in self.hf_configs:
            kwargs = {"streaming": True, "split": "train", "trust_remote_code": True}
            if "name" in cfg:
                kwargs["name"] = cfg["name"]
            ds = load_dataset(cfg["path"], **kwargs)
            streams.append((ds, cfg.get("text_col", "text")))

        buffer: list[int] = []
        stream_idx = 0

        while True:
            ds, text_col = streams[stream_idx % len(streams)]
            for row in ds:
                text = row.get(text_col, "")
                if not text:
                    continue
                ids = self.tokenizer.encode(text)
                buffer.extend(ids)

                while len(buffer) >= self.max_seq_len + 1:
                    chunk = buffer[: self.max_seq_len + 1]
                    buffer = buffer[self.max_seq_len + 1 :]
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    yield {"input_ids": input_ids, "labels": labels}

            stream_idx += 1


def load_pretrain_dataset(
    stage: int,
    tokenizer,
    max_seq_len: int = 4096,
) -> IterableDataset:
    """Return a streaming pretrain dataset for the given stage (1, 2, or 3)."""
    if not HAS_DATASETS:
        raise ImportError("Install the `datasets` package: pip install datasets")
    configs = PRETRAIN_DATASETS.get(stage, PRETRAIN_DATASETS[1])
    return StreamingTextDataset(configs, tokenizer, max_seq_len)


# ---------------------------------------------------------------------------
# SFT loaders
# ---------------------------------------------------------------------------

def _extract_answer_number(text: str) -> str:
    """Pull the final numeric answer from a GSM8K-style solution."""
    matches = re.findall(r'####\s*(.+)', text)
    if matches:
        return matches[-1].strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()


def _hf_to_sft_samples_qa(
    hf_dataset,
    question_col: str,
    answer_col: str,
    max_samples: Optional[int] = None,
    with_reasoning: bool = True,
) -> List[Dict]:
    """Convert a question/answer HF dataset into SFT sample dicts."""
    samples = []
    for i, row in enumerate(hf_dataset):
        if max_samples and i >= max_samples:
            break
        q = row.get(question_col, "")
        a = row.get(answer_col, "")
        if not q or not a:
            continue
        turn = {"user": q, "assistant": _extract_answer_number(a) if "####" in a else a}
        if with_reasoning:
            turn["reasoning"] = a
        samples.append({"conversations": [turn]})
    return samples


def _hf_to_sft_samples_messages(
    hf_dataset,
    messages_col: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Convert a messages-format HF dataset into SFT sample dicts."""
    samples = []
    for i, row in enumerate(hf_dataset):
        if max_samples and i >= max_samples:
            break
        messages = row.get(messages_col, [])
        if not messages:
            continue
        convs = []
        j = 0
        while j < len(messages) - 1:
            user_msg = messages[j]
            asst_msg = messages[j + 1]
            u_content = user_msg.get("content", "") if isinstance(user_msg, dict) else str(user_msg)
            a_content = asst_msg.get("content", "") if isinstance(asst_msg, dict) else str(asst_msg)
            if u_content and a_content:
                convs.append({"user": u_content, "assistant": a_content})
            j += 2
        if convs:
            samples.append({"conversations": convs})
    return samples


def load_sft_dataset(
    variant: str,
    tokenizer,
    max_seq_len: int = 8192,
    max_samples: Optional[int] = None,
    thinking_enabled: bool = True,
) -> Dataset:
    """Load an SFT dataset.

    Args:
        variant: ``"cold_start"`` for math/code CoT data, ``"fusion"`` for
                 mixed thinking + general data.
        tokenizer: tokenizer instance with ``.encode()`` method.
        max_seq_len: maximum sequence length after tokenisation.
        max_samples: cap on number of samples (useful for debugging).
        thinking_enabled: whether to enable ``<think>`` blocks.

    Returns:
        An ``SFTDataset`` instance ready for a ``DataLoader``.
    """
    if not HAS_DATASETS:
        raise ImportError("Install the `datasets` package: pip install datasets")

    from training.sft import SFTDataset

    configs = SFT_DATASETS.get(variant, SFT_DATASETS["cold_start"])
    all_samples: list[dict] = []

    for cfg in configs:
        kwargs: dict = {"split": "train", "trust_remote_code": True}
        if "name" in cfg:
            kwargs["name"] = cfg["name"]
        ds = load_dataset(cfg["path"], **kwargs)

        if "question_col" in cfg:
            samples = _hf_to_sft_samples_qa(
                ds, cfg["question_col"], cfg["answer_col"],
                max_samples=max_samples,
                with_reasoning=(variant == "cold_start"),
            )
        elif "messages_col" in cfg:
            samples = _hf_to_sft_samples_messages(
                ds, cfg["messages_col"], max_samples=max_samples,
            )
        else:
            continue
        all_samples.extend(samples)

    if max_samples:
        all_samples = all_samples[:max_samples]

    return SFTDataset(
        samples=all_samples,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        thinking_enabled=thinking_enabled,
    )


# ---------------------------------------------------------------------------
# GRPO / RL loaders
# ---------------------------------------------------------------------------

def load_grpo_dataset(
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load queries with verifiable answers for GRPO training.

    Returns a list of ``{"prompt": str, "answer": str}`` dicts.
    """
    if not HAS_DATASETS:
        raise ImportError("Install the `datasets` package: pip install datasets")

    queries: list[dict] = []
    for cfg in GRPO_DATASETS:
        kwargs: dict = {"split": "train", "trust_remote_code": True}
        if "name" in cfg:
            kwargs["name"] = cfg["name"]
        ds = load_dataset(cfg["path"], **kwargs)
        q_col = cfg.get("question_col", "question")
        a_col = cfg.get("answer_col", "answer")
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            q = row.get(q_col, "")
            a = row.get(a_col, "")
            if q and a:
                queries.append({"prompt": q, "answer": _extract_answer_number(a)})

    if max_samples:
        queries = queries[:max_samples]
    return queries


# ---------------------------------------------------------------------------
# DPO / Preference loaders
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
    """Dataset of chosen/rejected pairs for DPO training."""

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer,
        max_seq_len: int = 2048,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        def _tokenize(text: str) -> torch.Tensor:
            ids = self.tokenizer.encode(text, allowed_special="all")
            ids = ids[: self.max_seq_len]
            if len(ids) < self.max_seq_len:
                ids = ids + [0] * (self.max_seq_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        chosen_ids = _tokenize(pair["chosen"])
        rejected_ids = _tokenize(pair["rejected"])

        return {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_labels": chosen_ids.clone(),
            "rejected_labels": rejected_ids.clone(),
        }


def _build_ultrafeedback_pairs(ds) -> List[Dict]:
    """Convert UltraFeedback dataset rows into chosen/rejected text pairs."""
    pairs: list[dict] = []
    for row in ds:
        completions = row.get("completions", [])
        if len(completions) < 2:
            continue
        scored = []
        for c in completions:
            rating = c.get("overall_score", c.get("fine-grained_score", 0))
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except ValueError:
                    rating = 0.0
            text = c.get("response", "")
            if text:
                scored.append((float(rating), text))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        instruction = row.get("instruction", "")
        pairs.append({
            "chosen": f"User: {instruction}\nAssistant: {scored[0][1]}",
            "rejected": f"User: {instruction}\nAssistant: {scored[-1][1]}",
        })
    return pairs


def load_dpo_dataset(
    tokenizer,
    max_seq_len: int = 2048,
    max_samples: Optional[int] = None,
) -> DPODataset:
    """Load a preference dataset for DPO training."""
    if not HAS_DATASETS:
        raise ImportError("Install the `datasets` package: pip install datasets")

    all_pairs: list[dict] = []
    for cfg in DPO_DATASETS:
        kwargs: dict = {"split": "train", "trust_remote_code": True}
        if "name" in cfg:
            kwargs["name"] = cfg["name"]
        ds = load_dataset(cfg["path"], **kwargs)

        if "UltraFeedback" in cfg["path"]:
            pairs = _build_ultrafeedback_pairs(ds)
        else:
            pairs = []
            for row in ds:
                chosen = row.get("chosen", "")
                rejected = row.get("rejected", "")
                if chosen and rejected:
                    pairs.append({"chosen": chosen, "rejected": rejected})
        all_pairs.extend(pairs)

    if max_samples:
        all_pairs = all_pairs[:max_samples]

    return DPODataset(all_pairs, tokenizer, max_seq_len)


# ---------------------------------------------------------------------------
# Multimodal loaders
# ---------------------------------------------------------------------------

def load_multimodal_dataset(
    tokenizer,
    image_processor=None,
    max_seq_len: int = 4096,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load a multimodal image-text dataset.

    Returns a ``MultimodalDataset`` from our data pipeline.
    """
    if not HAS_DATASETS:
        raise ImportError("Install the `datasets` package: pip install datasets")

    from data.multimodal_dataset import MultimodalDataset, ImageProcessor

    cfg = MULTIMODAL_DATASETS[0]
    ds = load_dataset(cfg["path"], split="train", trust_remote_code=True)

    samples: list[dict] = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        convs = row.get("conversations", [])
        if not convs or len(convs) < 2:
            continue
        text = " ".join(c.get("value", "") for c in convs if c.get("from") in ("human", "user"))
        samples.append({"text": text, "images": [], "image_positions": []})

    proc = image_processor or ImageProcessor()
    return MultimodalDataset(
        samples=samples,
        tokenizer=tokenizer,
        image_processor=proc,
        max_seq_len=max_seq_len,
    )
