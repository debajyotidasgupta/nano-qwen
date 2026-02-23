# Qwen3 Multimodal MoE — From-Scratch Implementation

A complete, from-scratch PyTorch implementation of a Qwen3-style multimodal
Mixture-of-Experts (MoE) language model with custom Triton kernels,
composable N-D distributed training, and a full pretraining → post-training
→ reinforcement learning pipeline.

> **Reference:** Qwen3 Technical Report (arXiv:2505.09388), Qwen3-VL
> (arXiv:2511.21631), DeepSeek-R1 / GRPO (arXiv:2501.12948).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Model Configurations](#model-configurations)
4. [Training Pipeline](#training-pipeline)
5. [Datasets](#datasets)
6. [Distributed Training](#distributed-training)
7. [Triton Kernels](#triton-kernels)
8. [Quick Start](#quick-start)
9. [Usage Examples](#usage-examples)
10. [Evaluation & Inference](#evaluation--inference)
11. [References](#references)

---

## Architecture Overview

The model follows the Qwen3 architecture scaled from a tiny debug config
(~25 M params) to the full 235B-A22B flagship:

- **235 B total / 22 B active** parameters, 94 layers, 128 experts, top-8 routing
- Grouped Query Attention (GQA) with 64 Q heads / 4 KV heads
- **QK-Normalization** (L2-norm on Q and K, replaces QKV bias from Qwen2)
- RoPE positional encoding with θ = 1 000 000 and **YARN** extension for
  context-length scaling (32 K → 128 K)
- SwiGLU feed-forward networks
- No shared experts; **global-batch load-balancing loss** (coefficient 0.001)
- Vision: ViT encoder + DeepStack multi-level fusion + Interleaved MRoPE

### Model Architecture Diagram

```
                          ┌─────────────────────────────────────┐
                          │          Qwen3 Model                │
                          │                                     │
  Text ──► Tokenizer ──► │  Token Embedding                    │
                          │       │                             │
  Image ──► PatchEmbed ──►│  ViT Blocks ──► DeepStack ──► Merger│
                          │       │               │             │
                          │       └───────────────┘             │
                          │              │                      │
                          │    N × Transformer Blocks           │
                          │    ┌─────────────────────┐          │
                          │    │ RMSNorm             │          │
                          │    │ GQA Attention        │          │
                          │    │   (QK-Norm + RoPE)  │          │
                          │    │ Residual Add        │          │
                          │    │ RMSNorm             │          │
                          │    │ MoE / SwiGLU FFN    │          │
                          │    │ Residual Add        │          │
                          │    └─────────────────────┘          │
                          │              │                      │
                          │         RMSNorm                     │
                          │         LM Head ──► logits          │
                          └─────────────────────────────────────┘
```

### Transformer Block Detail

Each block uses **pre-norm** residual connections:

```
x  ──► RMSNorm ──► GQA Attention (QK-Norm + RoPE) ──► + ──►
   │                                                   ▲
   └───────────────────────────────────────────────────┘

   ──► RMSNorm ──► MoE Layer (128 experts, top-8) ──► + ──►
   │                                                   ▲
   └───────────────────────────────────────────────────┘
```

### Vision Encoder

```
Image ──► Patch Embedding (14×14 conv)
      ──► ViT Blocks (24 layers)
      ──► DeepStack (fuse layers 4, 8, 12, …, L)
      ──► Merger (2×2 spatial pooling + linear projection)
      ──► Visual tokens interleaved with text tokens
```

---

## Project Structure

```
nanogpt/
├── configs/
│   ├── model/
│   │   ├── debug.yaml            # ~25M dense, for local testing
│   │   ├── debug_moe.yaml        # ~50M MoE, for MoE testing
│   │   ├── small.yaml            # Qwen3-1.7B-like dense
│   │   ├── medium.yaml           # Qwen3-8B-like dense
│   │   └── large.yaml            # Qwen3-235B-A22B MoE
│   └── training/
│       ├── pretrain.yaml         # Pretraining defaults
│       ├── sft.yaml              # SFT defaults
│       └── rl.yaml               # GRPO/DPO defaults
├── model/
│   ├── config.py                 # ModelConfig dataclass
│   ├── embedding.py              # Token embedding
│   ├── norm.py                   # RMSNorm
│   ├── rope.py                   # RoPE + YARN extension
│   ├── attention.py              # GQA with QK-Norm + KV-cache
│   ├── ffn.py                    # SwiGLU FFN
│   ├── moe.py                    # Top-K router, expert FFN, MoE layer
│   ├── transformer.py            # Transformer block + Qwen3Model
│   ├── multimodal.py             # Qwen3VLModel wrapper
│   └── vision/
│       ├── vit.py                # Vision Transformer encoder
│       ├── patch_embed.py        # Patch embedding (2D conv)
│       ├── deepstack.py          # Multi-level feature fusion
│       ├── merger.py             # Vision-language token merger
│       └── mrope.py              # Interleaved 3D MRoPE
├── kernels/
│   ├── rmsnorm.py                # Triton fused RMSNorm
│   ├── rope.py                   # Triton fused RoPE
│   ├── flash_attn.py             # Triton FlashAttention-2
│   ├── swiglu.py                 # Triton fused SwiGLU
│   ├── cross_entropy.py          # Triton fused cross-entropy
│   └── moe_dispatch.py           # Triton MoE scatter/gather
├── distributed/
│   ├── comm.py                   # Communication primitives
│   ├── parallel.py               # N-D parallelism orchestrator
│   ├── fsdp.py                   # FSDP2 wrapping
│   ├── tensor_parallel.py        # Column/row parallel linear
│   ├── pipeline_parallel.py      # Pipeline stage splitting
│   ├── expert_parallel.py        # Expert parallelism (all-to-all)
│   └── context_parallel.py       # Ring attention for long sequences
├── data/
│   ├── tokenizer.py              # BPE tokenizer (tiktoken/sentencepiece)
│   ├── dataset.py                # Packed text dataset (memmap)
│   ├── multimodal_dataset.py     # Image-text dataset + ImageProcessor
│   ├── sampler.py                # Distributed sampler with resume
│   ├── collator.py               # Text and multimodal collators
│   └── hf_datasets.py            # HuggingFace dataset loaders
├── training/
│   ├── pretrain.py               # 3-stage pretraining loop
│   ├── sft.py                    # SFT + Thinking Mode Fusion
│   ├── grpo.py                   # Group Relative Policy Optimization
│   ├── dpo.py                    # Direct Preference Optimization
│   ├── reward_model.py           # Reward model + verifiable rewards
│   ├── optimizer.py              # AdamW (fused variants)
│   ├── scheduler.py              # Warmup-Cosine / WSD schedulers
│   ├── checkpoint.py             # Distributed checkpointing (DCP)
│   └── logger.py                 # WandB + TensorBoard logging
├── eval/
│   ├── evaluate.py               # Perplexity, MCQ, generative eval
│   ├── generate.py               # Autoregressive generation + KV-cache
│   └── benchmarks.py             # Benchmark runner
├── scripts/
│   ├── pretrain.sh               # torchrun launch: pretraining
│   ├── sft.sh                    # torchrun launch: SFT
│   ├── rl.sh                     # torchrun launch: GRPO/DPO
│   └── eval.sh                   # Evaluation launcher
├── train.py                      # Unified CLI orchestrator
├── requirements.txt
└── description.md                # (this file)
```

---

## Model Configurations

| Config         | Params (est.) | Layers | Q Heads | KV Heads | FFN Dim | Experts | Top-K | Max Ctx  |
|----------------|---------------|--------|---------|----------|---------|---------|-------|----------|
| `debug`        | ~25 M         | 6      | 8       | 2        | 688     | 0       | —     | 2 048    |
| `debug_moe`    | ~50 M         | 6      | 8       | 2        | 688     | 8       | 2     | 2 048    |
| `small`        | ~1.7 B        | 28     | 16      | 8        | 5 504   | 0       | —     | 32 768   |
| `medium`       | ~8 B          | 36     | 32      | 8        | 12 288  | 0       | —     | 131 072  |
| `large`        | ~235 B (22B)  | 94     | 64      | 4        | 12 288  | 128     | 8     | 131 072  |

### ModelConfig Fields

The `ModelConfig` dataclass (`model/config.py`) provides:

**Core:**
`hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads`, `head_dim`, `intermediate_size`, `vocab_size`,
`max_position_embeddings`, `rms_norm_eps`, `hidden_act`

**Attention:**
`attention_bias`, `attention_dropout`, `qk_norm`

**RoPE:**
`rope_theta`, `rope_scaling` (dict for YARN parameters)

**MoE:**
`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`,
`decoder_sparse_step`, `router_aux_loss_coef`, `norm_topk_prob`,
`moe_drop_tokens`, `moe_capacity_factor`

**Vision (ViT):**
`vision_enabled`, `vision_hidden_size`, `vision_num_hidden_layers`,
`vision_num_attention_heads`, `vision_intermediate_size`,
`vision_patch_size`, `vision_image_size`, `vision_num_channels`,
`vision_deepstack_layers`, `vision_merger_spatial_factor`

**Training:**
`gradient_checkpointing`, `initializer_range`

Configs can be loaded from YAML/JSON and support presets:

```python
from model.config import ModelConfig

config = ModelConfig.from_yaml("configs/model/large.yaml")
print(config.total_params_estimate())  # ~235B
```

---

## Training Pipeline

The Qwen3 training pipeline consists of 4 major stages executed sequentially.

### Stage 1 — Pretraining (3 sub-stages)

| Sub-stage | Focus | Context Length | Data |
|-----------|-------|---------------|------|
| S1 | General knowledge | 4 096 | Web text (FineWeb, SlimPajama) |
| S2 | Reasoning & code | 4 096 | Math proofs, code (OpenWebMath, Proof-Pile-2) |
| S3 | Long context | 32 768 | Long documents (PG-19) with YARN + ABF |

The pretraining loop (`training/pretrain.py`) supports:
- Mixed precision (BF16 via `torch.amp`)
- Gradient accumulation across micro-batches
- Gradient clipping (default max_norm = 1.0)
- AdamW with β₁=0.9, β₂=0.95, weight_decay=0.1
- Warmup-cosine or WSD learning rate schedule
- MoE auxiliary load-balancing loss (global-batch)
- Distributed checkpointing (async DCP)
- WandB / TensorBoard logging

### Stage 2 — Supervised Fine-Tuning (SFT)

Two sub-phases handled by `training/sft.py`:

**Long-CoT Cold Start:** Train on curated math/code/logic problems with
verified chain-of-thought (CoT) solutions. Uses `<think>` blocks:

```
<|im_start|>user
Solve 2x + 3 = 7<|im_end|>
<|im_start|>assistant
<think>
2x + 3 = 7
2x = 4
x = 2
</think>
x = 2<|im_end|>
```

**Thinking Mode Fusion:** Mix "thinking" (long-CoT) and "non-thinking"
(standard) data so the model can serve as both a reasoning model and a
standard chat model.  Non-thinking samples use empty `<think></think>` blocks.

### Stage 3 — Reinforcement Learning

**GRPO** (Group Relative Policy Optimization — `training/grpo.py`):
1. For each query, sample G responses from the policy
2. Score responses with verifiable reward (math: exact match) or model-based reward
3. Compute group-relative advantage: A_i = (R_i − mean(R)) / std(R)
4. Update policy with clipped surrogate objective + KL penalty against frozen reference model

**DPO** (Direct Preference Optimization — `training/dpo.py`):
- Offline DPO on preference pairs (chosen vs rejected)
- Online DPO: generate preference pairs from the current policy, score with reward model
- Loss: −log σ(β · (log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)))

**Reward Model** (`training/reward_model.py`):
- Same backbone as the policy model with a scalar value head
- Verifiable rewards for math (exact answer match) and code (test execution)

### Pipeline Diagram

```
Pretrain S1 ──► Pretrain S2 ──► Pretrain S3
  (general)     (reasoning)     (long ctx)
                                    │
                                    ▼
                          SFT Cold Start (Long-CoT)
                                    │
                                    ▼
                     RL Stage 2 — Reasoning GRPO
                                    │
                                    ▼
                     SFT Stage 3 — Thinking Fusion
                                    │
                                    ▼
                     RL Stage 4 — General RL (GRPO + DPO)
                                    │
                                    ▼
                              Final Model
```

---

## Datasets

All datasets are loaded via `data/hf_datasets.py` using the HuggingFace
`datasets` library.  Streaming is used by default for large corpora.

### Pretraining Datasets

| Stage | Dataset | HuggingFace ID | Description |
|-------|---------|----------------|-------------|
| S1 | FineWeb (10B sample) | `HuggingFaceFW/fineweb` | High-quality web text |
| S1 | SlimPajama | `cerebras/SlimPajama-627B` | Curated web + books + code |
| S2 | OpenWebMath | `open-web-math/open-web-math` | Mathematical web pages |
| S2 | Proof-Pile-2 | `EleutherAI/proof-pile-2` | Mathematical proofs + papers |
| S3 | PG-19 | `emozilla/pg19` | Full-length books (long ctx) |

### SFT Datasets

| Variant | Dataset | HuggingFace ID | Description |
|---------|---------|----------------|-------------|
| Cold start | OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | Math problems with solutions |
| Cold start | NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | Competition math with CoT |
| Fusion | UltraChat 200K | `HuggingFaceH4/ultrachat_200k` | Multi-turn conversations |
| Fusion | OpenHermes-2.5 | `teknium/OpenHermes-2.5` | Diverse instruction data |

### RL / GRPO Datasets

| Dataset | HuggingFace ID | Description |
|---------|----------------|-------------|
| GSM8K | `openai/gsm8k` | Grade-school math (verifiable) |
| MATH | `lighteval/MATH` | Competition math (verifiable) |

### DPO / Preference Datasets

| Dataset | HuggingFace ID | Description |
|---------|----------------|-------------|
| UltraFeedback | `openbmb/UltraFeedback` | Multi-model preference ratings |

### Multimodal Datasets

| Dataset | HuggingFace ID | Description |
|---------|----------------|-------------|
| LLaVA-Instruct-150K | `liuhaotian/LLaVA-Instruct-150K` | Image-text instruction pairs |

### Using Datasets

```python
from data.hf_datasets import (
    load_pretrain_dataset,
    load_sft_dataset,
    load_grpo_dataset,
    load_dpo_dataset,
    load_multimodal_dataset,
)
from data.tokenizer import Tokenizer

tok = Tokenizer(backend="tiktoken")

# Streaming pretrain data
pretrain_ds = load_pretrain_dataset(stage=1, tokenizer=tok, max_seq_len=4096)

# SFT data
sft_ds = load_sft_dataset("cold_start", tokenizer=tok, max_seq_len=8192, max_samples=10000)

# GRPO queries with verifiable answers
grpo_queries = load_grpo_dataset(max_samples=5000)

# DPO preference pairs
dpo_ds = load_dpo_dataset(tokenizer=tok, max_seq_len=2048)
```

---

## Distributed Training

The project implements composable N-D parallelism following `torchtitan`
patterns, orchestrated by `distributed/parallel.py`.

### Parallelism Strategies

```
                    ┌────────────────────────────────────┐
                    │      DeviceMesh (e.g. 64 GPUs)     │
                    │                                    │
                    │   DP=8 × TP=2 × PP=4              │
                    │                                    │
                    │   ┌─────────┐  ┌────────────────┐  │
                    │   │  FSDP2  │  │ Tensor Parallel │  │
                    │   │(shard   │  │(col/row split   │  │
                    │   │ params, │  │ linear layers)  │  │
                    │   │ grads,  │  └────────────────┘  │
                    │   │ optim)  │                      │
                    │   └─────────┘  ┌────────────────┐  │
                    │                │Pipeline Parallel│  │
                    │   ┌─────────┐  │(layer sharding, │  │
                    │   │ Expert  │  │ 1F1B schedule)  │  │
                    │   │Parallel │  └────────────────┘  │
                    │   │(all2all)│                      │
                    │   └─────────┘  ┌────────────────┐  │
                    │                │Context Parallel │  │
                    │                │(ring attention) │  │
                    │                └────────────────┘  │
                    └────────────────────────────────────┘
```

| Strategy | Module | Description |
|----------|--------|-------------|
| **FSDP2** | `distributed/fsdp.py` | Fully Sharded Data Parallel — shards params, gradients, and optimizer states per transformer block. Mixed precision (BF16 compute, FP32 master weights). |
| **Tensor Parallel** | `distributed/tensor_parallel.py` | Column-parallel for Q/K/V/gate/up projections, row-parallel for output/down. Uses `DTensor` APIs. |
| **Pipeline Parallel** | `distributed/pipeline_parallel.py` | Splits model layers across stages. 1F1B schedule minimises pipeline bubble. |
| **Expert Parallel** | `distributed/expert_parallel.py` | Distributes MoE experts across ranks with all-to-all communication. |
| **Context Parallel** | `distributed/context_parallel.py` | Ring attention for 128K+ sequences by splitting across CP ranks. |

### Configuration

```yaml
# In training config YAML
distributed:
  dp_size: -1         # auto-compute from world_size
  tp_size: 2
  pp_size: 4
  ep_size: 8
  cp_size: 1
  fsdp_enabled: true
```

---

## Triton Kernels

Six custom Triton kernels in `kernels/` provide fused, memory-efficient
implementations of performance-critical operations.  Each has a pure PyTorch
fallback for correctness testing.

| Kernel | File | What It Fuses |
|--------|------|---------------|
| **RMSNorm** | `kernels/rmsnorm.py` | Variance → normalise → scale in a single kernel |
| **RoPE** | `kernels/rope.py` | In-place rotary embedding application |
| **FlashAttention-2** | `kernels/flash_attn.py` | Tiled softmax with online rescaling, GQA support, causal masking |
| **SwiGLU** | `kernels/swiglu.py` | SiLU activation + element-wise gate multiply |
| **Cross-Entropy** | `kernels/cross_entropy.py` | Chunk-wise softmax for large vocabularies without materialising full logits |
| **MoE Dispatch/Combine** | `kernels/moe_dispatch.py` | Token scatter to experts + weighted gather |

---

## Quick Start

### Installation

```bash
# Clone
git clone <repo-url> nanogpt && cd nanogpt

# Create environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.5.0
- Triton ≥ 3.0.0 (for custom kernels)
- CUDA-capable GPU (BF16 support recommended — Ampere or newer)

### Single-GPU Debug Training

```bash
# Pretraining (debug model, stage 1)
python train.py pretrain \
    --model-config configs/model/debug.yaml \
    --training-config configs/training/pretrain.yaml \
    --stage 1

# SFT (cold-start with math CoT data)
python train.py sft \
    --model-config configs/model/debug.yaml \
    --variant cold_start \
    --max-samples 500

# GRPO (reinforcement learning)
python train.py grpo \
    --model-config configs/model/debug.yaml \
    --max-samples 200

# DPO (preference optimisation)
python train.py dpo \
    --model-config configs/model/debug.yaml \
    --max-samples 200
```

### Multi-GPU Training

```bash
# 8-GPU pretraining with FSDP + TP
torchrun --nproc_per_node=8 train.py pretrain \
    --model-config configs/model/medium.yaml \
    --training-config configs/training/pretrain.yaml \
    --stage 1

# Using shell scripts
./scripts/pretrain.sh 8 configs/model/medium.yaml configs/training/pretrain.yaml 1
./scripts/sft.sh 8 configs/model/medium.yaml configs/training/sft.yaml cold_start
./scripts/rl.sh 8 configs/model/medium.yaml configs/training/rl.yaml grpo
```

### Multi-Node Training

```bash
# Node 0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --master_addr=<IP> --master_port=29500 \
    train.py pretrain --model-config configs/model/large.yaml --stage 1

# Node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
    --master_addr=<IP> --master_port=29500 \
    train.py pretrain --model-config configs/model/large.yaml --stage 1
```

---

## Usage Examples

### Full Training Pipeline (Debug Model)

```bash
# 1. Pretrain (3 stages)
python train.py pretrain --model-config configs/model/debug.yaml --stage 1
python train.py pretrain --model-config configs/model/debug.yaml --stage 2
python train.py pretrain --model-config configs/model/debug.yaml --stage 3

# 2. SFT — Long-CoT cold start
python train.py sft --model-config configs/model/debug.yaml --variant cold_start

# 3. GRPO — Reasoning RL
python train.py grpo --model-config configs/model/debug.yaml

# 4. SFT — Thinking mode fusion
python train.py sft --model-config configs/model/debug.yaml --variant fusion

# 5. DPO — General RL
python train.py dpo --model-config configs/model/debug.yaml

# 6. Evaluate
python train.py eval --model-config configs/model/debug.yaml \
    --checkpoint checkpoints/step_5000 --benchmarks hellaswag mmlu
```

### MoE Model Training

```bash
# Debug MoE (8 experts, top-2)
python train.py pretrain --model-config configs/model/debug_moe.yaml --stage 1

# Full-scale MoE (128 experts, top-8) on 64 GPUs
torchrun --nproc_per_node=8 --nnodes=8 \
    train.py pretrain --model-config configs/model/large.yaml --stage 1
```

### Resume from Checkpoint

```bash
python train.py pretrain \
    --model-config configs/model/debug.yaml \
    --resume checkpoints/step_10000
```

### Using HF Datasets Directly

```python
from data.hf_datasets import load_pretrain_dataset, load_sft_dataset
from data.tokenizer import Tokenizer
from torch.utils.data import DataLoader

tok = Tokenizer(backend="tiktoken")

# Streaming pretrain data — no upfront download
ds = load_pretrain_dataset(stage=1, tokenizer=tok, max_seq_len=4096)
dl = DataLoader(ds, batch_size=8)

for batch in dl:
    print(batch["input_ids"].shape)  # (8, 4096)
    break

# SFT dataset
sft_ds = load_sft_dataset("cold_start", tok, max_seq_len=8192, max_samples=1000)
print(f"SFT samples: {len(sft_ds)}")
```

---

## Evaluation & Inference

### Text Generation

```python
from model.config import ModelConfig
from model.transformer import Qwen3Model
from eval.generate import TextGenerator
from data.tokenizer import Tokenizer

config = ModelConfig.from_yaml("configs/model/debug.yaml")
model = Qwen3Model(config).eval()
tok = Tokenizer(backend="tiktoken")
gen = TextGenerator(model, tok)

# Standard generation
output = gen.generate("What is the meaning of life?", max_new_tokens=256)

# Thinking-mode generation
output = gen.generate("Solve x^2 - 5x + 6 = 0", max_new_tokens=512, thinking=True)
```

### Evaluation Harness

```python
from eval.evaluate import Evaluator

evaluator = Evaluator(model, device="cuda")

# Perplexity on validation data
ppl = evaluator.perplexity(val_dataloader)

# Multiple-choice benchmarks
acc = evaluator.multiple_choice(mc_questions)
```

### Benchmarks

The benchmark runner (`eval/benchmarks.py`) supports:

**Language:**
MMLU, GSM8K, MATH, HumanEval, ARC-Challenge, HellaSwag, WinoGrande, TriviaQA

**Multimodal:**
VQAv2, TextVQA, DocVQA, ChartQA

---

## Component Details

### Tokenizer (`data/tokenizer.py`)

Wraps `tiktoken` or `sentencepiece` with special tokens:

| Token | Usage |
|-------|-------|
| `<\|im_start\|>` | Start of message |
| `<\|im_end\|>` | End of message |
| `<think>` | Start of reasoning block |
| `</think>` | End of reasoning block |
| `<\|endoftext\|>` | End of text |

### Optimizer (`training/optimizer.py`)

AdamW with per-parameter weight decay:
- **Decayed:** all 2D+ weight tensors
- **Not decayed:** biases, LayerNorm/RMSNorm weights, embeddings

Fused AdamW variant uses `torch.optim.AdamW(fused=True)` for GPU efficiency.

### LR Schedulers (`training/scheduler.py`)

**Warmup-Cosine:** Linear warmup → cosine decay to `min_lr`

**WSD (Warmup-Stable-Decay):** Linear warmup → constant LR → linear/cosine decay

### Checkpointing (`training/checkpoint.py`)

- PyTorch DCP (`torch.distributed.checkpoint`) for sharded saves
- Saves: model params, optimizer states, scheduler, RNG states, step count, tokens seen
- Async checkpointing to avoid training stalls
- Resume with different parallelism configurations

### Logging (`training/logger.py`)

Unified logger supporting WandB and TensorBoard:

| Metric | Description |
|--------|-------------|
| `loss` | Cross-entropy LM loss |
| `lr` | Current learning rate |
| `grad_norm` | Gradient L2 norm (post-clip) |
| `tokens_per_sec` | Training throughput |
| `mfu` | Model FLOPs Utilisation |
| `aux_loss` | MoE load-balancing auxiliary loss |

---

## Build Order & Dependencies

The project was built in 11 phases, each tested before proceeding:

```
Phase 1:  Foundation (configs, tokenizer, project scaffold)
    │
Phase 2:  Dense Transformer (RMSNorm, RoPE, GQA, SwiGLU, Transformer)
    │
Phase 3:  Triton Kernels (fused RMSNorm, RoPE, FlashAttn, SwiGLU, XEntropy, MoE dispatch)
    │
Phase 4:  MoE (Router, Expert FFN, MoE Layer, load-balancing loss)
    │
Phase 5:  Vision Encoder (ViT, PatchEmbed, DeepStack, Merger, MRoPE)
    │
Phase 6:  Data Pipeline (PackedTextDataset, MultimodalDataset, DistributedSampler)
    │
Phase 7:  Distributed Training (FSDP2, TP, PP, EP, CP, Orchestrator)
    │
Phase 8:  Pretraining (3-stage loop, optimizer, scheduler, checkpointing, logging)
    │
Phase 9:  Post-Training (SFT, Long-CoT cold start, Thinking Mode Fusion)
    │
Phase 10: Reinforcement Learning (GRPO, DPO, Reward Model)
    │
Phase 11: Evaluation & Inference (generation, benchmarks)
```

---

## References

1. **Qwen3 Technical Report** — arXiv:2505.09388
   - Architecture, training pipeline, thinking mode, GRPO details
2. **Qwen3-VL** — arXiv:2511.21631
   - Vision encoder (ViT + DeepStack), Interleaved MRoPE, multimodal training
3. **DeepSeek-R1** — arXiv:2501.12948
   - Group Relative Policy Optimization (GRPO) algorithm
4. **DPO** — Rafailov et al., 2023
   - Direct Preference Optimization without reward models
5. **FlashAttention-2** — Dao, 2023
   - Tiled attention with online softmax rescaling
6. **RoPE** — Su et al., 2021
   - Rotary Position Embeddings
7. **YARN** — Peng et al., 2023
   - Context-length extension via modified RoPE frequencies
8. **SwiGLU** — Shazeer, 2020
   - Gated Linear Units with SiLU activation
9. **RMSNorm** — Zhang & Sennrich, 2019
   - Root Mean Square Layer Normalization
10. **torchtitan** — PyTorch team
    - Reference implementation for composable N-D parallelism
