#!/bin/bash
# RL launch script — supports both GRPO and DPO via the unified train.py
# Usage: ./scripts/rl.sh [NUM_GPUS] [MODEL_CONFIG] [TRAINING_CONFIG] [ALGO] [CHECKPOINT]

set -euo pipefail

NUM_GPUS=${1:-1}
MODEL_CONFIG=${2:-configs/model/debug.yaml}
TRAINING_CONFIG=${3:-configs/training/rl.yaml}
ALGO=${4:-grpo}
CHECKPOINT=${5:-""}

echo "=== Qwen3 RL ($ALGO) ==="
echo "GPUs:            $NUM_GPUS"
echo "Model config:    $MODEL_CONFIG"
echo "Training config: $TRAINING_CONFIG"
echo "Algorithm:       $ALGO"
echo "Checkpoint:      $CHECKPOINT"
echo "========================="

EXTRA_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    EXTRA_ARGS="--resume $CHECKPOINT"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29502 \
        train.py "$ALGO" \
        --model-config "$MODEL_CONFIG" \
        --training-config "$TRAINING_CONFIG" \
        $EXTRA_ARGS
else
    python train.py "$ALGO" \
        --model-config "$MODEL_CONFIG" \
        --training-config "$TRAINING_CONFIG" \
        $EXTRA_ARGS
fi
