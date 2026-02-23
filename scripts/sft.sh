#!/bin/bash
# SFT launch script — uses the unified train.py orchestrator
# Usage: ./scripts/sft.sh [NUM_GPUS] [MODEL_CONFIG] [TRAINING_CONFIG] [VARIANT] [CHECKPOINT]

set -euo pipefail

NUM_GPUS=${1:-1}
MODEL_CONFIG=${2:-configs/model/debug.yaml}
TRAINING_CONFIG=${3:-configs/training/sft.yaml}
VARIANT=${4:-cold_start}
CHECKPOINT=${5:-""}

echo "=== Qwen3 SFT ==="
echo "GPUs:            $NUM_GPUS"
echo "Model config:    $MODEL_CONFIG"
echo "Training config: $TRAINING_CONFIG"
echo "Variant:         $VARIANT"
echo "Checkpoint:      $CHECKPOINT"
echo "=================="

EXTRA_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    EXTRA_ARGS="--resume $CHECKPOINT"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29501 \
        train.py sft \
        --model-config "$MODEL_CONFIG" \
        --training-config "$TRAINING_CONFIG" \
        --variant "$VARIANT" \
        $EXTRA_ARGS
else
    python train.py sft \
        --model-config "$MODEL_CONFIG" \
        --training-config "$TRAINING_CONFIG" \
        --variant "$VARIANT" \
        $EXTRA_ARGS
fi
