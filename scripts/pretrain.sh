#!/bin/bash
# Pretraining launch script — uses the unified train.py orchestrator
# Usage: ./scripts/pretrain.sh [NUM_GPUS] [MODEL_CONFIG] [TRAINING_CONFIG] [STAGE]

set -euo pipefail

NUM_GPUS=${1:-1}
MODEL_CONFIG=${2:-configs/model/debug.yaml}
TRAINING_CONFIG=${3:-configs/training/pretrain.yaml}
STAGE=${4:-1}

echo "=== Qwen3 Pretraining ==="
echo "GPUs:            $NUM_GPUS"
echo "Model config:    $MODEL_CONFIG"
echo "Training config: $TRAINING_CONFIG"
echo "Stage:           $STAGE"
echo "========================="

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        train.py pretrain \
        --model-config "$MODEL_CONFIG" \
        --training-config "$TRAINING_CONFIG" \
        --stage "$STAGE"
else
    python train.py pretrain \
        --model-config "$MODEL_CONFIG" \
        --training-config "$TRAINING_CONFIG" \
        --stage "$STAGE"
fi
