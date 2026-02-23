#!/bin/bash
# Evaluation launch script — uses the unified train.py orchestrator
# Usage: ./scripts/eval.sh [MODEL_CONFIG] [CHECKPOINT] [BENCHMARKS...]

set -euo pipefail

MODEL_CONFIG=${1:-configs/model/debug.yaml}
CHECKPOINT=${2:?"ERROR: checkpoint path required"}
shift 2
BENCHMARKS=${@:-hellaswag mmlu}

echo "=== Qwen3 Evaluation ==="
echo "Model config: $MODEL_CONFIG"
echo "Checkpoint:   $CHECKPOINT"
echo "Benchmarks:   $BENCHMARKS"
echo "========================="

python train.py eval \
    --model-config "$MODEL_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --benchmarks $BENCHMARKS
