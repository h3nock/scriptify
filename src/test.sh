#!/usr/bin/env bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <text to predict>"
  exit 1
fi

input_text="$*"

# find the latest checkpoint
LATEST_CHECKPOINT=$(ls -v checkpoints/model-* 2>/dev/null | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ] || [ ! -f "$LATEST_CHECKPOINT" ]; then
  echo "Error: No model checkpoints found in the 'checkpoints' directory."
  exit 1
fi

echo "Using latest checkpoint: $LATEST_CHECKPOINT"

CUDA_LAUNCH_BLOCKING=1 python3 -m src.predict \
  --checkpoint "$LATEST_CHECKPOINT" \
  --text "$input_text" --bias 0.7