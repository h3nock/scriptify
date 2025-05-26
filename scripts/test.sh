#!/usr/bin/env bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <text to predict>"
  exit 1
fi

input_text="$*"

# find the latest run_* directory
LATEST_RUN_DIR=$(ls -v -d outputs/run_* 2>/dev/null | tail -n 1)

if [ -z "$LATEST_RUN_DIR" ] || [ ! -d "$LATEST_RUN_DIR" ]; then
  echo "Error: No run directories found in 'outputs'."
  exit 1
fi

echo "Using latest run directory: $LATEST_RUN_DIR"

# find the latest checkpoint within the latest run directory
LATEST_CHECKPOINT=$(ls -v "$LATEST_RUN_DIR"/checkpoints/model-* 2>/dev/null | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ] || [ ! -f "$LATEST_CHECKPOINT" ]; then
  echo "Error: No model checkpoints found in '$LATEST_RUN_DIR/checkpoints'."
  exit 1
fi

echo "Using latest checkpoint: $LATEST_CHECKPOINT"

CUDA_LAUNCH_BLOCKING=1 python3 -m src.predict \
  --checkpoint "$LATEST_CHECKPOINT" \
  --text "$input_text" --bias 0.7