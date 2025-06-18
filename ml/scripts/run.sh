#!/usr/bin/env bash
RUN_NAME=${1}
RUN_NAME_ARG=""

if [ -n "$RUN_NAME" ]; then
    echo "Using run name: $RUN_NAME" 
    RUN_NAME_ARG="--run_name $RUN_NAME"
fi

echo "Starting single GPU training..."

# move to ml dir 
cd "$(dirname "$0")/.."

# set single GPU
export CUDA_VISIBLE_DEVICES=0

# run torchrun for single GPU
torchrun --nnodes=1 --nproc_per_node=1 -m src.train $RUN_NAME_ARG

echo "Training finished with exit code $?"