#!/usr/bin/env bash

echo "Starting single GPU training..."

# move to ml dir 
cd "$(dirname "$0")/.."

# set single GPU
export CUDA_VISIBLE_DEVICES=0

# run torchrun for single GPU
torchrun --nnodes=1 --nproc_per_node=1 -m src.train

echo "Training finished with exit code $?"