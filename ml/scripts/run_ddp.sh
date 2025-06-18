#!/usr/bin/env bash

NODE_RANK=${1} 
RUN_NAME=${2}
RUN_NAME_ARG=""

if [ -z "$NODE_RANK" ]; then
    echo "Usage: $0 <node_rank>"
    echo "Example: $0 0"
    exit 1
fi

if [ -n "$RUN_NAME" ]; then
    echo "Using run name: $RUN_NAME" 
    RUN_NAME_ARG="--run_name $RUN_NAME"
fi
    
echo "Starting multi-node distributed training..."
echo "Node rank: $NODE_RANK, Run name: $RUN_NAME"

# cd to ml directory
cd "$(dirname "$0")/.."

# load environment variables from .env
if [ -f .env ]; then
    echo "Loading configuration from .env..."
    set -a
    source .env
    set +a
    
    MASTER_ADDR=${SCRIPTIFY_DIST_MASTER_ADDR}
    MASTER_PORT=${SCRIPTIFY_DIST_MASTER_PORT}
    NNODES=${SCRIPTIFY_DIST_NNODES}
    NPROC_PER_NODE=${SCRIPTIFY_DIST_NPROC_PER_NODE}
else
    echo "Error: .env file not found"
    exit 1
fi

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Total nodes: $NNODES, Processes per node: $NPROC_PER_NODE"

# load modules
module load Anaconda python39 cuda11.8
source activate scriptify_env

export CUDA_VISIBLE_DEVICES=0,1

# run torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m src.train $RUN_NAME_ARG 

echo "Training finished with exit code $?"