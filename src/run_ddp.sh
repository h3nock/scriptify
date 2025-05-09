#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 -m src.train > training_log.txt 2>&1