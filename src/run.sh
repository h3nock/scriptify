#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python -m src.train > training_log.txt 2>&1