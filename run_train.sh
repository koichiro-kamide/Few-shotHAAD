#!/bin/bash

python -m src.framework.train \
    --run_mode train \
    --cuda_index 0 \
    --num_augment 1 \
    --batch_size 18 \
    --num_epochs 101 \
    --lr 0.001 \
    --snapshot 1 \