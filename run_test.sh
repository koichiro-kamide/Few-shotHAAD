#!/bin/bash

# make list variable
prior_model="actions"
# actions=("warm_up" "walk" "run" "jump" "drink" "lift_dumbbell" "sit" "eat" "turn steering wheel" "phone" "boxing" "throw")
actions=("warm_up")

for item in "${actions[@]}"
do
    python -m src.framework.test \
    --run_mode test \
    --seed 0 \
    --cuda_index 0 \
    --num_support 3 \
    --num_augment 1 \
    --batch_size 20 \
    --num_epochs 1 \
    --normal_action "$item"
done
# --checkpoint_path checkpoints/20251229_034100/encoder101.p \
