#!/usr/bin/env bash


current_time=$(date '+%Y%d%m_%H%M%S')
model=resnet50
quantization=0
lr=0.1
epochs=90
quantization_type=4

torchrun --nproc_per_node 4 ./references/classification/train.py --data-path ./data/datasets/imagenet \
         --lr=${lr} --epochs ${epochs} --quantization=${quantization} \
         --model ${model} --output-dir ./data/checkpoints/${current_time}_imagenet_classification_${model}

