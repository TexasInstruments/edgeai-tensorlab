#!/usr/bin/env bash

current_time=$(date '+%Y%d%m_%H%M%S')

model=resnet50

torchrun --nproc_per_node 4 ./references/classification/train.py --data-path ./data/datasets/imagenet --model ${model} --output-dir ./data/checkpoints/${current_time}_imagenet_classification_${model}



