#!/usr/bin/env bash


current_time=$(date '+%Y%d%m_%H%M%S')
model=resnet50
quantization=1 #0
lr=0.0001 #0.1
epochs=8 #90
weights="ResNet50_Weights.IMAGENET1K_V1"
quantization_type=0

torchrun --nproc_per_node 4 ./references/classification/train.py --data-path ./data/datasets/imagenet \
         --lr=${lr} --epochs ${epochs} --quantization=${quantization} --weights=${weights} \
         --model ${model} --output-dir ./data/checkpoints/${current_time}_imagenet_classification_${model} \
         --quantization-type=${quantization_type}
