#!/usr/bin/env bash


current_time=$(date '+%Y%m%d_%H%M%S')
model=mobilenet_v2 #resnet50
epochs=150
batch_size=128
lr=0.1
lr_scheduler=cosineannealinglr
lr_warmup_epochs=5
wd=0.00004 #0.0001

torchrun --nproc_per_node 4 ./references/classification/train.py --data-path ./data/datasets/imagenet \
         --epochs ${epochs} --batch-size ${batch_size} --wd=${wd} --weights=${weights} \
         --lr=${lr} --lr-scheduler ${lr_scheduler} --lr-warmup-epochs ${lr_warmup_epochs} \
         --model ${model} --output-dir ./data/checkpoints/${current_time}_imagenet_classification_${model}

