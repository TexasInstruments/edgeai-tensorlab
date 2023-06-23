#!/usr/bin/env bash


current_time=$(date '+%Y%d%m_%H%M%S')
model=mobilenet_v2 #resnet50
lr=0.001 #0.1
wd=0.00004 #0.0001
epochs=16 #90
weights="MobileNet_V2_Weights.IMAGENET1K_V1" #"ResNet50_Weights.IMAGENET1K_V1"
quantization=QAT
quantization_type=8BIT_PERCH #8BIT_PERT #8BIT_PERT #8BIT_PERCH #8BIT_PERT_SYM_P2


torchrun --nproc_per_node 4 ./references/classification/train.py --data-path ./data/datasets/imagenet \
         --batch-size 128 --lr=${lr} --wd=${wd} --epochs ${epochs} --weights=${weights} \
         --model ${model} --output-dir ./data/checkpoints/${current_time}_imagenet_classification_${model} \
         --quantization=${quantization} --quantization-type=${quantization_type}
