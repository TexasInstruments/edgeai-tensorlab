#!/usr/bin/env bash


current_time=$(date '+%Y%m%d_%H%M%S')
model=mobilenet_v2 #resnet50
weights=MobileNet_V2_Weights.IMAGENET1K_V1 #ResNet50_Weights.IMAGENET1K_V1
epochs=60 #16 #150 #90
batch_size=128 #64 #32
lr=0.001 #0.0001 #0.001
lr_scheduler=cosineannealinglr
lr_warmup_epochs=0
wd=0.00004 #0.0001
quantization=1
#Options: DEFAULT W8T_A8T W8C_A8T W8T_A8T_P2 W8C_A8T_P2 W4C_A8T W4C_A4T W4C_A4T_RR4
# can use two types with a + for mixed precision: W4C_A4T_RR4+W8C_A8T
quantization_type="W4C_A4T_RR4+W8C_A8T"

torchrun --nproc_per_node 4 ./references/classification/train.py --data-path ./data/datasets/imagenet \
         --epochs ${epochs} --batch-size ${batch_size} --wd=${wd} --weights=${weights} \
         --lr=${lr} --lr-scheduler ${lr_scheduler} --lr-warmup-epochs ${lr_warmup_epochs} \
         --model ${model} --output-dir ./data/checkpoints/${current_time}_imagenet_classification_${model} \
         --quantization=${quantization} --quantization-type=${quantization_type} \
         --train-epoch-size-factor 0.1
