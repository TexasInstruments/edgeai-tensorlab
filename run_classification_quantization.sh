#!/usr/bin/env bash


current_time=$(date '+%Y%m%d_%H%M%S')
model=mobilenet_v2 #mobilenet_v2 #resnet50
weights=MobileNet_V2_Weights.IMAGENET1K_V1 #MobileNet_V2_Weights.IMAGENET1K_V1 #ResNet50_Weights.IMAGENET1K_V1
output_dir=./data/checkpoints/${current_time}_imagenet_classification_${model}
epochs=10 #60 #16 #150 #90 #25
batch_size=64 #32 #128
lr=0.0001 #0.0001 #0.001
lr_scheduler=cosineannealinglr
lr_warmup_epochs=0
wd=0.00004 #0.0001

quantization=1
# Options: DEFAULT WT8_AT8 WC8_AT8 WT8SP2_AT8SP2 WC8SP2_AT8SP2 WC4_AT8 WC4R4_AT8 WC4_AT4 WC4R4_AT4R4
# can use two types with a + for mixed precision: "WC4_AT4+WC8_AT8" "WC4R4_AT4R4+WC8_AT8"
quantization_type="DEFAULT"

# needed when using --use-deterministic-algorithms which sets torch.use_deterministic_algorithms(True)
# but this may increase the memory required.
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

## distributed data parallel training
torchrun --nproc_per_node 4 ./references/classification/train.py --data-path ./data/datasets/imagenet \
         --epochs ${epochs} --batch-size ${batch_size} --wd=${wd} --weights=${weights} \
         --lr=${lr} --lr-scheduler ${lr_scheduler} --lr-warmup-epochs ${lr_warmup_epochs} \
         --model ${model} --output-dir ${output_dir} \
         --quantization=${quantization} --quantization-type=${quantization_type} \
         --train-epoch-size-factor 0.1


python3 ./references/classification/train.py --data-path ./data/datasets/imagenet \
         --batch-size ${batch_size}  \
         --model ${model} --output-dir "${output_dir}/test" \
         --quantization=${quantization} --quantization-type=${quantization_type} \
         --resume "${output_dir}/checkpoint.pth" \
         --test-only

