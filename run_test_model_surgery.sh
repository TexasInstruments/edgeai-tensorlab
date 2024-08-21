#!/usr/bin/env bash

# torch.fx based model surgery and training

# PYTHONPATH must start with a : to be able to load local modules
# but this can cause a confusion the installed torchvision and the local torchvision
# export PYTHONPATH=:$PYTHONPATH

# Date/time in YYYYMMDD-HHmmSS format
DATE_TIME=`date +'%Y%m%d-%H%M%S'`

#=========================================================================================
# set the appropriate pretrained weights and  other parameters for the model.

dataset_path=${1:-"../datasets/imagenet1k   "}
# edgeai-model-zoo weights directory path 
# before running the script download edgeai-modelzoo repository either in same directory as edgeai-torchvision or modify the path below to correct corresponding directory
WEIGHTS_DIR=${2:-"../edgeai-modelzoo"}"/models/vision/classification/imagenet1k/edgeai-tv2"
models=( \
# "mobilenet_v3_large"
# "mobilenet_v2"
"mobilenet_v3_large_lite" \
# "mobilenet_v3_small_lite" \
"mobilenet_v2_lite"
)

ORIGINAL_ACCURACY=(\
# 74.042 \
# 71.878 \
71.732 \
# 67.668 \
72.752 \
)

#=========================================================================================
# set the appropriate pretrained weights for the above model
#model_weights="ResNet50_Weights.IMAGENET1K_V1"
#model_weights="MobileNet_V2_Weights.IMAGENET1K_V1"
model_weights=(\
# "MobileNet_V3_Large_Weights.IMAGENET1K_V1" \
# "MobileNet_V2_Weights.IMAGENET1K_V1" \
"${WEIGHTS_DIR}/mobilenet_v3_large_lite_wt-v2_20231011_checkpoint.pth" \
# "/home/a0507161/Kunal/checkpoints/mobilenet_v3_lite_small_20210429_checkpoint.pth" \
"${WEIGHTS_DIR}/mobilenet_v2_lite_wt-v2_20231101_checkpoint.pth" \
)


val_resize_size=232 #256 #232
val_crop_size=224
gpus=4
batch_sizes=( \
# 2048 \
# 2048 \
2048 \
# 2048 \
2048 \
)
out_dir="./data/checkpoints/torchvision/imagenet_classification/${DATE_TIME}"
result_path="${out_dir}/result.csv"

#=========================================================================================
#Run Evaluation and report generaing script for all models
for i in ${!models[@]}; do
    model=${models[$i]}
    command="./scripts/test_model_surgery.py --data-path=${dataset_path} --gpus=$gpus \
    --batch-size=${batch_sizes[$i]} --model=${model} \
    --original-accuracy=${ORIGINAL_ACCURACY[$i]} \
    --opset-version=18 --val-resize-size=$val_resize_size --val-crop-size=$val_crop_size"
    output_dir="${out_dir}/${model}"
    echo $model
    echo $command --weights=${model_weights[$i]} --output-dir=${output_dir} --result-path=${result_path}
    python $command --weights=${model_weights[$i]} --output-dir=${output_dir} --result-path=${result_path}
    echo "${i} Done"
done