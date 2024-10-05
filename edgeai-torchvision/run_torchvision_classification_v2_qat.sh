#!/usr/bin/env bash

# torch.fx based model surgery and training

# PYTHONPATH must start with a : to be able to load local modules
# but this can cause a confusion the installed torchvision and the local torchvision
# export PYTHONPATH=:$PYTHONPATH

# Date/time in YYYYMMDD-HHmmSS format
DATE_TIME=`date +'%Y%m%d-%H%M%S'`

#=========================================================================================
# sample models that can be used
#model=resnet50
#model=mobilenet_v2
#model=mobilenet_v2
#model=resnet18
#model=regnetx200mf
#model=regnetx400mf
#model=regnetx400mf
#model=regnetx800mf
#model=regnetx1p6gf

# these lite models are created using model surgery from models in torchvision
# these lite models will be available only if --model-surgery <argument> argument is set to one of these
# --model-surgery 1: legacy module based surgery
# --model-surgery 2: advanced model surgery with torch.fx (to be released)
#model=mobilenet_v3_large_lite
#model=mobilenet_v3_small_lite
model=mobilenet_v2_lite

#=========================================================================================
# set the appropriate pretrained weights for the above model
#model_weights="MobileNet_V2_Weights.IMAGENET1K_V1"
#model_weights="MobileNet_V2_Weights.IMAGENET1K_V2"
#model_weights="ResNet50_Weights.IMAGENET1K_V1"
model_weights="../edgeai-modelzoo/models/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v2_lite_wt-v2_20231101_checkpoint.pth"

output_dir="./data/checkpoints/torchvision/${DATE_TIME}_imagenet_classification_${model}"

val_resize_size=232 #256 #232
val_crop_size=224

# --quantization-type can be one of: WT8SP2_AT8SP2, WC8_AT8
# WC8_AT8SP2 would mean : 
#          Weight     -   channel-wise 8-bit quantized
#          Activation -   tensor-wise 8-bit quantized with a power-2 scale 

#==================================QAT=====================================================
# command="./references/classification/train.py --data-path=./data/datasets/imagenet \
# --epochs=25 --batch-size=64 --wd=4e-5 --lr=0.0001 --lr-scheduler=cosineannealinglr --lr-warmup-epochs=1 \
# --model=${model} --model-surgery=2 --quantization=2 --quantization-type=WT8SP2_AT8SP2 --quantization-method=QAT \
# --train-epoch-size-factor=0.2 --opset-version=17 --val-resize-size=$val_resize_size --val-crop-size=$val_crop_size"

#==================================PTC=====================================================
command="./references/classification/train.py --data-path=./data/datasets/imagenet \
--epochs=3 --batch-size=64 --wd=4e-5 --lr=0.0001 --lr-scheduler=cosineannealinglr --lr-warmup-epochs=1 \
--model=${model} --model-surgery=2 --quantization=2 --quantization-type=MSA_WC8_AT8 --quantization-method=PTC \
--quantize-calib-images=100 --opset-version=17 --val-resize-size=$val_resize_size --val-crop-size=$val_crop_size"

# training: single GPU (--device=cuda:0)or CPU (--device=cpu) run
# python3 ${command} --weights=${model_weights} --output-dir=${output_dir}

# training: multi-gpu distributed data parallel
torchrun --nproc_per_node 4 ${command} --weights=${model_weights} --output-dir=${output_dir}

# testing after the training
# torchrun --nproc_per_node 4 ${command} --test-only --weights=${output_dir}/checkpoint.pth --output-dir=${output_dir}/test
