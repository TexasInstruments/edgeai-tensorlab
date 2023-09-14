#!/usr/bin/env bash

# torch.fx based model surgery and training

# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

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

# these lite models will be available only if --model-surgery <argument> argument is set
# --model-surgery 1: legacy module based surgery
# --model-surgery 2: advanced model surgery with torch.fx (to be released)
#model=mobilenet_v2_lite
#model=mobilenet_v3_large_lite
#model=mobilenet_v3_small_lite

#=========================================================================================
# set the appropriate pretrained weights for the above model
#model_weights="MobileNet_V2_Weights.IMAGENET1K_V1"
#model_weights="ResNet50_Weights.IMAGENET1K_V1"

#=========================================================================================
model=mobilenet_v2
model_weights="MobileNet_V2_Weights.IMAGENET1K_V1"

#=========================================================================================
command="./references/classification/train.py --data-path=./data/datasets/imagenet --output-dir=./data/checkpoints/imagenet_classification_${model} \
--epochs=10 --batch-size=32 --wd=0.00004 --lr=0.0001 --lr-scheduler=cosineannealinglr --lr-warmup-epochs=0 \
--model=${model} --weights=${model_weights} --model-surgery=0 --quantization=2 --quantization-type=DEFAULT \
--train-epoch-size-factor=0.2 --opset-version=13"

# single GPU (--device=cuda:0)or CPU (--device=cpu) run
#python ${command}

# multi-gpu distributed data parallel
torchrun --nproc_per_node 4 ${command}

