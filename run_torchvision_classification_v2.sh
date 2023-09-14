#!/usr/bin/env bash

# torch.fx based model surgery and training

# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

# sample models that can be used
model=mobilenet_v2
#model=mobilenet_v2
#model=resnet18
#model=resnet50
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

torchrun --nproc_per_node 4 ./references/classification/train.py --data-path ./data/datasets/imagenet/ --model ${model} --model-surgery 2


