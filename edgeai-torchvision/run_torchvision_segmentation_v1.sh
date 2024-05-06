#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

# these lite models will be available only if --model-surgery <argument> argument is set
# --model-surgery 1: legacy module based surgery
# --model-surgery 2: advanced model surgery with torch.fx (to be released)
model=deeplabv3_mobilenet_v3_large_lite
#model=deeplabv3plus_mobilenet_v3_large_lite
#model=lraspp_mobilenet_v3_large_lite


torchrun --nproc_per_node 4 ./references/segmentation/train.py --data-path ./data/datasets/coco --model ${model} --epochs=60 --batch-size=8 --model-surgery 1
