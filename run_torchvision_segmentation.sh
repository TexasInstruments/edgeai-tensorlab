#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

#model=deeplabv3_mobilenet_v3_large_lite
#model=deeplabv3_mobilenet_v2_lite
model=deeplabv3plus_mobilenet_v3_large_lite
#model=deeplabv3plus_mobilenet_v2_lite
#model=lraspp_mobilenet_v3_large_lite


torchrun --nproc_per_node 4 ./references/segmentation/train.py --data-path ./data/datasets/coco/images  --model ${model} --epochs=90 -b=4 --gpus=4 --export
