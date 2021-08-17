#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

#model=resnet18
#model=resnet50
#model=regnetx200mf
#model=regnetx400mf
#model=regnetx400mf
#model=regnetx800mf
#model=regnetx1p6gf
#model=mobilenet_v2
#model=mobilenet_v2_lite
model=mobilenet_v3_lite_large
#model=mobilenet_v3_lite_small

python ./references/classification/train.py --data-path ./data/datasets/imagenet/ --model ${model} --pretrained --export
