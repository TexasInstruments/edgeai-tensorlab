#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

model=mobilenet_v3_large_lite
#model=mobilenet_v2
#model=mobilenet_v2_lite
#model=mobilenet_v3_small_lite
#model=resnet18
#model=resnet50
#model=regnetx200mf
#model=regnetx400mf
#model=regnetx400mf
#model=regnetx800mf
#model=regnetx1p6gf


torchrun --nproc_per_node 4 ./references/classification/train.py --data-path ./data/datasets/imagenet/ --model ${model} --export
