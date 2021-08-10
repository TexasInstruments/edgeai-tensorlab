#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH


python ./references/classification/train.py --data-path ./data/datasets/imagenet/ --model mobilenet_v3_small --pretrained --export
