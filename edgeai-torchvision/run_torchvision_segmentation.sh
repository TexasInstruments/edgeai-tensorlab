#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH


model=deeplabv3_mobilenet_v3_lite_large
#model=deeplabv3plus_mobilenet_v2_lite
#model=deeplabv3plus_mobilenet_v3_lite_large
#model=deeplabv3plus_mobilenet_v3_lite_small
#model=lraspp_mobilenet_v3_lite_large

python ./references/segmentation/segmentation_main.py --model ${model}
