#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH


# lraspp_mobilenet_v3_lite_large
# deeplabv3dws_mobilenet_v3_lite_large
# deeplabv3plusdws_mobilenet_v2_lite
# deeplabv3plusdws_mobilenet_v3_lite_large
# deeplabv3plusdws_mobilenet_v3_lite_small

python ./references/segmentation/segmentation_main.py --model deeplabv3plusdws_mobilenet_v2_lite
