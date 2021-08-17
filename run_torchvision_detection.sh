#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

#model=ssdlite_mobilenet_v2_lite_fpn
#model=ssdlite_mobilenet_v2_lite_bifpn
model=ssdlite_mobilenet_v3_lite_large
#model=ssdlite_mobilenet_v3_lite_small
#model=ssdlite_mobilenet_v3_lite_large_fpn
#model=ssdlite_mobilenet_v3_lite_small_fpn


python ./references/detection/detection_main.py --model ${model}

