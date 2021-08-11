#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

# ssdlite_mobilenet_v3_lite_large
# ssdlite_mobilenet_v3_lite_small
# ssdlite_mobilenet_v2_lite_fpn
# ssdlite_mobilenet_v3_lite_large_fpn
# ssdlite_mobilenet_v3_lite_small_fpn
# ssdlite_mobilenet_v2_lite_bifpn

python ./references/detection/detection_main.py --model ssdlite_mobilenet_v2_lite_bifpn

