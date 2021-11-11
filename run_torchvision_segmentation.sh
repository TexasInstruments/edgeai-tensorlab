#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH


model=deeplabv3_mobilenet_v3_lite_large
#model=deeplabv3plus_mobilenet_v2_lite
#model=deeplabv3plus_mobilenet_v3_lite_large
#model=deeplabv3plus_mobilenet_v3_lite_small
#model=lraspp_mobilenet_v3_lite_large

python3 -m torch.distributed.run --nproc_per_node 4 ./references/segmentation/train.py --model deeplabv3_mobilenet_v3_large_lite --epochs=60 --batch-size=8 --gpus=4
