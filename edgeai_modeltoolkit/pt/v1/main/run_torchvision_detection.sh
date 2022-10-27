#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

model_url_base="https://software-dl.ti.com/jacinto7/esd/modelzoo/common/models/vision/classification/imagenet1k/edgeai-tv"


##################################################################################################
# uncomment only one set of lines to train
# A sample of models supported and pretrained backbone checkpoints that can be used to train them
# set pretrained_backbone to True to use the backbone chekpoints defined by torchvision or provide a url or path

model=ssdlite_mobilenet_v2_fpn_lite
pretrained_backbone=True

#model=ssdlite_mobilenet_v3_large_fpn_lite
#pretrained_backbone=${model_url_base}/'mobilenet_v3_lite_large_20210507_checkpoint.pth'

#model=ssdlite_mobilenet_v3_small_fpn_lite
#pretrained_backbone=${model_url_base}/'mobilenet_v3_lite_small_20210429_checkpoint.pth'

#model=ssdlite_mobilenet_v3_large_lite
#pretrained_backbone=${model_url_base}/'mobilenet_v3_lite_large_20210507_checkpoint.pth'

# torchvision has backbone checkpoint defined for these backbones, so just set to True
#model=ssdlite_regnet_x_800mf_fpn_lite
#pretrained_backbone=True

#model=ssdlite_regnet_x_1_6gf_fpn_lite
#pretrained_backbone=True

# torchvision has backbone checkpoint defined for original (non-lite) models, we shall use them for time being.
#model=ssdlite_efficientnet_b0_fpn_lite
#pretrained_backbone=True

#model=ssdlite_efficientnet_b2_fpn_lite
#pretrained_backbone=True

#model=ssdlite_efficientnet_b0_bifpn_lite
#pretrained_backbone=True

#model=ssdlite_efficientnet_b2_bifpn_lite
#pretrained_backbone=True

##################################################################################################


# The multi-gpu training/test can be run using one of several methods
# 1. for cpu based training, specify --device cpu
# 1. elastic launch using torchrun (torch.distributed.run) or torch.distributed.launch with --nproc_per_node <num_gpus> <trainign script> <args...>
# 2. this script can launch torch.multiprocess internally (i.e. without using torch.distributed.run), if you set --distributed=True --gpus <num_gpus>

# training : using torch.distributed.run
torchrun --nproc_per_node 4 ./references/detection/train.py --model ${model} --epochs=240 --batch-size=8 \
--pretrained-backbone ${pretrained_backbone}
# alternative launch method supported by this script : using torch.multiprocess internally to launch processes
#python3 ./references/detection/train.py --model ${model} --epochs=240 --batch-size=8 --gpus 4 \
# --pretrained-backbone ${pretrained_backbone}

# test
#torchrun --nproc_per_node 4 ./references/detection/train.py --model ${model} --epochs=240 --batch-size=8 --gpus=4 \
# --pretrained ./data/checkpoints/detection/coco_${model}/checkpoint.pth --test-only

# export
#python3 ./references/detection/train.py --model ${model} --pretrained ./data/checkpoints/detection/coco_${model}/checkpoint.pth --export-only
