#!/usr/bin/env bash


# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

model_url_base="https://software-dl.ti.com/jacinto7/esd/modelzoo/common/models/vision/classification/imagenet1k/edgeai-tv"


##################################################################################################
# A sample of models supported and pretrained backbone checkpoints that can be used to train them
# uncomment only one set of lines to train

model=ssdlite_mobilenet_v2_fpn_lite
backbone_checkpoint=${model_url_base}/'mobilenet_v2_20191224_checkpoint.pth'

#model=ssdlite_mobilenet_v3_large_fpn_lite
#backbone_checkpoint=${model_url_base}/'mobilenet_v3_lite_large_20210507_checkpoint.pth'

#model=ssdlite_mobilenet_v3_small_fpn_lite
#backbone_checkpoint=${model_url_base}/'mobilenet_v3_lite_small_20210429_checkpoint.pth'

#model=ssdlite_regnet_x_800mf_fpn_lite
#backbone_checkpoint=True

#model=ssdlite_regnet_x_1_6gf_fpn
#backbone_checkpoint=True

#model=ssdlite_efficientnet_b0_fpn_lite
#backbone_checkpoint=True

#model=ssdlite_efficientnet_b0_bifpn_lite
#backbone_checkpoint=True

#model=ssdlite_mobilenet_v3_large_lite
#backbone_checkpoint=${model_url_base}/'mobilenet_v3_lite_large_20210507_checkpoint.pth'
##################################################################################################


# The multi-gpu training/test can be run using one of several methods
# 1. for cpu based training, specify --device cpu
# 1. elastic launch using torchrun (torch.distributed.run) or torch.distributed.launch with --nproc_per_node <num_gpus> <trainign script> <args...>
# 2. this script can launch torch.multiprocess internally (i.e. without using torch.distributed.run), if you set --distributed=True --gpus <num_gpus>

# training : using torch.distributed.run
torchrun --nproc_per_node 4 ./references/detection/train.py --model ${model} --epochs=240 --batch-size=8 \
--pretrained-backbone ${backbone_checkpoint}
# alternative launch method supported by this script : using torch.multiprocess internally to launch processes
#python3 ./references/detection/train.py --model ${model} --epochs=240 --batch-size=8 --gpus 4 \
# --pretrained-backbone ${backbone_checkpoint}

# test
#torchrun --nproc_per_node 4 ./references/detection/train.py --model ${model} --epochs=240 --batch-size=8 --gpus=4 \
# --pretrained ./data/checkpoints/detection/coco_${model}/checkpoint.pth --test-only

# export
#python3 ./references/detection/train.py --model ${model} --pretrained ./data/checkpoints/detection/coco_${model}/checkpoint.pth --export-only
