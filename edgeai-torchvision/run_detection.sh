#!/usr/bin/env bash

# PYTHONPATH must start with a : to be able to load local modules
# but this can cause a confusion the installed torchvision and the local torchvision
# export PYTHONPATH=:$PYTHONPATH

# Date/time in YYYYMMDD-HHmmSS format
DATE_TIME=`date +'%Y%m%d-%H%M%S'`

#=========================================================================================
# these lite models will be available only if --model-surgery <argument> argument is set
# --model-surgery 1: legacy module based surgery
# --model-surgery 2: advanced model surgery with torch.fx (to be released)

#models and weights
model=ssdlite320_mobilenet_v3_large
weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

# model=retinanet_resnet50_fpn
# weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1

# model=retinanet_resnet50_fpn_v2
# weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1

#=========================================================================================
output_dir="./data/checkpoints/torchvision/coco_detection_${model}"

#=========================================================================================
torchrun --nproc_per_node 4 ./references/detection/train.py --data-path ./data/datasets/coco --model ${model} \
--epochs=1 --aspect-ratio-group-factor 3 --lr-scheduler cosineannealinglr --lr 0.15 \
--batch-size 16 --weight-decay 0.00004 --data-augmentation ssdlite \
--model-surgery 1 --output-dir=${output_dir} --weights=${weights}

onnx_file="${output_dir}/${model}.onnx"
python ./scripts/simplify_onnx.py --onnx_path=${onnx_file}
