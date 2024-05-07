#!/usr/bin/env bash

#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################
export PYTHONPATH=.:$PYTHONPATH

#python3 ./scripts/convert_dataset.py --source_format=cityscapes --source_anno /user/a0393608/work//code/ti/algoref/vision-dataset/annotatedJSON/tiscapes/data/gtFine --source_data=/user/a0393608/work//code/ti/algoref/vision-dataset/annotatedJSON/tiscapes/data/leftImg8bit --dest_anno=/user/a0393608/work//code/ti/algoref/vision-dataset/annotatedJSON/tiscapes/data/annotations/instances.json

#python3 ./scripts/convert_dataset.py --source_format=image_splits --source_data=./data/examples/datasets/potatatochips_classification/images --dest_anno=./data/examples/datasets/potatatochips_classification/annotations/instances.json

#python3 ./scripts/convert_dataset.py --source_format=labelstudio_detection --source_anno=./data/examples/datasets/animal_detection/annotations/instances_labelstudio-detection-json-min.json --source_data=./data/examples/datasets/animal_detection/images --dest_anno=./data/examples/datasets/animal_detection/annotations/instances.json

#python3 ./scripts/convert_dataset.py --source_format=labelstudio_classification --source_anno=./data/examples/datasets/animal_classification/annotations/labels_labelstudio-classification-json-min.json --source_data=./data/examples/datasets/animal_classification/images --dest_anno=./data/examples/datasets/animal_classification/annotations/labels.json

#python3 ./scripts/convert_dataset.py --source_format=coco_splits --source_anno=./data/examples/datasets/coco_detection/annotations/instances_train2017.json,./data/examples/datasets/coco_detection/annotations/instances_val2017.json --source_data=./data/examples/datasets/coco_detection/train2017,./data/examples/datasets/coco_detection/val2017 --dest_anno=./data/examples/datasets/coco_detection/annotations/instances.json

# ---------------------------For Segmentation -----------------------------------
#--------- Model Maker folder structure formatting-------------------------------
#python3 ./scripts/convert_dataset.py \
# --source_format=modelmaker_format \
# --input_dataset_path="/home/a0504871/Downloads/labelstudio_extracted"
#--------------------------------------------------------------------------------

#-------- Sorting the annotation file based on the preference order--------------
#python3 ./scripts/convert_dataset.py \
#--source_format=sort_annotations \

#--annotation_file_path="./data/downloads/tiscapes2017_driving/annotations/stuff.json" \
#--preference_order="road,vehicle,trafficsign,human"
#---------------------------------------------------------------------------------

#-------- Sorting the annotation file based on the preference order--------------
#python3 ./scripts/convert_dataset.py \
#--source_format=sort_annotations \
#--annotation_file_path="/home/a0504871/Downloads/dataset/annotations/instances.json" \
#--preference_order="vehicle,trafficsign,human"

#--annotation_file_path="/home/a0504871/work/ti/edgeai-algo/edgeai-modelmaker/data/projects/tiscapes2017_driving_det/dataset/annotations/stuff.json" \
#--preference_order="road,vehicle,trafficsign,human"
#---------------------------------------------------------------------------------