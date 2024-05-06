#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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

# Summary of commands - uncomment one and run this script
#### For the datasets in sections marked as "Automatic Download", dataset will be downloaded automatically downloaded before training begins. For "Manual Download", it is expected that it is manually downloaded and kept in the folder specified agaianst the --data_path option.

## =====================================================================================
## Training
## =====================================================================================
#### KITTI Depth (Manual Download) - Training with MobileNetV2+DeeplabV3Lite
#python3 ./references/edgeailite/main/train_depth_main.py --dataset_name kitti_depth --model_name deeplabv3plus_mobilenetv2_tv_edgeailite --data_path ./data/datasets/kitti/kitti_depth/data --img_resize 384 768 --output_size 374 1242 \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

#### KITTI Depth (Manual Download) - Training with ResNet50+FPN
#python3 ./references/edgeailite/main/train_depth_main.py --dataset_name kitti_depth --model_name fpn_edgeailite_aspp_resnet50 --data_path ./data/datasets/kitti/kitti_depth/data --img_resize 384 768 --output_size 374 1242 \
#--pretrained https://download.pytorch.org/models/resnet50-19c8e357.pth

## =====================================================================================
## Validation
## =====================================================================================
#### KITTI Depth (Manual Download) - Validation - populate the pretrained path in ??
#python3 ./references/edgeailite/main/train_depth_main.py --phase validation --dataset_name kitti_depth --model_name deeplabv3plus_mobilenetv2_tv_edgeailite --data_path ./data/datasets/kitti/kitti_depth/data --img_resize 384 768 --output_size 374 1242 \
#--pretrained=???

#### KITTI Depth (Manual Download) - Inference - populate the pretrained filename in ??
#python3 ./references/edgeailite/main/infer_depth_main.py --dataset_name kitti_depth --model_name deeplabv3plus_mobilenetv2_tv_edgeailite --data_path ./data/datasets/kitti/kitti_depth/data --img_resize 384 768 --output_size 374 1242 \
#--pretrained ???

