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
#### Manual Download: It is expected that the dataset is manually downloaded and kept in the folder specified agaianst the --data_path option.

## =====================================================================================
# Extra Flags
## =====================================================================================
EXTRA_OPTIONS="" #"--enable_fp16 True"

## =====================================================================================
# Models Supported:
## =====================================================================================
# fpn_aspp_regnetx800mf_edgeailite: uses regnetx800mf encoder and group_width according to that (even in the decoder)
# unet_aspp_regnetx800mf_edgeailite: uses regnetx800mf encoder and group_width according to that (even in the decoder)
# deeplabv3plus_regnetx800mf_edgeailite: uses regnetx800mf encoder and group_width according to that (even in the decoder)
#
# deeplabv3plus_mobilenetv2_tv_edgeailite: deeplabv3plus_edgeailite decoder
# fpn_aspp_mobilenetv2_tv_edgeailite: fpn decoder
# unet_aspp_mobilenetv2_tv_edgeailite: unet decoder
# deeplabv3plus_mobilenetv2_tv_fd_edgeailite: low complexity model with fast downsampling strategy
# fpn_aspp_mobilenetv2_tv_fd_edgeailite: low complexity model with fast downsampling strategy
# unet_aspp_mobilenetv2_tv_fd_edgeailite: low complexity model with fast downsampling strategy
#
# deeplabv3plus_resnet50_edgeailite: uses resnet50 encoder
# deeplabv3plus_resnet50_p5_edgeailite: low complexity model - uses resnet50 encoder with half the number of channels (1/4 the complexity).
# note this need specially trained resnet50 pretrained weights
# fpn_aspp_resnet50_fd_edgeailite: low complexity model - with fast downsampling strategy


## =====================================================================================
## Training
## =====================================================================================
#RegNetX based Models
#Note: to use BGR input, set: --input_channel_reverse True, for RGB input ommit this argument or set it to False.
#------------------------
## Cityscapes Semantic Segmentation - Training with RegNetX800MF+DeeplabV3Lite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name deeplabv3plus_regnetx800mf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth --batch_size 16 \
#--optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}

### Cityscapes Semantic Segmentation - Training with RegNetX800MF+UNetEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name unet_aspp_regnetx800mf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth --batch_size 16 \
#--optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}

### Cityscapes Semantic Segmentation - Training Training with RegNetX400MF+FPNEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_regnetx400mf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth --batch_size 16 \
#--optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}

### Cityscapes Semantic Segmentation - Training with RegNetX800MF+FPNEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_regnetx800mf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth --batch_size 16 \
#--optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}

## Higher Resolution - 1024x512 - regnetx1.6gf
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_regnetx1p6gf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 512 1024 --output_size 1024 2048 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth --batch_size 16 \
#--optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}
#
## Higher Resolution - 1024x512 - regnetx3.2gf
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_regnetx3p2gf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 512 1024 --output_size 1024 2048 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth --batch_size 16 \
#--optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}

## Higher Resolution - 1536x768 - regnetx400mf
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_regnetx400mf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 768 1536 --rand_crop 512 1024 --output_size 1024 2048 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth --batch_size 16 \
#--optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}

## Higher Resolution - 1536x768 - regnetx3.2gf
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_regnetx3p2gf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 768 1536 --rand_crop 512 1024 --output_size 1024 2048 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth --batch_size 16 \
#--optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}



#MobileNetV2 based Models
#------------------------
#### Cityscapes Semantic Segmentation - Training with MobileNetV2+DeeplabV3Lite
python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name deeplabv3plus_mobilenetv2_tv_edgeailite \
--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth ${EXTRA_OPTIONS}

#### Cityscapes Semantic Segmentation - Training with MobileNetV2+FPNEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth ${EXTRA_OPTIONS}

#### Cityscapes Semantic Segmentation - Training with MobileNetV2+UNetEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name unet_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth ${EXTRA_OPTIONS}

#Higher Resolution
#------------------------
#### Cityscapes Semantic Segmentation - Training with MobileNetV2+DeeplabV3Lite, Higher Resolution
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 768 1536 --rand_crop 512 1024 --output_size 1024 2048 --gpus 0 1 \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth ${EXTRA_OPTIONS}

#### Cityscapes Semantic Segmentation - original fpn - no aspp model, stride 64 model, Higher Resolution - Low Complexity Model
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name deeplabv3plus_mobilenetv2_tv_fd_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 768 1536 --rand_crop 512 1024 --output_size 1024 2048 --gpus 0 1 \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth ${EXTRA_OPTIONS}


#ResNet50 based Models
#------------------------
#### Cityscapes Semantic Segmentation - Training with ResNet50+DeeplabV3Lite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name deeplabv3plus_resnet50_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained https://download.pytorch.org/models/resnet50-19c8e357.pth ${EXTRA_OPTIONS}

#### Cityscapes Semantic Segmentation - Training with FD-ResNet50+FPN - High Resolution - Low Complexity Model
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name deeplabv3plus_edgeailite_resnet50_fd \
#--data_path ./data/datasets/cityscapes/data --img_resize 768 1536 --rand_crop 512 1024 --output_size 1024 2048 --gpus 0 1 \
#--pretrained https://download.pytorch.org/models/resnet50-19c8e357.pth ${EXTRA_OPTIONS}

#### Cityscapes Semantic Segmentation - Training with ResNet50_p5+DeeplabV3Lite (ResNet50 encoder with half the channels):
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name deeplabv3plus_resnet50_p5_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained "./data/modelzoo/pretrained/pytorch/imagenet_classification/jacinto_ai/resnet50-0.5_2018-07-23_12-10-23.pth" ${EXTRA_OPTIONS}

# ADE20K Segmentation training
#------------------------
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name ade20k_seg_class32 --model_name deeplabv3plus_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/ADEChallengeData2016 --img_resize 512 512 --output_size 768 768 --gpus 0 1 2 3  \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 5e-2 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name ade20k_seg_class32 --model_name fpn_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/ADEChallengeData2016 --img_resize 512 512 --output_size 768 768 --gpus 0 1 2 3  \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 5e-2 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name ade20k_seg_class32 --model_name unet_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/ADEChallengeData2016 --img_resize 512 512 --output_size 768 768 --gpus 0 1 2 3  \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 5e-2 ${EXTRA_OPTIONS}


#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name ade20k_seg_class32 --model_name fpn_aspp_regnetx400mf_edgeailite \
#--data_path ./data/datasets/ADEChallengeData2016 --img_resize 384 384 --output_size 768 768 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name ade20k_seg_class32 --model_name fpn_aspp_regnetx800mf_edgeailite \
#--data_path ./data/datasets/ADEChallengeData2016 --img_resize 512 512 --output_size 768 768 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name ade20k_seg_class32 --model_name fpn_aspp_regnetx1p6gf_edgeailite \
#--data_path ./data/datasets/ADEChallengeData2016 --img_resize 512 512 --output_size 768 768 --gpus 0 1 2 3  \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name ade20k_seg_class32 --model_name fpn_aspp_regnetx3p2gf_edgeailite \
#--data_path ./data/datasets/ADEChallengeData2016 --img_resize 256 256 --output_size 768 768 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 1e-1 ${EXTRA_OPTIONS}


## =====================================================================================
## Validation
## =====================================================================================
##### Validation - Cityscapes Semantic Segmentation - Validation with RegNetX+DeeplabV3Lite - replace the pretrained checkpoint with what you have.
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase validation --dataset_name cityscapes_segmentation --model_name deeplabv3plus_regnetx800mf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/jai-modelzoo/pytorch/vision/segmentation/jai-devkit/cityscapes/deeplabv3plus_edgeailite_regnetx800mf_768x384_768x384_2020-08-08_18-47-29_checkpoint.pth
#
##### Validation - Cityscapes Semantic Segmentation - Inference with RegNetX+FPNEdgeAILite - replace the pretrained checkpoint with what you have.
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase validation --dataset_name cityscapes_segmentation --model_name fpn_aspp_regnetx800mf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/jai-modelzoo/pytorch/vision/segmentation/jai-devkit/cityscapes/fpn_aspp_regnetx_edgeailite800mf_768x384_2020-08-04_21-14-35_checkpoint.pth
#
##### Validation - VOC Segmentation - Validation with RegNetX+UNetEdgeAILite - replace the pretrained checkpoint with what you have.
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase validation --dataset_name cityscapes_segmentation --model_name unet_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/jai-modelzoo/pytorch/vision/segmentation/jai-devkit/cityscapes/unet_edgeailite_aspp_regnetx800mf_768x384_768x384_2020-08-08_18-47-54_checkpoint.pth
#
##### Validation - Cityscapes Semantic Segmentation - Validation with MobileNetV2+DeeplabV3Lite - replace the pretrained checkpoint with what you have.
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase validation --dataset_name cityscapes_segmentation --model_name deeplabv3plus_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/jai-modelzoo/pytorch/vision/segmentation/jai-devkit/cityscapes/deeplabv3plus_edgeailite_mobilenetv2_tv_768x384_2019-06-26-08-59-32_checkpoint.pth
#
##### Validation - Cityscapes Semantic Segmentation - Inference with MobileNetV2+FPNEdgeAILite - replace the pretrained checkpoint with what you have.
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase validation --dataset_name cityscapes_segmentation --model_name fpn_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/jai-modelzoo/pytorch/vision/segmentation/jai-devkit/cityscapes/fpn_edgeailite_aspp_mobilenetv2_tv_768x384_2020-01-20_13-57-01_checkpoint.pth
#
##### Validation - VOC Segmentation - Validation with MobileNetV2+UNetEdgeAILite - replace the pretrained checkpoint with what you have.
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase validation --dataset_name cityscapes_segmentation --model_name unet_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/jai-modelzoo/pytorch/vision/segmentation/jai-devkit/cityscapes/unet_edgeailite_aspp_mobilenetv2_tv_768x384_2020-01-29_16-43-40_checkpoint.pth


## =====================================================================================
## Inference
## =====================================================================================
##### Inference - Cityscapes Semantic Segmentation - Inference with RegNetX+DeeplabV3Lite - replace the pretrained checkpoint with what you have.
#python3 ./references/edgeailite/main/infer_segmentation_main.py --dataset_name cityscapes_segmentation_measure --model_name fpn_aspp_regnetx800mf_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/jai-modelzoo/pytorch/vision/segmentation/jai-devkit/cityscapes/fpn_aspp_regnetx_edgeailite800mf_768x384_2020-08-04_21-14-35_checkpoint.pth


## =====================================================================================
# VOC Segmentation Training
## =====================================================================================
#### VOC Segmentation - Training with RegNetX+DeeplabV3Lite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name voc_segmentation --model_name fpn_aspp_regnetx800mf_edgeailite \
#--data_path ./data/datasets/voc --img_resize 512 512 --output_size 512 512 --gpus 0 1 \
#--phase validation --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth


## =====================================================================================
# COCO 80 class Segmentation Training
## =====================================================================================
# compute accuracy at resized resolution - to speedup training
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name coco_segmentation --model_name fpn_aspp_regnetx800mf_edgeailite \
#--data_path ./data/datasets/coco --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth --batch_size 16 \
#--optimizer sgd --scheduler cosine --lr 1e-1 --epochs 30 ${EXTRA_OPTIONS}


## =====================================================================================
# COCO 21 class Segmentation Training
## =====================================================================================
# compute accuracy at resized resolution - to speedup training
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name coco_seg21 --model_name deeplabv3plus_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/coco --img_resize 512 512 --output_size 1024 1024 --gpus 0 1 2 3 \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 5e-2 \
#--interpolation 1 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name coco_seg21 --model_name fpn_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/coco --img_resize 512 512 --output_size 1024 1024 --gpus 0 1 2 3 \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 5e-2 \
#--interpolation 1 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name coco_seg21 --model_name unet_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/coco --img_resize 512 512 --output_size 1024 1024 --gpus 0 1 2 3 \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 5e-2 \
#--interpolation 1 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name coco_seg21 --model_name fpn_aspp_regnetx400mf_edgeailite \
#--data_path ./data/datasets/coco --img_resize 384 384 --output_size 1024 1024 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 1e-1 \
#--interpolation 1 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name coco_seg21 --model_name fpn_aspp_regnetx800mf_edgeailite \
#--data_path ./data/datasets/coco --img_resize 512 512 --output_size 1024 1024 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth \
#--weight_decay 4e-5 --batch_size 32 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 1e-1 \
#--interpolation 1 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name coco_seg21 --model_name fpn_aspp_regnetx1p6gf_edgeailite \
#--data_path ./data/datasets/coco --img_resize 768 768 --output_size 1024 1024 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
#--weight_decay 4e-5 --batch_size 16 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 1e-1 \
#--interpolation 1 ${EXTRA_OPTIONS}

#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name coco_seg21 --model_name fpn_aspp_regnetx3p2gf_edgeailite \
#--data_path ./data/datasets/coco --img_resize 1024 1024  --rand_crop 768 768 --output_size 1024 1024 --gpus 0 1 2 3 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth \
#--weight_decay 4e-5 --batch_size 16 --epochs 60 --milestones 30 45 --optimizer sgd --scheduler cosine --lr 1e-1 \
#--interpolation 1 ${EXTRA_OPTIONS}
