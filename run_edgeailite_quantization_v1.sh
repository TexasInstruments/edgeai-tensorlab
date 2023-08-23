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

## Quantization
##
## =====================================================================================
## Quantization Aware Training
## =====================================================================================
#
#### Image Classification - Quantization Aware Training - MobileNetV2
# python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name mobilenetv2_tv_x1 \
# --data_path ./data/datasets/image_folder_classification \
# --pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
# --batch_size 256 --quantize True --epochs 10 --lr 1e-5 --warmup_epochs 0 --evaluate_start False
#
#
#### Image Classification - Quantization Aware Training - MobileNetV2(Shicai) - a TOUGH MobileNetV2 pretrained model
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name mobilenetv2_shicai_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained ./data/modelzoo/pytorch/image_classification/imagenet1k/shicai/mobilenetv2_shicai_rgb.pth \
#--batch_size 256 --quantize True --epochs 10 --lr 1e-5 --warmup_epochs 0 --evaluate_start False
#
#
#### Semantic Segmentation - Quantization Aware Training for MobileNetV2+DeeplabV3Lite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name deeplabv3plus_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/pytorch/semantic_seg/cityscapes/jacinto_ai/deeplabv3plus_edgeailite_mobilenetv2_tv_768x384_best.pth \
#--batch_size 12 --quantize True --epochs 10  --lr 1e-5 --warmup_epochs 0 --evaluate_start False
##
##
#### Semantic Segmentation - Quantization Aware Training for MobileNetV2+UNetEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name unet_edgeailite_pixel2pixel_aspp_mobilenetv2_tv \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/pytorch/semantic_seg/cityscapes/jacinto_ai/unet_aspp_mobilenetv2_tv_768x384_best.pth \
#--batch_size 12 --quantize True --epochs 10 --lr 1e-5 --warmup_epochs 0 --evaluate_start False
##
##
#### Semantic Segmentation - Quantization Aware Training for MobileNetV2+FPNEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/pytorch/semantic_seg/cityscapes/jacinto_ai/fpn_aspp_mobilenetv2_tv_768x384_best.pth \
#--batch_size 12 --quantize True --epochs 10 --lr 1e-5 --warmup_epochs 0 --evaluate_start False
##
##
#### Depth Estimation - Quantization Aware Training for MobileNetV2+DeeplabV3Lite
#python3 ./references/edgeailite/main/pixel2pixel/train_depth_main.py --dataset_name kitti_depth --model_name deeplabv3plus_mobilenetv2_tv_edgeailite --img_resize 384 768 --output_size 1024 2048 \
#--pretrained ./data/modelzoo/pytorch/monocular_depth/kitti_depth/jacinto_ai/deeplabv3plus_edgeailite_mobilenetv2_tv_768x384_best.pth \
#--batch_size 32 --quantize True --epochs 10 --lr 1e-5 --warmup_epochs 0 --evaluate_start False
#
#

## =====================================================================================
## Post Training Calibration & Quantization - this is fast, but may not always yield best quantized accuracy (not recommended)
## =====================================================================================
#
#### Image Classification - Post Training Calibration & Quantization - ResNet50
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase calibration --dataset_name image_folder_classification --model_name resnet50_x1 \
#--data_path ./data/datasets/image_folder_classification --gpus 0 \
#--pretrained https://download.pytorch.org/models/resnet50-19c8e357.pth \
#--batch_size 64 --quantize True --epochs 1 --epoch_size 0.1 --evaluate_start False
#
#
#### Image Classification - Post Training Calibration & Quantization - ResNet18
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase calibration --dataset_name image_folder_classification --model_name resnet18_x1 \
#--data_path ./data/datasets/image_folder_classification --gpus 0 \
#--pretrained https://download.pytorch.org/models/resnet18-5c106cde.pth \
#--batch_size 64 --quantize True --epochs 1 --epoch_size 0.1 --evaluate_start False
#
#
#### Image Classification - Post Training Calibration & Quantization - MobileNetV2
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase calibration --dataset_name image_folder_classification --model_name mobilenetv2_tv_x1 \
#--data_path ./data/datasets/image_folder_classification --gpus 0 \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
#--batch_size 64 --quantize True --epochs 1 --epoch_size 0.1 --evaluate_start False
#
#
#### Image Classification - Post Training Calibration & Quantization - ShuffleNetV2
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase calibration --dataset_name image_folder_classification --model_name shufflenetv2_x1p0 \
#--data_path ./data/datasets/image_folder_classification --gpus 0 \
#--pretrained https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth \
#--batch_size 64 --quantize True --epochs 1 --epoch_size 0.1 --evaluate_start False
#
#
#### Image Classification - Post Training Calibration & Quantization for a TOUGH MobileNetV2 pretrained model
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase calibration --dataset_name image_folder_classification --model_name mobilenetv2_shicai_x1 \
#--data_path ./data/datasets/image_folder_classification --gpus 0 \
#--pretrained ./data/modelzoo/pytorch/image_classification/imagenet1k/shicai/mobilenetv2_shicai_rgb.pth \
#--batch_size 64 --quantize True --epochs 1 --epoch_size 0.1 --evaluate_start False
#
#
### Semantic Segmentation - Post Training Calibration &  Quantization for MobileNetV2+DeeplabV3Lite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase calibration --dataset_name cityscapes_segmentation --model_name deeplabv3plus_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 \
#--pretrained ./data/modelzoo/pytorch/semantic_seg/cityscapes/jacinto_ai/deeplabv3plus_edgeailite_mobilenetv2_tv_768x384_best.pth \
#--batch_size 4 --quantize True --epochs 1 --evaluate_start False
#
#
### Semantic Segmentation - Post Training Calibration &  Quantization for MobileNetV2+UNetEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase calibration --dataset_name cityscapes_segmentation --model_name unet_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 \
#--pretrained ./data/modelzoo/pytorch/semantic_seg/cityscapes/jacinto_ai/unet_aspp_mobilenetv2_tv_768x384_best.pth \
#--batch_size 4 --quantize True --epochs 1 --evaluate_start False
#
#
### Semantic Segmentation - Post Training Calibration &  Quantization for MobileNetV2+FPNEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase calibration --dataset_name cityscapes_segmentation --model_name fpn_aspp_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 \
#--pretrained ./data/modelzoo/pytorch/semantic_seg/cityscapes/jacinto_ai/fpn_aspp_mobilenetv2_tv_768x384_best.pth \
#--batch_size 4 --quantize True --epochs 1 --evaluate_start False
#
#
### Depth Estimation - Post Training Calibration &  Quantization for MobileNetV2+DeeplabV3Lite
#python3 ./references/edgeailite/main/pixel2pixel/train_depth_main.py --phase calibration --dataset_name kitti_depth --model_name deeplabv3plus_mobilenetv2_tv_edgeailite  --gpus 0 \
#--pretrained ./data/modelzoo/pytorch/monocular_depth/kitti_depth/jacinto_ai/deeplabv3plus_edgeailite_mobilenetv2_tv_768x384_best.pth \
#--batch_size 4 --quantize True --epochs 1 --evaluate_start False


## =====================================================================================
## Acuracy Evaluation with Post Training Quantization - this is not supported anymore.
## Either Calibration or QAT has to be performed first, to get correct accuracy.
## Please use one of the sections above.
## =====================================================================================
#
#### Image Classification - Accuracy Estimation with Post Training Quantization - MobileNetV2
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name mobilenetv2_tv_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
#--batch_size 64 --quantize True
#
#### Image Classification - Accuracy Estimation with Post Training Quantization - ResNet50
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name resnet50_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/resnet50-19c8e357.pth \
#--batch_size 64 --quantize True
#
#### Image Classification - Accuracy Estimation with Post Training Quantization - ResNet18
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name resnet18_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/resnet18-5c106cde.pth \
#--batch_size 64 --quantize True
#
#### Image Classification - Accuracy Estimation with Post Training Quantization - A TOUGH MobileNetV2 pretrained model
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name mobilenetv2_shicai_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained ./data/modelzoo/pytorch/image_classification/imagenet1k/shicai/mobilenetv2_shicai_rgb.pth \
#--batch_size 64 --quantize True
#
#### Semantic Segmentation - Accuracy Estimation with Post Training Quantization - MobileNetV2+DeeplabV3Lite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase validation --dataset_name cityscapes_segmentation --model_name deeplabv3plus_mobilenetv2_tv_edgeailite \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained './data/modelzoo/pytorch/semantic_seg/cityscapes/jacinto_ai/deeplabv3plus_edgeailite_mobilenetv2_tv_768x384_best.pth' \
#--batch_size 1 --quantize True
#
#### Semantic Segmentation - Accuracy Estimation with Post Training Quantization - MobileNetV2+UNetEdgeAILite
#python3 ./references/edgeailite/main/pixel2pixel/train_segmentation_main.py --phase validation --dataset_name cityscapes_segmentation --model_name unet_edgeailite_pixel2pixel_aspp_mobilenetv2_tv \
#--data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 \
#--pretrained ./data/modelzoo/pytorch/semantic_seg/cityscapes/jacinto_ai/unet_aspp_mobilenetv2_tv_768x384_best.pth \
#--batch_size 1 --quantize True


## =====================================================================================
# Not completely supported feature - ONNX Model Import and PTQ
## =====================================================================================
#### Image Classification - Post Training Calibration & Quantization
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase calibration --dataset_name image_folder_classification --gpus 0 \
#--model_name resnet18-v1-7 --model /data/tensorlabdata1/modelzoo/pytorch/image_classification/imagenet1k/onnx-model-zoo/resnet18-v1-7.onnx \
#--data_path ./data/datasets/image_folder_classification --batch_size 64 --quantize True --epochs 1 --epoch_size 0.1 --evaluate_start False