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

#### Some of the models that can be used in this script:
#regnetx400mf_x1
#regnetx800mf_x1
#regnetx1p6gf_x1
#regnetx3p2gf_x1
#mobilenetv2_tv_x1
#resnet50_x1
#resnet50_xp5

## =====================================================================================
## Cifar Training (Dataset will be automatically downloaded)
## =====================================================================================
## Cifar100 Classification (Automatic Download)
#### Training with MobileNetV2
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name cifar100_classification --model_name regnetx800mf_x1 \
#--data_path ./data/datasets/cifar100_classification --img_resize 32 --img_crop 32 --rand_scale 0.5 1.0 --strides 1 1 1 2 2

## Cifar10 Classification (Automatic Download)
#### Training with MobileNetV2
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name cifar10_classification --model_name regnetx800mf_x1 \
#--data_path ./data/datasets/cifar10_classification --img_resize 32 --img_crop 32 --rand_scale 0.5 1.0 --strides 1 1 1 2 2

## =====================================================================================
## ImageNet Training (Assuming ImageNet data is already Manually Downloaded)
## =====================================================================================
#RegNetX based Models
#Note the original repository that provided RegNet models (https://github.com/facebookresearch/pycls) trained it with BGR input
#Note: to use BGR input, set: --input_channel_reverse True and provide reversed mean/scale values
#Example: --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125

#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name regnetx800mf_x1_bgr \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
#--data_path ./data/datasets/image_folder_classification --weight_decay 5e-5 --epochs 120 --batch_size 512 --lr 0.4

#MobileNetV2 based Models
#------------------------
#### Training with MobileNetV2
python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name mobilenetv2_tv_x1 \
--data_path ./data/datasets/image_folder_classification --weight_decay 4e-5 --epochs 150 --batch_size 512 --lr 0.1

#### Training with MobileNetV2 - Small Resolution
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name mobilenetv2_tv_x1 \
#--data_path ./data/datasets/image_folder_classification --img_resize 146 --img_crop 128  --weight_decay 4e-5 --epochs 150 --batch_size 1024 --lr 0.2

#### Training with MobileNetV2 with 2x channels and expansion factor of 2
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name mobilenetv2_tv_x2_t2 \
#--data_path ./data/datasets/image_folder_classification --weight_decay 4e-5 --epochs 150 --batch_size 512 --lr 0.1

#MobileNetV3Lite based Models - Training
#------------------------
# For fast training, remove auto_augument, random erasing and reduce the epochs to 150
#
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name mobilenetv3_lite_small_x1 \
#--data_path ./data/datasets/image_folder_classification --weight_decay 4e-5 --epochs 150 --batch_size 512 --lr 0.1

#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name mobilenetv3_lite_large_x1 \
#--data_path ./data/datasets/image_folder_classification --weight_decay 4e-5 --epochs 600 --batch_size 512 --lr 0.1 \
#--auto_augument imagenet --random_erasing 0.2

#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name mobilenetv3_lite_large_x2r \
#--data_path ./data/datasets/image_folder_classification --weight_decay 4e-5 --epochs 150 --batch_size 512 --lr 0.1

#ResNet50 based Models
#------------------------
### Training with ResNet18
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name resnet18_x1 \
#--data_path ./data/datasets/image_folder_classification

### Training with ResNet50
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name resnet50_x1 \
#--data_path ./data/datasets/image_folder_classification

### Training with ResNet50 - with half the number of channels - so roughly 1/4 the complexity
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name image_folder_classification --model_name resnet50_xp5 \
#--data_path ./data/datasets/image_folder_classification

## =====================================================================================
## Validation
## =====================================================================================
#### cifar100 Validation - populate the pretrained model path below in ??
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name cifar100_classification --model_name mobilenetv2_tv_x1 \
#--data_path ./data/datasets/cifar100_classification --img_resize 32 --img_crop 32 \
#--pretrained=???

#### cifar10 Validation - populate the pretrained model path below in ??
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name cifar10_classification --model_name mobilenetv2_tv_x1 \
#--data_path ./data/datasets/cifar10_classification --img_resize 32 --img_crop 32 \
#--pretrained=???

#RegNetX based Models
#------------------------
#### Validation - ImageNet
#Note: to use BGR input, set: --input_channel_reverse True and provide reversed mean/scale values
#facebookresearch/pycls, trains with BGR input, so if you use pretrained models from there, it is important.
#Example:--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125
#Howerver,  we are using RGB input in our training, to be consistent with other models in this repository.
#So if you use our pretrained weights, the lines starting with --input_channel_reverse True should not be there.

#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name regnetx400mf_x1_bgr \
#--data_path ./data/datasets/image_folder_classification \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth

#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name regnetx800mf_x1_bgr \
#--data_path ./data/datasets/image_folder_classification \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth

#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr \
#--data_path ./data/datasets/image_folder_classification \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth

#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name regnetx3p2gf_x1_bgr \
#--data_path ./data/datasets/image_folder_classification \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth

#MobileNetV2 based Models
#------------------------
#### Validation - ImageNet - populate the pretrained model path below in ?? or use https://download.pytorch.org/models/mobilenet_v2-b0353104.pth for mobilenetv2_tv_x1
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name mobilenetv2_tv_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

#ShuffleNetV2 based Models
#------------------------
#### Validation - ImageNet - populate the pretrained model path below in ?? or use https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth for shufflenetv2_x1p0
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name shufflenetv2_x1p0 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

#ResNet50 based Models
#------------------------
#### Validation - ImageNet - populate the pretrained model path below in ?? or use https://download.pytorch.org/models/resnet50-19c8e357.pth for resnet50_x1
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name resnet50_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/resnet50-19c8e357.pth

#### Validation - ImageNet - populate the pretrained model path below in ?? for resnet50_xp5
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name resnet50_xp5 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained ./data/modelzoo/pytorch/image_classification/imagenet1k/jacinto_ai/resnet50-0.5_2018-07-23_12-10-23.pth

#MobileNetV3
#-----------
#### Validation - ImageNet - MobileNetV3
#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name mobilenetv3_lite_small_x1 \
#--data_path ./data/datasets/image_folder_classification --batch_size 4 \
#--pretrained "../edgeai-modelzoo/models/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_small_20210429_checkpoint.pth"

#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name mobilenetv3_lite_large_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained "../edgeai-modelzoo/models/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_large_20210507_checkpoint.pth"

#python3 ./references/edgeailite/main/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name mobilenetv3_lite_large_x2r \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained "../jacinto-ai-modelzoo/models/vision/classification/imagenet1k/torchvision/mobilenet_v3_lite_large_x2r_20210522_checkpoint.pth"


## =====================================================================================
#Training with ImageNet data download - download may take too much time - we have not tested this.
## =====================================================================================
#### Training with MobileNetV2
#python3 ./references/edgeailite/main/classification/train_classification_main.py --dataset_name imagenet_classification --model_name mobilenetv2_tv_x1 \
#--data_path ./data/datasets/imagenet_classification
