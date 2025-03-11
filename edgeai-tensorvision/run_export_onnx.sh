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

##depthwise Convolution based Models
##------------------------
#### ImageNet - ShuffleNetV2-1.0
#python3 ./references/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name shufflenetv2_x1p0 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

#### ImageNet - MobileNetV2
#python3 ./references/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name mobilenetv2_tv_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

##ResNet50 based Models
##------------------------
#### ImageNet - ResNet50
#python3 ./references/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name resnet50_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/resnet50-19c8e357.pth

#### ImageNet - resnet50_xp5
#python3 ./references/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name resnet50_xp5 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained ./data/modelzoo/pytorch/image_classification/imagenet1k/jacinto_ai/resnet50-0.5_2018-07-23_12-10-23.pth


##VGG based Models
##------------------------
#### ImageNet - VGG16
#python3 ./references/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name vgg16_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--pretrained https://download.pytorch.org/models/vgg16-397923af.pth


##RegNetX based Models
##------------------------
#### ImageNet regnetx800mf_x1 with BGR input
##Note: to use BGR input, set: --input_channel_reverse True, for RGB input ommit this argument or set it to False.
#python3 ./references/classification/train_classification_main.py --phase validation --dataset_name image_folder_classification --model_name resnet50_x1 \
#--data_path ./data/datasets/image_folder_classification \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 --model_name regnetx800mf_x1 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth

