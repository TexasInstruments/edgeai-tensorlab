#!/bin/bash

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

# ----------------------------------
# Quantization Aware Training (QAT) Example
# Texas Instruments (C) 2018-2020
# All Rights Reserved
# ----------------------------------

# ----------------------------------
date_var=`date '+%Y-%m-%d_%H-%M-%S'`
base_dir="./data/checkpoints/quantization_example"
save_path="$base_dir"/"$date_var"_quantization_example
log_file=$save_path/run.log
echo Logging the output to: $log_file

# ----------------------------------
mkdir -p $save_path
exec &> >(tee -a "$log_file")

# ----------------------------------
# model names and pretrained paths from torchvision - add more as required
declare -A model_pretrained=(
  [mobilenetv2_tv_x1]=https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
  #[resnet50_x1]=https://download.pytorch.org/models/resnet50-19c8e357.pth
  #[shufflenet_v2_x1_0]=https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
  #[mobilenetv2_shicai]='./data/modelzoo/pytorch/image_classification/imagenet1k/shicai/mobilenetv2_shicai_rgb.pth'
)

# ----------------------------------
# parameters for quantization aware training
# ----------------------------------
lr=1e-5             # initial learning rate for quantization aware training - recommend to use 1e-5 (or at max 5e-5)
lr_step_size=10     # adjust lr after so many epochs
batch_size=256      # use a relatively smaller batch size as quantization aware training does not use multi-gpu
epochs=10           # number of epochs to train
epoch_size=0        # use a fraction to limit the number of images used in one epoch - set to 0 to use the full training epoch
epoch_size_val=0    # use a fraction to limit the number of images used in one epoch - set to 0 to use the full validation epoch


# ----------------------------------
for model in "${!model_pretrained[@]}"; do
  echo ==========================================================
  pretrained="${model_pretrained[$model]}"

  echo ----------------------------------------------------------
  echo Quantization Aware Training for $model
  # note: this example uses only a part of the training epoch and only 10 such (partial) epochs during quantized training to save time,
  # but it may necessary to use the full training epoch if the accuracy is not satisfactory.
  python3 -u ./references/edgeailite/scripts/train_classification_quantization.py ./data/datasets/image_folder_classification \
               --arch $model --batch_size $batch_size --lr $lr --epoch_size $epoch_size --epoch_size_val $epoch_size_val \
               --epochs $epochs --pretrained $pretrained --save_path $save_path \
               --img_resize 256 --img_crop 224 \
               --quantize True --use_gpu True
done
