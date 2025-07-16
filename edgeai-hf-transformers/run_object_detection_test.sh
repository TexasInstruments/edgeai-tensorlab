#!/bin/bash

#!/usr/bin/env bash

# Copyright (c) 2018-2021, Texas Instruments
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

##################################################################


# method 1 of specifying mean, std: (img/255 - mean)/std
#--image_mean 0.485 0.456 0.406
#--image_std 0.229 0.224 0.225
#
# method 2 of specifying mean, scale: (img*rescale_factor - mean)*scale
#--rescale_factor 1.0
#--image_mean 123.675 116.28 103.53
#--image_scale 0.017125 0.017507 0.017429

MODEL_NAME_OR_PATH="facebook/detr-resnet-50"


CUDA_VISIBLE_DEVICES="0" \
python3 examples/pytorch/object-detection/run_object_detection.py \
                --trust_remote_code True \
                --dataset_name /home/a0491009/datasets/coco \
                --model_name_or_path ${MODEL_NAME_OR_PATH} \
                --size 800 \
                --rescale_factor 1.0 \
                --image_mean "123.675 116.28 103.53" \
                --image_scale "0.017125 0.017507 0.017429" \
                --output_dir outputs \
                --remove_unused_columns False \
                --do_train True \
                --do_eval True \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --overwrite_output_dir \
                --ignore_mismatched_sizes True \
                --do_onnx_export True \
                --dataloader_num_workers 12 \
                --eval_do_concat_batches False \
                --label_names labels \
                --save_strategy no \
                --dataloader_drop_last True \
                --quantization 3 \
                --quantize_type PTQ \
                --quantize_calib_images 5 \
                --qconfig_type MSA_WC8_AT8 \
                # --use_cpu True \
                # --qconfig_type MSA_WC8P2_AT8P2 
                #--max_eval_samples 1000 \
                #--max_train_samples 200 \
