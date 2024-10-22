#!/usr/bin/env bash

# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved
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

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11
TF_MODELS_REPO="/user/a0393608/files/work/github/tensorflow/models"
export PYTHONPATH=:$PYTHONPATH:/user/a0393608/files/work/github/tensorflow/models/research

############################################################
#./downloads/tf2/classification/resnet_50_classification_1
#./downloads/tf2/classification/imagenet_resnet_v2_50_classification_4
#./downloads/tf2/classification/imagenet_resnet_v1_101_classification_4
#./downloads/tf2/classification/imagenet_resnet_v2_50_classification_4
#./downloads/tf2/classification/imagenet_inception_v1_classification_4
MODEL_NAMES="
./downloads/tf2/classification/tfgan_eval_inception_1
"

for model_name in ${MODEL_NAMES}; do
LOCAL_DIR="${model_name}"

tflite_convert \
    --enable_v1_converter \
    --experimental_new_converter=True \
    --change_concat_input_ranges=false \
    --allow_custom_ops \
    --saved_model_dir=${LOCAL_DIR}/saved_model \
    --output_file=${LOCAL_DIR}/tflite/model.tflite



done


