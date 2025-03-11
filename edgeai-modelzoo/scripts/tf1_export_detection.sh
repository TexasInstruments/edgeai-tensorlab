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

#LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11
TF_MODELS_REPO="/user/a0393608/files/work/github/tensorflow/models"
export PYTHONPATH=:$PYTHONPATH:/user/a0393608/files/work/github/tensorflow/models/research:/user/a0393608/files/work/github/tensorflow/models/research/slim

############################################################
#[./downloads/tf1/od/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19/fp32]=320
#[./downloads/tf1/od/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19/fp32]=320
#[./downloads/tf1/od/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03]=640
#[./downloads/tf1/od/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18]=320
#[./downloads/tf1/od/ssdlite_mobilenet_v2_coco_2018_05_09]=300
declare -A MODEL_NAMES=(
[./downloads/tf1/od/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03]=640
)

for model_name in "${!MODEL_NAMES[@]}"; do

LOCAL_DIR="${model_name}"
input_size="${MODEL_NAMES[$model_name]}"
echo $LOCAL_DIR
echo $input_size

python3 ${TF_MODELS_REPO}/research/object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=${LOCAL_DIR}/pipeline.config \
--output_directory=${LOCAL_DIR}/tflite \
--trained_checkpoint_prefix=${LOCAL_DIR}/model.ckpt \
--add_postprocessing_op=true --use_regular_nms=true

tflite_convert --graph_def_file ${LOCAL_DIR}/tflite/tflite_graph.pb \
--enable_v1_converter \
--experimental_new_converter=False \
--output_file ${LOCAL_DIR}/tflite/model.tflite \
--input_shapes=1,$input_size,$input_size,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--change_concat_input_ranges=false \
--allow_custom_ops

done


