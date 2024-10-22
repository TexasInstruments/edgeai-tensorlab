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
#################################################################################

# ver:35 2024-08-17
# mapping from artifacts id to readable artifact names
# generated using ./run_generate_configs.sh -> this will write model_infos.py into edgeai-modelzoo/models/model_infos.py

# Conventions
# RunTime  Task                  start_id
# TF	   Classification	        0000
# TF	   human pose estimation	1000
# TF	   3D OD	                1100
# TF	   object pose estimation	1200
# TF	   depth estimation	        1300
# TF	   OD                       2000
# TF	   Semantic Segmentation	2500

# TVM	   Classification	        3000
# TVM	   human pose estimation	4000
# TVM	   3D OD	                4100
# TVM	   object pose estimation	4200
# TVM	   depth estimation	        4300
# TVM	   OD                       5000
# TVM	   Semantic Segmentation	5500

# ONR	   Classification	        6000
# ONR	   human pose estimation	7000
# ONR	   3D OD	                7100
# ONR	   6dobject pose estimation	7200
# ONR	   depth estimation	        7300
# ONR	   OD                       8000
# ONR	   Semantic Segmentation	8500


MODEL_INFOS_DICT = \
{
    'cl-0000': { 'model_id': 'cl-0000', 'recommended': True, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0000-mobileNetV1-mlperf', },
    'cl-3090': { 'model_id': 'cl-3090', 'recommended': True, 'shortlisted': True, 'session_name': 'tvmdlr', 'artifact_name': 'TVM-CL-3090-mobileNetV2-tv', },
    'cl-6360': { 'model_id': 'cl-6360', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6360-regNetx-200mf', },
    'kd-7060': { 'model_id': 'kd-7060', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-KD-7060-human-pose-yolox-s-640x640', },
    'od-2020': { 'model_id': 'od-2020', 'recommended': True, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2020-ssdLite-mobDet-DSP-coco-320x320', },
    'od-5120': { 'model_id': 'od-5120', 'recommended': True, 'shortlisted': True, 'session_name': 'tvmdlr', 'artifact_name': 'TVM-OD-5120-ssdLite-mobDet-DSP-coco-320x320', },
    'od-8020': { 'model_id': 'od-8020', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8020-ssd-lite-mobv2-mmdet-coco-512x512', },
    'od-8200': { 'model_id': 'od-8200', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8200-yolox-nano-lite-mmdet-coco-416x416', },
    'od-8220': { 'model_id': 'od-8220', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640', },
    'od-8270': { 'model_id': 'od-8270', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8270-yolox-pico-lite-mmdet-coco-320x320', },
    'od-8410': { 'model_id': 'od-8410', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8410-yolox-tiny-lite-mmdet-widerface-416x416', },
    'od-8420': { 'model_id': 'od-8420', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8420-yolox-s-lite-mmdet-widerface-640x640', },
    'ss-2580': { 'model_id': 'ss-2580', 'recommended': True, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-SS-2580-deeplabv3_mobv2-ade20k32-mlperf-512x512', },
    'ss-5710': { 'model_id': 'ss-5710', 'recommended': True, 'shortlisted': True, 'session_name': 'tvmdlr', 'artifact_name': 'TVM-SS-5710-deeplabv3lite-mobv2-cocoseg21-512x512', },
    'ss-7618': { 'model_id': 'ss-7618', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-7618-deeplabv3lite-mobv2-qat-robokit-768x432', },
    'ss-8610': { 'model_id': 'ss-8610', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8610-deeplabv3lite-mobv2-ade20k32-512x512', },
    '3dod-7100': { 'model_id': '3dod-7100', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-3DOD-7100-pointPillars-lidar-10000-1c-496x432', },
    '6dpose-7200': { 'model_id': '6dpose-7200', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-6DPOSE-7200-yolox-s-6d-object_pose-640x480', },
    'cl-0010': { 'model_id': 'cl-0010', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0010-mobileNetV2', },
    'cl-0080': { 'model_id': 'cl-0080', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0080-mobileNet-edgeTPU-mlperf', },
    'cl-0090': { 'model_id': 'cl-0090', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0090-efficientNet-edgeTPU-s', },
    'cl-0130': { 'model_id': 'cl-0130', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0130-efficientNet-lite0', },
    'cl-0140': { 'model_id': 'cl-0140', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0140-efficientNet-lite4', },
    'cl-0160': { 'model_id': 'cl-0160', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0160-resNet50V1p5-mlperf', },
    'cl-6090': { 'model_id': 'cl-6090', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6090-mobileNetV2-tv', },
    'cl-6098': { 'model_id': 'cl-6098', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6098-mobileNetV2-tv-qat', },
    'cl-6100': { 'model_id': 'cl-6100', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6100-resNet18', },
    'cl-6101': { 'model_id': 'cl-6101', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6101-resNet18-512x512', },
    'cl-6102': { 'model_id': 'cl-6102', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6102-resNet18-1024x1024', },
    'cl-6110': { 'model_id': 'cl-6110', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6110-resNet50', },
    'cl-6151': { 'model_id': 'cl-6151', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6151-mobileNetV2-1p4-qat-512x512', },
    'cl-6160': { 'model_id': 'cl-6160', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6160-regNetX-400mf-tv', },
    'cl-6170': { 'model_id': 'cl-6170', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6170-regNetX-800mf-tv', },
    'cl-6180': { 'model_id': 'cl-6180', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6180-regNetX-1.6gf-tv', },
    'cl-6480': { 'model_id': 'cl-6480', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6480-mobv3-lite-small', },
    'cl-6488': { 'model_id': 'cl-6488', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6488-mobv3-lite-small-qat', },
    'cl-6490': { 'model_id': 'cl-6490', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6490-mobv3-lite-large', },
    'od-2000': { 'model_id': 'od-2000', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2000-ssd-mobV1-coco-mlperf-300x300', },
    'od-2010': { 'model_id': 'od-2010', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2010-ssd-mobV2-coco-mlperf-300x300', },
    'od-2030': { 'model_id': 'od-2030', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2030-ssdLite-mobDet-EdgeTPU-coco-320x320', },
    'od-2080': { 'model_id': 'od-2080', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2080-ssd-mobV2-fpnlite-coco-tpu-8-320x320', },
    'od-8040': { 'model_id': 'od-8040', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8040-ssd-lite-regNetX-200mf-fpn-bgr-mmdet-coco-320x320', },
    'od-8210': { 'model_id': 'od-8210', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8210-yolox-tiny-lite-mmdet-coco-416x416', },
    'ss-8630': { 'model_id': 'ss-8630', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8630-unetlite-aspp-mobv2-tv-ade20k32-512x512', },
    'ss-8710': { 'model_id': 'ss-8710', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8710-deeplabv3lite-mobv2-cocoseg21-512x512', },
    'ss-8720': { 'model_id': 'ss-8720', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8720-deeplabv3lite-regnetx800mf-cocoseg21-512x512', },
    'ss-8740': { 'model_id': 'ss-8740', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8740-lraspp-mobV3-ti-lite-large-cocoseg21-512x512', },
    'cl-0038': { 'model_id': 'cl-0038', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0038-InceptionNetV1', },
    'cl-0040': { 'model_id': 'cl-0040', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0040-InceptionNetV3', },
    'cl-0100': { 'model_id': 'cl-0100', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0100-efficientNet-edgeTPU-m', },
    'cl-0170': { 'model_id': 'cl-0170', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0170-efficientNet-lite1', },
    'cl-0200': { 'model_id': 'cl-0200', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0200-mobileNetV2-1p4', },
    'cl-6061': { 'model_id': 'cl-6061', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6061-mobileNetV1-512x512', },
    'cl-6062': { 'model_id': 'cl-6062', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6062-mobileNetV1-1024x1024', },
    'cl-6080': { 'model_id': 'cl-6080', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6080-shuffleNetV2', },
    'cl-6091': { 'model_id': 'cl-6091', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6091-mobileNetV2-tv-512x512', },
    'cl-6092': { 'model_id': 'cl-6092', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6092-mobileNetV2-tv-1024x1024', },
    'cl-6111': { 'model_id': 'cl-6111', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6111-resNet50-512x512', },
    'cl-6112': { 'model_id': 'cl-6112', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6112-resNet50-1024x1024', },
    'cl-6121': { 'model_id': 'cl-6121', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6121-regNetX-400mf-pycls-512x512', },
    'cl-6122': { 'model_id': 'cl-6122', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6122-regNetX-400mf-pycls-1024x1024', },
    'cl-6131': { 'model_id': 'cl-6131', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6131-regNetX-800mf-pycls-512x512', },
    'cl-6132': { 'model_id': 'cl-6132', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6132-regNetX-800mf-pycls-1024x1024', },
    'cl-6141': { 'model_id': 'cl-6141', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6141-regNetX-1.6gf-pycls-512x512', },
    'cl-6142': { 'model_id': 'cl-6142', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6142-regNetX-1.6gf-pycls-1024x1024', },
    'cl-6152': { 'model_id': 'cl-6152', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6152-mobileNetV2-1p4-qat-1024x1024', },
    'cl-6500': { 'model_id': 'cl-6500', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6500-mobileNetV2-lite-wtv2-224', },
    'cl-6508': { 'model_id': 'cl-6508', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6508-mobileNetV2-lite-wtv2-qatv2-perc-224', },
    'cl-6510': { 'model_id': 'cl-6510', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6510-mobileNetV3-large-lite-wtv2-224', },
    'cl-6530': { 'model_id': 'cl-6530', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6530-resNet50-wtv2-224', },
    'cl-6540': { 'model_id': 'cl-6540', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6540-resNet101-wtv2-224', },
    'cl-6550': { 'model_id': 'cl-6550', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6550-resNeXt50-32x4d-wtv2-224', },
    'cl-6560': { 'model_id': 'cl-6560', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6560-resNeXt101-32x8d-wtv2-224', },
    'cl-6570': { 'model_id': 'cl-6570', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6570-regNetX-1.6gf-wtv2-224', },
    'cl-6580': { 'model_id': 'cl-6580', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6580-regNetX-400mf-wtv2-224', },
    'cl-6590': { 'model_id': 'cl-6590', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6590-regNetX-800mf-wtv2-224', },
    'cl-6700': { 'model_id': 'cl-6700', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6700-ViT-tiny-patch16-transformer-224', },
    'cl-6710': { 'model_id': 'cl-6710', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6710-ViT-base-patch16-transformer-224', },
    'cl-6720': { 'model_id': 'cl-6720', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6720-DeiT-tiny-patch16-transformer-224', },
    'cl-6730': { 'model_id': 'cl-6730', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6730-LeViT-128-transformer-224', },
    'cl-6740': { 'model_id': 'cl-6740', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6740-LeViT-256-transformer-224', },
    'cl-6750': { 'model_id': 'cl-6750', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6750-Swin-tiny-patch4-window7-transformer-224', },
    'cl-6760': { 'model_id': 'cl-6760', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6760-Swin-small-patch4-window7-transformer-224', },
    'cl-6770': { 'model_id': 'cl-6770', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6770-Swin-base-patch4-window7-transformer-224', },
    'cl-6800': { 'model_id': 'cl-6800', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6800-convNext-small-tv-224', },
    'cl-6810': { 'model_id': 'cl-6810', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6810-efficientNet-b0-224', },
    'cl-6820': { 'model_id': 'cl-6820', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6820-efficientNet-b1-224', },
    'cl-6830': { 'model_id': 'cl-6830', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6830-mobilenetV3-small-224', },
    'cl-6840': { 'model_id': 'cl-6840', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6840-convNext-tiny-hf-224', },
    'de-7300': { 'model_id': 'de-7300', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-DE-7300-depth-est-fast-depth-224x224', },
    'de-7310': { 'model_id': 'de-7310', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-DE-7310-depth-est-midas-small-256x256', },
    'od-2070': { 'model_id': 'od-2070', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2070-ssd-mobV1-fpn-coco-tpu-8-640x640', },
    'od-2150': { 'model_id': 'od-2150', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2150-efficientDet-lite1-relu-coco-384x384', },
    'od-8800': { 'model_id': 'od-8800', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8800-scaled-yolov4-csp-lite-640x640-gplv3', },
    'od-8810': { 'model_id': 'od-8810', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8810-yolov5-nano-v61-lite-640x640-gplv3', },
    'od-8820': { 'model_id': 'od-8820', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8820-yolov5-small-v61-lite-640x640-gplv3', },
    'od-8850': { 'model_id': 'od-8850', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8850-yolov7-tiny-lite-640x640-gplv3', },
    'od-8860': { 'model_id': 'od-8860', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8860-yolov7-large-lite-640x640-gplv3', },
    'od-8870': { 'model_id': 'od-8870', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8870-yolov8-nano-lite-640x640-gplv3', },
    'od-8880': { 'model_id': 'od-8880', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8880-yolov8-small-lite-640x640-gplv3', },
    'od-8890': { 'model_id': 'od-8890', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8890-yolox-tiny-lite-416x416-gplv3', },
    'od-8900': { 'model_id': 'od-8900', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8900-yolox-small-lite-640x640-gplv3', },
    'od-8920': { 'model_id': 'od-8920', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8920-DETR-fb-resnet50-transformer-coco-800x800', },
    'od-8930': { 'model_id': 'od-8930', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8930-FCOS-r50-fpn-gn-head-coco-1216x800', },
    'od-8940': { 'model_id': 'od-8940', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8940-CenterNet-update-r50-fpn-coco-1216x800', },
    'od-8950': { 'model_id': 'od-8950', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8950-CenterNet-r18-coco-672x448', },
    'od-8960': { 'model_id': 'od-8960', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8960-DETR-r50-mmdet-transformer-coco-800x800', },
    'ss-8650': { 'model_id': 'ss-8650', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8650-fpnlite-aspp-mobv2-ade20k32-512x512', },
    'ss-8670': { 'model_id': 'ss-8670', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512', },
    'ss-8690': { 'model_id': 'ss-8690', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8690-fpnlite-aspp-regnetx400mf-ade20k32-384x384', },
    'ss-8700': { 'model_id': 'ss-8700', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8700-fpnlite-aspp-regnetx800mf-ade20k32-512x512', },
    'ss-8730': { 'model_id': 'ss-8730', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8730-deeplabv3-mobv3-lite-large-cocoseg21-512x512', },
    'ss-8750': { 'model_id': 'ss-8750', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8750-segformerB0-transformer-ade-512x512', },
    'ss-8760': { 'model_id': 'ss-8760', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8760-segformerB5-transformer-ade-640x640', },
    'visloc-7500': { 'model_id': 'visloc-7500', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-VISLOC-7500-tiad-dkaze-carla-768x384', },
}

