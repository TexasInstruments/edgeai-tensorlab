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
    'od-9204': { 'model_id': 'od-9204', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-9204-yolov9-s-lite-mmdet-coco-640x640', },
    'od-9205': { 'model_id': 'od-9205', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-9205-yolov9-s-mod-mmdet-coco-640x640', },
    'od-9206': { 'model_id': 'od-9206', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-9206-rtmdet-m-lite-mmdet-coco-640x640', },
    'od-9207': { 'model_id': 'od-9207', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-9207-rtmdet-m-orig-mmdet-coco-640x640', },
    'od-9208': { 'model_id': 'od-9208', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-9208-rtmdet-l-orig-mmdet-coco-640x640', },
    'od-9209': { 'model_id': 'od-9209', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-9209-rtmdet-l-lite-mmdet-coco-640x640', },
    'ss-2580': { 'model_id': 'ss-2580', 'recommended': True, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-SS-2580-deeplabv3_mobv2-ade20k32-mlperf-512x512', },
    'ss-5710': { 'model_id': 'ss-5710', 'recommended': True, 'shortlisted': True, 'session_name': 'tvmdlr', 'artifact_name': 'TVM-SS-5710-deeplabv3lite-mobv2-cocoseg21-512x512', },
    'ss-7618': { 'model_id': 'ss-7618', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-7618-deeplabv3lite-mobv2-qat-robokit-768x432', },
    'ss-8610': { 'model_id': 'ss-8610', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8610-deeplabv3lite-mobv2-ade20k32-512x512', },
    'ss-8690': { 'model_id': 'ss-8690', 'recommended': True, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8690-fpnlite-aspp-regnetx400mf-ade20k32-384x384', },
    '6dpose-7200': { 'model_id': '6dpose-7200', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-6DPOSE-7200-yolox-s-6d-object_pose-640x480', },
    'cl-0010': { 'model_id': 'cl-0010', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0010-mobileNetV2', },
    'cl-0080': { 'model_id': 'cl-0080', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0080-mobileNet-edgeTPU-mlperf', },
    'cl-0090': { 'model_id': 'cl-0090', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0090-efficientNet-edgeTPU-s', },
    'cl-0130': { 'model_id': 'cl-0130', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0130-efficientNet-lite0', },
    'cl-0140': { 'model_id': 'cl-0140', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0140-efficientNet-lite4', },
    'cl-0160': { 'model_id': 'cl-0160', 'recommended': False, 'shortlisted': True, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0160-resNet50V1p5-mlperf', },
    'cl-6010': { 'model_id': 'cl-6010', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6010-resNet50V1p5-mlperf-onnx', },
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
    'od-8230': { 'model_id': 'od-8230', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8230-yolox-m-lite-mmdet-coco-640x640', },
    'ss-8630': { 'model_id': 'ss-8630', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8630-unetlite-aspp-mobv2-tv-ade20k32-512x512', },
    'ss-8710': { 'model_id': 'ss-8710', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8710-deeplabv3lite-mobv2-cocoseg21-512x512', },
    'ss-8720': { 'model_id': 'ss-8720', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8720-deeplabv3lite-regnetx800mf-cocoseg21-512x512', },
    'ss-8740': { 'model_id': 'ss-8740', 'recommended': False, 'shortlisted': True, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8740-lraspp-mobV3-ti-lite-large-cocoseg21-512x512', },
    '3dod-8160': { 'model_id': '3dod-8160', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-3DOD-8160-fastbev_mod_pandaset_r18_f1_256x704_20250507', },
    '3dod-8161': { 'model_id': '3dod-8161', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-3DOD-8161-fastbev_mod_pandaset_nms_r18_f1_256x704_20250507', },
    'cl-0020': { 'model_id': 'cl-0020', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0020-SqueezeNet', },
    'cl-0038': { 'model_id': 'cl-0038', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0038-InceptionNetV1', },
    'cl-0040': { 'model_id': 'cl-0040', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0040-InceptionNetV3', },
    'cl-0050': { 'model_id': 'cl-0050', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0050-resNet50V1', },
    'cl-0060': { 'model_id': 'cl-0060', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0060-resNet50V2', },
    'cl-0070': { 'model_id': 'cl-0070', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0070-mnasNet', },
    'cl-0100': { 'model_id': 'cl-0100', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0100-efficientNet-edgeTPU-m', },
    'cl-0170': { 'model_id': 'cl-0170', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0170-efficientNet-lite1', },
    'cl-0190': { 'model_id': 'cl-0190', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0190-efficientNet-edgeTPU-l', },
    'cl-0200': { 'model_id': 'cl-0200', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0200-mobileNetV2-1p4', },
    'cl-0218': { 'model_id': 'cl-0218', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0218-mobileNetV1-qat', },
    'cl-0260': { 'model_id': 'cl-0260', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0260-mobv3-large-minimalistic', },
    'cl-0270': { 'model_id': 'cl-0270', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-CL-0270-mobv3-small-minimalistic', },
    'cl-3098': { 'model_id': 'cl-3098', 'recommended': False, 'shortlisted': False, 'session_name': 'tvmdlr', 'artifact_name': 'TVM-CL-3098-mobileNetV2-tv-qat', },
    'cl-3110': { 'model_id': 'cl-3110', 'recommended': False, 'shortlisted': False, 'session_name': 'tvmdlr', 'artifact_name': 'TVM-CL-3110-resNet50', },
    'cl-3520': { 'model_id': 'cl-3520', 'recommended': False, 'shortlisted': False, 'session_name': 'tvmdlr', 'artifact_name': 'TVM-CL-3520-mobileNetV1', },
    'cl-3530': { 'model_id': 'cl-3530', 'recommended': False, 'shortlisted': False, 'session_name': 'tvmdlr', 'artifact_name': 'TVM-CL-3530-resNet50-v1', },
    'cl-6060': { 'model_id': 'cl-6060', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6060-mobileNetV1', },
    'cl-6061': { 'model_id': 'cl-6061', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6061-mobileNetV1-512x512', },
    'cl-6062': { 'model_id': 'cl-6062', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6062-mobileNetV1-1024x1024', },
    'cl-6070': { 'model_id': 'cl-6070', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6070-mobileNetV2', },
    'cl-6078': { 'model_id': 'cl-6078', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6078-mobileNetV2-qat', },
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
    'cl-6158': { 'model_id': 'cl-6158', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6158-mobileNetV2-1p4-qat', },
    'cl-6440': { 'model_id': 'cl-6440', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6440-harDNet68', },
    'cl-6450': { 'model_id': 'cl-6450', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6450-harDNet85', },
    'cl-6460': { 'model_id': 'cl-6460', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6460-harDNet68ds', },
    'cl-6470': { 'model_id': 'cl-6470', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6470-harDNet39ds', },
    'cl-6500': { 'model_id': 'cl-6500', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6500-mobileNetV2-lite-wtv2-224', },
    'cl-6507': { 'model_id': 'cl-6507', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6507-mobileNetV2-lite-wtv2-qatv2-pert-sp2-224', },
    'cl-6508': { 'model_id': 'cl-6508', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6508-mobileNetV2-lite-wtv2-qatv2-perc-224', },
    'cl-6510': { 'model_id': 'cl-6510', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6510-mobileNetV3-large-lite-wtv2-224', },
    'cl-6520': { 'model_id': 'cl-6520', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6520-resNet18-wtv1-224', },
    'cl-6530': { 'model_id': 'cl-6530', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6530-resNet50-wtv2-224', },
    'cl-6540': { 'model_id': 'cl-6540', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6540-resNet101-wtv2-224', },
    'cl-6550': { 'model_id': 'cl-6550', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6550-resNeXt50-32x4d-wtv2-224', },
    'cl-6560': { 'model_id': 'cl-6560', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6560-resNeXt101-32x8d-wtv2-224', },
    'cl-6570': { 'model_id': 'cl-6570', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6570-regNetX-1.6gf-wtv2-224', },
    'cl-6580': { 'model_id': 'cl-6580', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6580-regNetX-400mf-wtv2-224', },
    'cl-6590': { 'model_id': 'cl-6590', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6590-regNetX-800mf-wtv2-224', },
    'cl-6600': { 'model_id': 'cl-6600', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6600-mobileNetV2-wtv2-224', },
    'cl-6700': { 'model_id': 'cl-6700', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6700-ViT-tiny-patch16-transformer-224', },
    'cl-6701': { 'model_id': 'cl-6701', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6701-ViT-base-patch16-transformer-224', },
    'cl-6710': { 'model_id': 'cl-6710', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6710-DeiT-tiny-patch16-transformer-224', },
    'cl-6711': { 'model_id': 'cl-6711', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6711-DeiT-small-patch16-transformer-224', },
    'cl-6720': { 'model_id': 'cl-6720', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6720-LeViT-128-transformer-224', },
    'cl-6721': { 'model_id': 'cl-6721', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6721-LeViT-256-transformer-224', },
    'cl-6730': { 'model_id': 'cl-6730', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6730-Swin-tiny-patch4-window7-transformer-224', },
    'cl-6731': { 'model_id': 'cl-6731', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6731-Swin-small-patch4-window7-transformer-224', },
    'cl-6732': { 'model_id': 'cl-6732', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6732-Swin-base-patch4-window7-transformer-224', },
    'cl-6740': { 'model_id': 'cl-6740', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6740-convNext-tiny-tv-224', },
    'cl-6741': { 'model_id': 'cl-6741', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6741-convNext-small-tv-224', },
    'cl-6750': { 'model_id': 'cl-6750', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6750-efficientNet-b0-224', },
    'cl-6751': { 'model_id': 'cl-6751', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6751-efficientNet-b1-240', },
    'cl-6752': { 'model_id': 'cl-6752', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6752-efficientNet-b2-288', },
    'cl-6753': { 'model_id': 'cl-6753', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6753-efficientNet-b3-300', },
    'cl-6754': { 'model_id': 'cl-6754', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6754-efficientNet-b4-380', },
    'cl-6755': { 'model_id': 'cl-6755', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6755-efficientNet-b5-456', },
    'cl-6756': { 'model_id': 'cl-6756', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6756-efficientNet-b6-528', },
    'cl-6760': { 'model_id': 'cl-6760', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6760-mobilenetV3-small-224', },
    'cl-6761': { 'model_id': 'cl-6761', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6761-mobilenetV3-large-224', },
    'cl-6770': { 'model_id': 'cl-6770', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6770-convNext-tiny-hf-224', },
    'cl-6780': { 'model_id': 'cl-6780', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6780-FastViT-s12-transformer-256', },
    'cl-6781': { 'model_id': 'cl-6781', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6781-FastViT-s12-transformer-256', },
    'cl-6790': { 'model_id': 'cl-6790', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6790-MaxVit-tiny-transformer-224', },
    'cl-6791': { 'model_id': 'cl-6791', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6791-MaxVit-small-transformer-224', },
    'cl-6800': { 'model_id': 'cl-6800', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6800-InternImage-tiny', },
    'cl-6801': { 'model_id': 'cl-6801', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6801-InternImage-small', },
    'cl-6810': { 'model_id': 'cl-6810', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6810-VAN-tiny', },
    'cl-6811': { 'model_id': 'cl-6811', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6811-VAN-small', },
    'cl-6812': { 'model_id': 'cl-6812', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6812-VAN-base', },
    'cl-6820': { 'model_id': 'cl-6820', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6820-CAFormer-s18', },
    'cl-6821': { 'model_id': 'cl-6821', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6821-CAFormer-s36', },
    'cl-6830': { 'model_id': 'cl-6830', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6830-EfficientVit_B0', },
    'cl-6831': { 'model_id': 'cl-6831', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6831-EfficientVit_B1', },
    'cl-6832': { 'model_id': 'cl-6832', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6832-EfficientVit_B2', },
    'cl-6841': { 'model_id': 'cl-6841', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6841-EfficientVit_L1', },
    'cl-6842': { 'model_id': 'cl-6842', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6842-EfficientVit_L2', },
    'cl-6855': { 'model_id': 'cl-6855', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6855-EfficientVit_M5', },
    'cl-6860': { 'model_id': 'cl-6860', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6860-ResNet50-V2-PreBN', },
    'cl-6870': { 'model_id': 'cl-6870', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6870-EfficientNet-V2-S-384', },
    'cl-6871': { 'model_id': 'cl-6871', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-CL-6871-EfficientNet-V2-M-480', },
    'de-7300': { 'model_id': 'de-7300', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-DE-7300-depth-est-fast-depth-224x224', },
    'de-7310': { 'model_id': 'de-7310', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-DE-7310-depth-est-midas-small-256x256', },
    'kd-7070': { 'model_id': 'kd-7070', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-KD-7070-yoloxpose_tiny_lite_416x416_20240808_model', },
    'kd-7080': { 'model_id': 'kd-7080', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-KD-7080-yoloxpose_s_lite_coco-640x640_20250119_model', },
    'od-2060': { 'model_id': 'od-2060', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2060-ssdlite-mobV2-coco-300x300', },
    'od-2070': { 'model_id': 'od-2070', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2070-ssd-mobV1-fpn-coco-tpu-8-640x640', },
    'od-2090': { 'model_id': 'od-2090', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2090-ssd-mobV2-fpnlite-coco-tpu-8-640x640', },
    'od-2100': { 'model_id': 'od-2100', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2100-ssd-res50V1-fpn-coco-tpu-8-640x640', },
    'od-2110': { 'model_id': 'od-2110', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2110-efficientDet-ti-lite0-coco-512x512', },
    'od-2130': { 'model_id': 'od-2130', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2130-ssd-mobV2-coco-tpu-8-320x320', },
    'od-2150': { 'model_id': 'od-2150', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2150-efficientDet-lite1-relu-coco-384x384', },
    'od-2170': { 'model_id': 'od-2170', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-OD-2170-efficientDet-lite3-relu-coco-512x512', },
    'od-5100': { 'model_id': 'od-5100', 'recommended': False, 'shortlisted': False, 'session_name': 'tvmdlr', 'artifact_name': 'TVM-OD-5100-ssd-mobV1-coco-mlperf-300x300', },
    'od-8000': { 'model_id': 'od-8000', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8000-ssd1200-resNet34-mlperf-coco-1200x1200', },
    'od-8030': { 'model_id': 'od-8030', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8030-ssd-lite-mobv2-fpn-mmdet-coco-512x512', },
    'od-8050': { 'model_id': 'od-8050', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8050-ssd-lite-regNetX-800mf-fpn-bgr-mmdet-coco-512x512', },
    'od-8060': { 'model_id': 'od-8060', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8060-ssd-lite-regNetX-1.6gf-fpn-bgr-mmdet-coco-768x768', },
    'od-8070': { 'model_id': 'od-8070', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8070-yolov3-d53-relu-mmdet-coco-416x416', },
    'od-8080': { 'model_id': 'od-8080', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8080-yolov3-lite-regNetX-1.6gf-bgr-mmdet-coco-512x512', },
    'od-8090': { 'model_id': 'od-8090', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8090-retina-lite-regNetX-800mf-fpn-bgr-mmdet-coco-512x512', },
    'od-8260': { 'model_id': 'od-8260', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8260-yolox-femto-lite-mmdet-coco-320x320', },
    'od-8421': { 'model_id': 'od-8421', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8421-yolox_s_lite_1024x1024_20220317_model', },
    'od-8920': { 'model_id': 'od-8920', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8920-DETR-fb-resnet50-transformer-coco-800x800', },
    'od-8930': { 'model_id': 'od-8930', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8930-FCOS-r50-fpn-gn-head-coco-1216x800', },
    'od-8940': { 'model_id': 'od-8940', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8940-CenterNet-update-r50-fpn-coco-1216x800', },
    'od-8950': { 'model_id': 'od-8950', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8950-CenterNet-r18-coco-672x448', },
    'od-8970': { 'model_id': 'od-8970', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-8970-efficientDet-b0-bifpn-lite-coco-512x512', },
    'od-9202': { 'model_id': 'od-9202', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-9202-yolov7-l-lite-mmdet-coco-640x640', },
    'od-9203': { 'model_id': 'od-9203', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-OD-9203-yolov7-l-mmdet-coco-640x640', },
    'ss-2540': { 'model_id': 'ss-2540', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-SS-2540-deeplabv3-mobv2-ade20k-512x512', },
    'ss-2590': { 'model_id': 'ss-2590', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512', },
    'ss-2600': { 'model_id': 'ss-2600', 'recommended': False, 'shortlisted': False, 'session_name': 'tflitert', 'artifact_name': 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512', },
    'ss-8650': { 'model_id': 'ss-8650', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8650-fpnlite-aspp-mobv2-ade20k32-512x512', },
    'ss-8670': { 'model_id': 'ss-8670', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512', },
    'ss-8700': { 'model_id': 'ss-8700', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8700-fpnlite-aspp-regnetx800mf-ade20k32-512x512', },
    'ss-8730': { 'model_id': 'ss-8730', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8730-deeplabv3-mobv3-lite-large-cocoseg21-512x512', },
    'ss-8750': { 'model_id': 'ss-8750', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8750-segformerB0-transformer-ade-512x512', },
    'ss-8760': { 'model_id': 'ss-8760', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8760-segformerB1-transformer-ade-512x512', },
    'ss-8770': { 'model_id': 'ss-8770', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-SS-8770-segformerB2-transformer-ade-512x512', },
    'visloc-7500': { 'model_id': 'visloc-7500', 'recommended': False, 'shortlisted': False, 'session_name': 'onnxrt', 'artifact_name': 'ONR-VISLOC-7500-tiad-dkaze-carla-768x384', },
}