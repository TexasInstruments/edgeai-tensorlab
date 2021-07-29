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
import os

#mapping from artifacts id to readable model names
#ver:13-2021-04-01
model_id_artifacts_pair = {
    # TFLite CL
    'vcls-10-010-0_tflitert': 'TFL-CL-0000-mobileNetV1-mlperf',
    'vcls-10-401-0_tflitert': 'TFL-CL-0010-mobileNetV2',
    'vcls-10-403-0_tflitert': 'TFL-CL-0020-SqueezeNet',
    'vcls-10-405-8_tflitert': 'TFL-CL-0030-InceptionNetV1',
    'vcls-10-406-0_tflitert': 'TFL-CL-0040-InceptionNetV3',
    'vcls-10-409-0_tflitert': 'TFL-CL-0050-resNet50V1',
    'vcls-10-410-0_tflitert': 'TFL-CL-0060-resNet50V2',
    'vcls-10-407-0_tflitert': 'TFL-CL-0070-mnasNet',
    'vcls-10-011-0_tflitert': 'TFL-CL-0080-mobileNet-edgeTPU-mlperf',
    'vcls-10-440-0_tflitert': 'TFL-CL-0090-efficientNet-edgeTPU-s',
    'vcls-10-441-0_tflitert': 'TFL-CL-0100-efficientNet-edgeTPU-m',
    'vcls-10-430-0_tflitert': 'TFL-CL-0130-efficientNet-lite0',
    'vcls-10-434-0_tflitert': 'TFL-CL-0140-efficientNet-lite4',
    'vcls-10-404-0_tflitert': 'TFL-CL-0150-denseNet',
    'vcls-10-012-0_tflitert': 'TFL-CL-0160-resNet50V1p5-mlperf',
    'vcls-10-431-0_tflitert': 'TFL-CL-0170-efficientNet-lite1',
    'vcls-10-432-0_tflitert': 'TFL-CL-0180-efficientNet-lite2',
    'vcls-10-442-0_tflitert': 'TFL-CL-0190-efficientNet-edgeTPU-l',
    'vcls-10-402-0_tflitert': 'TFL-CL-0200-mobileNetV2-1p4',
    'vcls-10-400-0_tflitert': 'TFL-CL-0210-mobileNetV1', 
    'vcls-10-400-8_tflitert': 'TFL-CL-0220-mobileNetV1-qat', 
    'vcls-10-401-8_tflitert': 'TFL-CL-0230-mobileNetV2-qat', 
    'vcls-10-408-0_tflitert': 'TFL-CL-0240-nasNet-mobile-tflite',
    'vcls-10-450-0_tflitert': 'TFL-CL-0250-xceptionNet-tflite', # mxnet model replaced with with tflite model now

    # TFLite OD
    'vdet-12-010-0_tflitert': 'TFL-OD-2000-ssd-mobV1-coco-mlperf-300x300',
    'vdet-12-011-0_tflitert': 'TFL-OD-2010-ssd-mobV2-coco-mlperf-300x300', 
    'vdet-12-400-0_tflitert': 'TFL-OD-2020-ssdLite-mobDet-DSP-coco-320x320',
    'vdet-12-401-0_tflitert': 'TFL-OD-2030-ssdLite-mobDet-EdgeTPU-coco-320x320',
    'vdet-12-404-0_tflitert': 'TFL-OD-2040-ssd-mobV1-FPN-coco-640x640',
    'vdet-12-403-0_tflitert': 'TFL-OD-2050-ssd-mobV2-mnas-fpn-coco-320x320',
    'vdet-12-402-0_tflitert': 'TFL-OD-2060-ssd-mobV2-coco-mlperf-300x300',

    # TFLite SS
    'vseg-17-010-0_tflitert': 'TFL-SS-2500-deeplab-mobV2-ade20k-512x512',
    'vseg-17-400-0_tflitert': 'TFL-SS-2540-deeplabv3-mobv2-ade20k-512x512',
    'vseg-16-400-0_tflitert': 'TFL-SS-2550-deeplabv3-mobv2_cs-2048x1024',
    'vseg-18-010-0_tflitert': 'TFL-SS-2580-deeplabv3_mobv2-ade20k32-mlperf-512x512',
    'vseg-19-400-0_tflitert': 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512',
    'vseg-19-401-0_tflitert': 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512',

    # TVM- CL
    'vcls-10-020-0_tvmdlr': 'TVM-CL-3000-resNet18V2',
    'vcls-10-450-0_tvmdlr': 'TVM-CL-3020-xceptionNet-mxnet',
    'vcls-10-408-0_tvmdlr': 'TVM-CL-3040-nasNet-mobile-tflite',
    'vcls-10-100-0_tvmdlr': 'TVM-CL-3060-mobileNetV1',
    'vcls-10-101-0_tvmdlr': 'TVM-CL-3070-mobileNetV2',
    'vcls-10-301-0_tvmdlr': 'TVM-CL-3080-shuffleNetV2',
    'vcls-10-302-0_tvmdlr': 'TVM-CL-3090-mobileNetV2-tv',
    'vcls-10-304-0_tvmdlr': 'TVM-CL-3100-resNet18',
    'vcls-10-305-0_tvmdlr': 'TVM-CL-3110-resNet50',
    'vcls-10-031-0_tvmdlr': 'TVM-CL-3120-regNetX-400mf',
    'vcls-10-032-0_tvmdlr': 'TVM-CL-3130-regNetX-800mf',
    'vcls-10-033-0_tvmdlr': 'TVM-CL-3140-regNetX-1.6gf',
    'vcls-10-102-8_tvmdlr': 'TVM-CL-3150-mobileNetV2-1p4-qat',
    
    #512x512
    'vcls-10-100-1_tvmdlr': 'TVM-CL-3160-mobileNetV1-512x512',
    'vcls-10-101-1_tvmdlr': 'TVM-CL-3170-mobileNetV2-512x512',
    'vcls-10-301-1_tvmdlr': 'TVM-CL-3180-shuffleNetV2-512x512',
    'vcls-10-302-1_tvmdlr': 'TVM-CL-3190-mobileNetV2-tv-512x512',
    'vcls-10-304-1_tvmdlr': 'TVM-CL-3200-resNet18-512x512',
    'vcls-10-305-1_tvmdlr': 'TVM-CL-3210-resNet50-512x512',
    'vcls-10-031-1_tvmdlr': 'TVM-CL-3220-regNetX-400mf-512x512',
    'vcls-10-032-1_tvmdlr': 'TVM-CL-3230-regNetX-800mf-512x512',
    'vcls-10-033-1_tvmdlr': 'TVM-CL-3240-regNetX-1.6gf-512x512',
    'vcls-10-102-1_tvmdlr': 'TVM-CL-3250-mobileNetV2-1p4-qat-512x512',

    #1024x1024
    'vcls-10-100-2_tvmdlr': 'TVM-CL-3260-mobileNetV1-1024x1024',
    'vcls-10-101-2_tvmdlr': 'TVM-CL-3270-mobileNetV2-1024x1024',
    'vcls-10-301-2_tvmdlr': 'TVM-CL-3280-shuffleNetV2-1024x1024',
    'vcls-10-302-2_tvmdlr': 'TVM-CL-3290-mobileNetV2-tv-1024x1024',
    'vcls-10-304-2_tvmdlr': 'TVM-CL-3300-resNet18-1024x1024',
    'vcls-10-305-2_tvmdlr': 'TVM-CL-3310-resNet50-1024x1024',
    'vcls-10-031-2_tvmdlr': 'TVM-CL-3320-regNetX-400mf-1024x1024',
    'vcls-10-032-2_tvmdlr': 'TVM-CL-3330-regNetX-800mf-1024x1024',
    'vcls-10-033-2_tvmdlr': 'TVM-CL-3340-regNetX-1.6gf-1024x1024',
    'vcls-10-102-2_tvmdlr': 'TVM-CL-3350-mobileNetV2-1p4-qat-1024x1024',
    ########
    
    'vcls-10-030-0_tvmdlr': 'TVM-CL-3360-regNetx-200mf',
    'vcls-10-306-0_tvmdlr': 'TVM-CL-3370-vgg16',
    'vcls-10-101-8_tvmdlr': 'TVM-CL-3380-mobileNetV2-qat',
    'vcls-10-302-8_tvmdlr': 'TVM-CL-3400-mobileNetV2-tv-qat',
    'vcls-10-060-0_tvmdlr': 'TVM-CL-3410-gluoncv-mxnet-mobv2',
    'vcls-10-061-0_tvmdlr': 'TVM-CL-3420-gluoncv-mxnet-resNet50-v1',
    'vcls-10-062-0_tvmdlr': 'TVM-CL-3430-gluoncv-mxnet-xception',
    # HarD Net
    'vcls-10-900-0_tvmdlr': 'TVM-CL-3440-harDNet68',
    'vcls-10-901-0_tvmdlr': 'TVM-CL-3450-harDNet85',
    'vcls-10-902-0_tvmdlr': 'TVM-CL-3460-harDNet68ds',
    'vcls-10-903-0_tvmdlr': 'TVM-CL-3470-harDNet39ds',
    
    # TVM - OD
    'vdet-12-012-0_tvmdlr': 'TVM-OD-5000-ssd1200-resNet34-mlperf-1200x1200',
    'vdet-12-020-0_tvmdlr': 'TVM-OD-5010-yolov3-416x416',
    'vdet-12-060-0_tvmdlr': 'TVM-OD-5020-yolov3-mobv1-gluon-mxnet-416x416',
    'vdet-12-061-0_tvmdlr': 'TVM-OD-5030-ssd-mobv1-gluon-mxnet-512x512',
    # TVM - SS - CS
    'vseg-16-100-0_tvmdlr': 'TVM-SS-5500-deeplabv3lite-mobv2-cs-768x384',
    'vseg-16-100-8_tvmdlr': 'TVM-SS-5510-deeplabv3lite-mobv2-cs-qat-768x384',
    'vseg-16-101-0_tvmdlr': 'TVM-SS-5520-fpnlite-aspp-mobv2-cs-768x384',
    'vseg-16-101-8_tvmdlr': 'TVM-SS-5530-fpnlite-aspp-mobv2-cs-qat-768x384',
    'vseg-16-102-0_tvmdlr': 'TVM-SS-5540-unetlite-aspp-mobv2-tv-cs-768x384',
    'vseg-16-102-8_tvmdlr': 'TVM-SS-5550-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'vseg-16-103-0_tvmdlr': 'TVM-SS-5560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'vseg-16-104-0_tvmdlr': 'TVM-SS-5570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'vseg-16-105-0_tvmdlr': 'TVM-SS-5580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'vseg-16-300-0_tvmdlr': 'TVM-SS-5590-deeplabv3-res50-1040x520',
    'vseg-16-301-0_tvmdlr': 'TVM-SS-5600-fcn-res50-1040x520',

    # TVM - SS - ADE20k
    'vseg-18-100-0_tvmdlr': 'TVM-SS-5610-deeplabv3lite-mobv2-ade20k32-512x512',
    'vseg-18-100-8_tvmdlr': 'TVM-SS-5620-deeplabv3lite-mobv2-ade20k32-qat-512x512', 
    'vseg-18-101-0_tvmdlr': 'TVM-SS-5630-unetlite-aspp-mobv2-tv-ade20k32-512x512',
    'vseg-18-101-8_tvmdlr': 'TVM-SS-5640-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512',
    'vseg-18-102-0_tvmdlr': 'TVM-SS-5650-fpnlite-aspp-mobv2-ade20k32-512x512',
    'vseg-18-102-8_tvmdlr': 'TVM-SS-5660-fpnlite-aspp-mobv2-ade20k32-qat-512x512', 
    'vseg-18-103-0_tvmdlr': 'TVM-SS-5670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512',
    'vseg-18-103-8_tvmdlr': 'TVM-SS-5680-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512',
    'vseg-18-110-0_tvmdlr': 'TVM-SS-5690-fpnlite-aspp-regnetx400mf-ade20k32-384x384',
    'vseg-18-111-0_tvmdlr': 'TVM-SS-5700-fpnlite-aspp-regnetx800mf-ade20k32-512x512',

    # TVM -SS - CocoSeg21
    'vseg-21-100-0_tvmdlr' : 'TVM-SS-5710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'vseg-21-110-0_tvmdlr' : 'TVM-SS-5720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',

    # ONNXRT- CL
    'vcls-10-020-0_onnxrt': 'ONR-CL-6000-resNet18V2',
    'vcls-10-408-0_onnxrt': 'ONR-CL-6040-nasNet-mobile-tflite',
    'vcls-10-100-0_onnxrt': 'ONR-CL-6060-mobileNetV1',
    'vcls-10-101-0_onnxrt': 'ONR-CL-6070-mobileNetV2',
    'vcls-10-301-0_onnxrt': 'ONR-CL-6080-shuffleNetV2',
    'vcls-10-302-0_onnxrt': 'ONR-CL-6090-mobileNetV2-tv',
    'vcls-10-304-0_onnxrt': 'ONR-CL-6100-resNet18',
    'vcls-10-305-0_onnxrt': 'ONR-CL-6110-resNet50',
    'vcls-10-031-0_onnxrt': 'ONR-CL-6120-regNetX-400mf',
    'vcls-10-032-0_onnxrt': 'ONR-CL-6130-regNetX-800mf',
    'vcls-10-033-0_onnxrt': 'ONR-CL-6140-regNetX-1.6gf',
    'vcls-10-102-8_onnxrt': 'ONR-CL-6150-mobileNetV2-1p4-qat',
    
    #512x512
    'vcls-10-100-1_onnxrt': 'ONR-CL-6160-mobileNetV1-512x512',
    'vcls-10-101-1_onnxrt': 'ONR-CL-6170-mobileNetV2-512x512',
    'vcls-10-301-1_onnxrt': 'ONR-CL-6180-shuffleNetV2-512x512',
    'vcls-10-302-1_onnxrt': 'ONR-CL-6190-mobileNetV2-tv-512x512',
    'vcls-10-304-1_onnxrt': 'ONR-CL-6200-resNet18-512x512',
    'vcls-10-305-1_onnxrt': 'ONR-CL-6210-resNet50-512x512',
    'vcls-10-031-1_onnxrt': 'ONR-CL-6220-regNetX-400mf-512x512',
    'vcls-10-032-1_onnxrt': 'ONR-CL-6230-regNetX-800mf-512x512',
    'vcls-10-033-1_onnxrt': 'ONR-CL-6240-regNetX-1.6gf-512x512',
    'vcls-10-102-1_onnxrt': 'ONR-CL-6250-mobileNetV2-1p4-qat-512x512',

    #1024x1024
    'vcls-10-100-2_onnxrt': 'ONR-CL-6260-mobileNetV1-1024x1024',
    'vcls-10-101-2_onnxrt': 'ONR-CL-6270-mobileNetV2-1024x1024',
    'vcls-10-301-2_onnxrt': 'ONR-CL-6280-shuffleNetV2-1024x1024',
    'vcls-10-302-2_onnxrt': 'ONR-CL-6290-mobileNetV2-tv-1024x1024',
    'vcls-10-304-2_onnxrt': 'ONR-CL-6300-resNet18-1024x1024',
    'vcls-10-305-2_onnxrt': 'ONR-CL-6310-resNet50-1024x1024',
    'vcls-10-031-2_onnxrt': 'ONR-CL-6320-regNetX-400mf-1024x1024',
    'vcls-10-032-2_onnxrt': 'ONR-CL-6330-regNetX-800mf-1024x1024',
    'vcls-10-033-2_onnxrt': 'ONR-CL-6340-regNetX-1.6gf-1024x1024',
    'vcls-10-102-2_onnxrt': 'ONR-CL-6350-mobileNetV2-1p4-qat-1024x1024',
    ########
    
    'vcls-10-030-0_onnxrt': 'ONR-CL-6360-regNetx-200mf',
    'vcls-10-306-0_onnxrt': 'ONR-CL-6370-vgg16',
    'vcls-10-101-8_onnxrt': 'ONR-CL-6380-mobileNetV2-qat',
    'vcls-10-302-8_onnxrt': 'ONR-CL-6400-mobileNetV2-tv-qat',
    'vcls-10-060-0_onnxrt': 'ONR-CL-6410-gluoncv-mxnet-mobv2',
    'vcls-10-061-0_onnxrt': 'ONR-CL-6420-gluoncv-mxnet-resNet50-v1',
    'vcls-10-062-0_onnxrt': 'ONR-CL-6430-gluoncv-mxnet-xception',
    # HarD Net
    'vcls-10-900-0_onnxrt': 'ONR-CL-6440-harDNet68',
    'vcls-10-901-0_onnxrt': 'ONR-CL-6450-harDNet85',
    'vcls-10-902-0_onnxrt': 'ONR-CL-6460-harDNet68ds',
    'vcls-10-903-0_onnxrt': 'ONR-CL-6470-harDNet39ds',
    
    # ONNX - OD
    'vdet-12-012-0_onnxrt': 'ONR-OD-8000-ssd1200-resNet34-mlperf-1200x1200',
    'vdet-12-020-0_onnxrt': 'ONR-OD-8010-yolov3-416x416',

    # ONNX - SS - CS
    'vseg-16-100-0_onnxrt': 'ONR-SS-8500-deeplabv3lite-mobv2-cs-768x384',
    'vseg-16-100-8_onnxrt': 'ONR-SS-8510-deeplabv3lite-mobv2-cs-qat-768x384',
    'vseg-16-101-0_onnxrt': 'ONR-SS-8520-fpnlite-aspp-mobv2-cs-768x384',
    'vseg-16-101-8_onnxrt': 'ONR-SS-8530-fpnlite-aspp-mobv2-cs-qat-768x384',
    'vseg-16-102-0_onnxrt': 'ONR-SS-8540-unetlite-aspp-mobv2-tv-cs-768x384',
    'vseg-16-102-8_onnxrt': 'ONR-SS-8550-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'vseg-16-103-0_onnxrt': 'ONR-SS-8560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'vseg-16-104-0_onnxrt': 'ONR-SS-8570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'vseg-16-105-0_onnxrt': 'ONR-SS-8580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'vseg-16-300-0_onnxrt': 'ONR-SS-8590-deeplabv3-res50-1040x520',
    'vseg-16-301-0_onnxrt': 'ONR-SS-8600-fcn-res50-1040x520',

    # ONNX - SS - ADE20k
    'vseg-18-100-0_onnxrt': 'ONR-SS-8610-deeplabv3lite-mobv2-ade20k32-512x512',
    'vseg-18-100-8_onnxrt': 'ONR-SS-8620-deeplabv3lite-mobv2-ade20k32-qat-512x512', 
    'vseg-18-101-0_onnxrt': 'ONR-SS-8630-unetlite-aspp-mobv2-tv-ade20k32-512x512',
    'vseg-18-101-8_onnxrt': 'ONR-SS-8640-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512',
    'vseg-18-102-0_onnxrt': 'ONR-SS-8650-fpnlite-aspp-mobv2-ade20k32-512x512',
    'vseg-18-102-8_onnxrt': 'ONR-SS-8660-fpnlite-aspp-mobv2-ade20k32-qat-512x512', 
    'vseg-18-103-0_onnxrt': 'ONR-SS-8670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512',
    'vseg-18-103-8_onnxrt': 'ONR-SS-8680-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512',
    'vseg-18-110-0_onnxrt': 'ONR-SS-8690-fpnlite-aspp-regnetx400mf-ade20k32-384x384',
    'vseg-18-111-0_onnxrt': 'ONR-SS-8700-fpnlite-aspp-regnetx800mf-ade20k32-512x512',

    # ONNX-SS - CocoSeg21
    'vseg-21-100-0_onnxrt' : 'ONR-SS-8710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'vseg-21-110-0_onnxrt' : 'ONR-SS-8720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',
}

removed_model_list = {
    'vcls-10-450-0_tvmdlr' : 'TVM-CL-3020-xceptionNet-mxnet', # this is replaced with tflite model now (that was also eventually removed)
    'vcls-10-401-8_tflitert': 'TFL-CL-0230-mobileNetV2-qat',  # QAT model is not giving good accuracy so keep only float
    'vdet-12-012-0_tvmdlr': 'TVM-OD-5000-ssd1200-resNet34-mlperf-1200x1200', # Not working with TVM. Will need to park it till ONNX RT OD support is available.
    #'vdet-12-012-0_onnxrt': 'ONR-OD-800-ssd1200-resNet34-mlperf-1200x1200', #ONNX does not support OD yet
    ################ CS models
    'vseg-16-100-0_tvmdlr': 'TVM-SS-5500-deeplabv3lite-mobv2-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-100-8_tvmdlr': 'TVM-SS-5510-deeplabv3lite-mobv2-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-101-0_tvmdlr': 'TVM-SS-5520-fpnlite-aspp-mobv2-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-101-8_tvmdlr': 'TVM-SS-5530-fpnlite-aspp-mobv2-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-102-0_tvmdlr': 'TVM-SS-5540-unetlite-aspp-mobv2-tv-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-102-8_tvmdlr': 'TVM-SS-5550-unetlite-aspp-mobv2-tv-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-103-0_tvmdlr': 'TVM-SS-5560-fpnlite-aspp-regNetx800mf-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-104-0_tvmdlr': 'TVM-SS-5570-fpnlite-aspp-regNetx1.6gf-cs-1024x512', # cityscapes model not part of Model Zoo
    'vseg-16-105-0_tvmdlr': 'TVM-SS-5580-fpnlite-aspp-regNetx3.2gf-cs-1536x768', # cityscapes model not part of Model Zoo

    'vseg-16-100-0_onnxrt': 'ONR-SS-8500-deeplabv3lite-mobv2-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-100-8_onnxrt': 'ONR-SS-8510-deeplabv3lite-mobv2-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-101-0_onnxrt': 'ONR-SS-8520-fpnlite-aspp-mobv2-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-101-8_onnxrt': 'ONR-SS-8530-fpnlite-aspp-mobv2-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-102-0_onnxrt': 'ONR-SS-8540-unetlite-aspp-mobv2-tv-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-102-8_onnxrt': 'ONR-SS-8550-unetlite-aspp-mobv2-tv-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-103-0_onnxrt': 'ONR-SS-8560-fpnlite-aspp-regNetx800mf-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-104-0_onnxrt': 'ONR-SS-8570-fpnlite-aspp-regNetx1.6gf-cs-1024x512', # cityscapes model not part of Model Zoo
    'vseg-16-105-0_onnxrt': 'ONR-SS-8580-fpnlite-aspp-regNetx3.2gf-cs-1536x768', # cityscapes model not part of Model Zoo
    ################
    'vcls-10-408-0_tflitert': 'TFL-CL-0240-nasNet-mobile-tflite', # Kumar removed model (Multiple sub-graphs) (Accuracy issue)
    'vcls-10-432-0_tflitert': 'TFL-CL-0180-efficientNet-lite2', # Kumar removed model  (Accuracy issue)
    'vdet-12-011-0_tflitert': 'TFL-OD-2010-ssd-mobV2-coco-300x300-mlperf',  # Kumar removed model (Duplocate)
    'vseg-16-400-0_tflitert': 'TFL-SS-2550-deeplabv3-mobv2_cs-2048x1024',  # Kumar removed model, (not in plan)

    'vcls-10-306-0_tvmdlr': 'TVM-CL-3370-vgg16', # Kumar removed model
    'vcls-10-020-0_tvmdlr': 'TVM-CL-3000-resNet18V2', # Kumar removed model
    'vseg-16-300-0_tvmdlr': 'TVM-SS-5590-deeplabv3-res50-1040x520', # Kumar removed model, nc does not have info for this
    'vseg-16-301-0_tvmdlr': 'TVM-SS-5600-fcn-res50-1040x520', # Kumar removed model

    'vcls-10-306-0_onnxrt': 'ONR-CL-6370-vgg16', # Kumar removed model for TVM 
    'vcls-10-020-0_onnxrt': 'ONR-CL-6000-resNet18V2', # Kumar removed model  for TVM 
    'vseg-16-300-0_onnxrt': 'ONR-SS-8590-deeplabv3-res50-1040x520', # Kumar removed model, nc does not have info for this  for TVM 
    'vseg-16-301-0_onnxrt': 'ONR-SS-8600-fcn-res50-1040x520', # Kumar removed model  for TVM 
    
    #########################
    'vseg-17-010-0_tflitert': 'TFL-SS-2500-deeplab-mobV2-ade20k-512x512', # Manu said incorrect model ID removed. vseg-17-010 is replaced with vseg-18-010
    'vcls-10-450-0_tflitert': 'TFL-CL-0250-xceptionNet-tflite', # mxnet model replaced with with tflite model now. Eventually removed as size is quite big.
    'vdet-12-404-0_tflitert': 'TFL-OD-2040-ssd-mobV1-FPN-coco-640x640', # does not run, import crashes. Manu
    'vdet-12-403-0_tflitert': 'TFL-OD-2050-ssd-mobV2-mnas-fpn-coco-320x320', # does not run, import crashes. Manu

    'vcls-10-408-0_tvmdlr': 'TVM-CL-3040-nasNet-mobile-tflite', # not part of benchmarking script yet. tflite model with TVM.
    'vdet-12-020-0_tvmdlr': 'TVM-OD-5010-yolov3-416x416', # not supported yet

    'vcls-10-408-0_onnxrt': 'ONR-CL-6040-nasNet-mobile-tflite', # not part of benchmarking script yet. tflite model with TVM.
    'vdet-12-020-0_onnxrt': 'ONR-OD-8010-yolov3-416x416', # not supported yet
    'vcls-10-404-0_tflitert': 'TFL-CL-0150-denseNet', # too far from optimal pareto line
    'vdet-12-060-0_tvmdlr': 'TVM-OD-5020-yolov3-mobv1-gluon-mxnet-416x416', #segmentation fault while running import

    #removed from ONNX-RT
    'vcls-10-060-0_onnxrt': 'ONR-CL-6410-gluoncv-mxnet-mobv2',
    'vcls-10-061-0_onnxrt': 'ONR-CL-6420-gluoncv-mxnet-resNet50-v1',
    'vcls-10-062-0_onnxrt': 'ONR-CL-6430-gluoncv-mxnet-xception',

    #ADE20k32 models
    'vseg-18-100-8_tvmdlr': 'TVM-SS-5620-deeplabv3lite-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-18-101-8_tvmdlr': 'TVM-SS-5640-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512', # import fails
    'vseg-18-102-8_tvmdlr': 'TVM-SS-5660-fpnlite-aspp-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-18-103-8_tvmdlr': 'TVM-SS-5680-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed

    'vseg-18-100-8_onnxrt': 'ONR-SS-8620-deeplabv3lite-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-18-101-8_onnxrt': 'ONR-SS-8640-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512', # import fails
    'vseg-18-102-8_onnxrt': 'ONR-SS-8660-fpnlite-aspp-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-18-103-8_onnxrt': 'ONR-SS-8680-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed

    #512x512 (Only for performance)
    'vcls-10-100-1_tvmdlr': 'TVM-CL-3160-mobileNetV1-512x512',
    'vcls-10-101-1_tvmdlr': 'TVM-CL-3170-mobileNetV2-512x512',
    'vcls-10-301-1_tvmdlr': 'TVM-CL-3180-shuffleNetV2-512x512',
    'vcls-10-302-1_tvmdlr': 'TVM-CL-3190-mobileNetV2-tv-512x512',
    'vcls-10-304-1_tvmdlr': 'TVM-CL-3200-resNet18-512x512',
    'vcls-10-305-1_tvmdlr': 'TVM-CL-3210-resNet50-512x512',
    'vcls-10-031-1_tvmdlr': 'TVM-CL-3220-regNetX-400mf-512x512',
    'vcls-10-032-1_tvmdlr': 'TVM-CL-3230-regNetX-800mf-512x512',
    'vcls-10-033-1_tvmdlr': 'TVM-CL-3240-regNetX-1.6gf-512x512',
    'vcls-10-102-1_tvmdlr': 'TVM-CL-3250-mobileNetV2-1p4-qat-512x512',
    
    'vcls-10-100-1_onnxrt': 'ONR-CL-6160-mobileNetV1-512x512',
    'vcls-10-101-1_onnxrt': 'ONR-CL-6170-mobileNetV2-512x512',
    'vcls-10-301-1_onnxrt': 'ONR-CL-6180-shuffleNetV2-512x512',
    'vcls-10-302-1_onnxrt': 'ONR-CL-6190-mobileNetV2-tv-512x512',
    'vcls-10-304-1_onnxrt': 'ONR-CL-6200-resNet18-512x512',
    'vcls-10-305-1_onnxrt': 'ONR-CL-6210-resNet50-512x512',
    'vcls-10-031-1_onnxrt': 'ONR-CL-6220-regNetX-400mf-512x512',
    'vcls-10-032-1_onnxrt': 'ONR-CL-6230-regNetX-800mf-512x512',
    'vcls-10-033-1_onnxrt': 'ONR-CL-6240-regNetX-1.6gf-512x512',
    'vcls-10-102-1_onnxrt': 'ONR-CL-6250-mobileNetV2-1p4-qat-512x512',


    #1024x1024  (Only for performance)
    'vcls-10-100-2_tvmdlr': 'TVM-CL-3260-mobileNetV1-1024x1024',
    'vcls-10-101-2_tvmdlr': 'TVM-CL-3270-mobileNetV2-1024x1024',
    'vcls-10-301-2_tvmdlr': 'TVM-CL-3280-shuffleNetV2-1024x1024',
    'vcls-10-302-2_tvmdlr': 'TVM-CL-3290-mobileNetV2-tv-1024x1024',
    'vcls-10-304-2_tvmdlr': 'TVM-CL-3300-resNet18-1024x1024',
    'vcls-10-305-2_tvmdlr': 'TVM-CL-3310-resNet50-1024x1024',
    'vcls-10-031-2_tvmdlr': 'TVM-CL-3320-regNetX-400mf-1024x1024',
    'vcls-10-032-2_tvmdlr': 'TVM-CL-3330-regNetX-800mf-1024x1024',
    'vcls-10-033-2_tvmdlr': 'TVM-CL-3340-regNetX-1.6gf-1024x1024',
    'vcls-10-102-2_tvmdlr': 'TVM-CL-3350-mobileNetV2-1p4-qat-1024x1024',

    'vcls-10-100-2_onnxrt': 'ONR-CL-6260-mobileNetV1-1024x1024',
    'vcls-10-101-2_onnxrt': 'ONR-CL-6270-mobileNetV2-1024x1024',
    'vcls-10-301-2_onnxrt': 'ONR-CL-6280-shuffleNetV2-1024x1024',
    'vcls-10-302-2_onnxrt': 'ONR-CL-6290-mobileNetV2-tv-1024x1024',
    'vcls-10-304-2_onnxrt': 'ONR-CL-6300-resNet18-1024x1024',
    'vcls-10-305-2_onnxrt': 'ONR-CL-6310-resNet50-1024x1024',
    'vcls-10-031-2_onnxrt': 'ONR-CL-6320-regNetX-400mf-1024x1024',
    'vcls-10-032-2_onnxrt': 'ONR-CL-6330-regNetX-800mf-1024x1024',
    'vcls-10-033-2_onnxrt': 'ONR-CL-6340-regNetX-1.6gf-1024x1024',
    'vcls-10-102-2_onnxrt': 'ONR-CL-6350-mobileNetV2-1p4-qat-1024x1024',

    # ONNX-SS - CocoSeg21 Models are not added yet
    'vseg-21-100-0_onnxrt' : 'ONR-SS-8710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'vseg-21-110-0_onnxrt' : 'ONR-SS-8720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',

    # TVM-SS-CocoSeg21 Models are not added yet
    'vseg-21-100-0_tvmdlr' : 'TVM-SS-5710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'vseg-21-110-0_tvmdlr' : 'TVM-SS-5720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',

    #removed low accuracy models where drop is more wrt reference accuracy. This will be corrected in the next release.
    'vseg-17-400-0_tflitert' : 'TFL-SS-2540-deeplabv3-mobv2-ade20k-512x512',
    'vseg-19-400-0_tflitert' : 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512',
    'vcls-10-410-0_tflitert' : 'TFL-CL-0060-resNet50V2',
    'vseg-19-401-0_tflitert' : 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512',
    'vcls-10-400-8_tflitert' : 'TFL-CL-0220-mobileNetV1-qat',
    'vcls-10-401-0_tflitert' : 'TFL-CL-0010-mobileNetV2',
    'vcls-10-301-0_onnxrt' : 'ONR-CL-6080-shuffleNetV2',
    'vcls-10-402-0_tflitert' : 'TFL-CL-0200-mobileNetV2-1p4',
    'vcls-10-062-0_tvmdlr' : 'TVM-CL-3430-gluoncv-mxnet-xception',

}

removed_models_from_plots = {
    'vseg-19-400-0_tflitert': 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512', # SS on pascal has only 2 models. So removed from plots.
    'vseg-19-401-0_tflitert': 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512', # # SS on pascal has only 2 models.  So removed from plots.
}

#sampled on 18th Mar
super_set = [    
    'vcls-10-010-0_tflitert',
    'vcls-10-011-0_tflitert',
    'vcls-10-012-0_tflitert',
    'vcls-10-020-0_tvmdlr',
    'vcls-10-030-0_tvmdlr',
    'vcls-10-031-0_tvmdlr',
    'vcls-10-032-0_tvmdlr',
    'vcls-10-033-0_tvmdlr',
    'vcls-10-060-0_tvmdlr',
    'vcls-10-061-0_tvmdlr',
    'vcls-10-062-0_tvmdlr',
    'vcls-10-100-0_tvmdlr',
    'vcls-10-101-0_tvmdlr',
    'vcls-10-101-8_tvmdlr',
    'vcls-10-102-8_tvmdlr',
    'vcls-10-301-0_tvmdlr',
    'vcls-10-302-0_tvmdlr',
    'vcls-10-302-8_tvmdlr',
    'vcls-10-304-0_tvmdlr',
    'vcls-10-305-0_tvmdlr',
    'vcls-10-400-0_tflitert',
    'vcls-10-400-8_tflitert',
    'vcls-10-401-0_tflitert',
    'vcls-10-401-8_tflitert',
    'vcls-10-402-0_tflitert',
    'vcls-10-403-0_tflitert',
    'vcls-10-404-0_tflitert',
    'vcls-10-405-8_tflitert',
    'vcls-10-406-0_tflitert',
    'vcls-10-407-0_tflitert',
    'vcls-10-408-0_tflitert',
    'vcls-10-409-0_tflitert',
    'vcls-10-410-0_tflitert',
    'vcls-10-430-0_tflitert',
    'vcls-10-431-0_tflitert',
    'vcls-10-432-0_tflitert',
    'vcls-10-434-0_tflitert',
    'vcls-10-440-0_tflitert',
    'vcls-10-441-0_tflitert',
    'vcls-10-442-0_tflitert',
    'vdet-12-010-0_tflitert',
    'vdet-12-011-0_tflitert',
    'vdet-12-060-0_tvmdlr',
    'vdet-12-061-0_tvmdlr',
    'vdet-12-400-0_tflitert',
    'vdet-12-401-0_tflitert',
    'vdet-12-402-0_tflitert',
    'vseg-16-100-0_tvmdlr',
    'vseg-16-101-0_tvmdlr',
    'vseg-16-102-0_tvmdlr',
    'vseg-16-103-0_tvmdlr',
    'vseg-16-104-0_tvmdlr',
    'vseg-16-105-0_tvmdlr',
    'vseg-16-400-0_tflitert',
    'vseg-17-400-0_tflitert',
    'vseg-18-010-0_tflitert',
    'vseg-18-100-0_tvmdlr',
    'vseg-18-100-8_tvmdlr',
    'vseg-18-101-0_tvmdlr',
    'vseg-18-101-8_tvmdlr',
    'vseg-18-102-0_tvmdlr',
    'vseg-18-102-8_tvmdlr',
    'vseg-18-103-0_tvmdlr',
    'vseg-18-103-8_tvmdlr',
    'vseg-18-110-0_tvmdlr',
    'vseg-18-111-0_tvmdlr',
    'vseg-19-400-0_tflitert',
    'vseg-19-401-0_tflitert',
]    

def test_against_super_set():
    for artifacts_id in super_set:
        if not artifacts_id in model_id_artifacts_pair:
            print("{} is part of super-set but not in model names".format(artifacts_id))

def excel_to_dict(excel_file=None, numeric_cols=None):
    try:
        import pandas as pd
    except:
        raise ImportError("excel_to_dict is not supported, check if 'import pandas' work")

    if os.path.splitext(excel_file)[1] == '.xlsx':
        df = pd.read_excel( excel_file, engine='openpyxl')
    elif os.path.splitext(excel_file)[1] == '.csv':
        df = pd.read_csv(excel_file, skipinitialspace=True)
    elif os.path.splitext(excel_file)[1] == '.xls':
        df = pd.read_excel(excel_file)
    else:
        exit(0)

    for pick_key in numeric_cols:
        df[pick_key] = pd.to_numeric(df[pick_key], errors='coerce', downcast='signed').fillna(0.0).astype(float)

    #models_info_list = df.to_dict('list')
    models_info_index = df.to_dict('index')

    #change key form serial number to model_id
    models_info_dict = dict()
    for k,v in models_info_index.items():
        #report changed column name from run_time to runtime_name
        run_time = v['run_time'] if 'run_time' in v else v['runtime_name']
        models_info_dict[v['model_id']+'_'+run_time] = v

    return models_info_dict

def get_missing_models(report_file=None, selected_models_list=None):
    numeric_cols = ['serial_num',	'metric_8bits',	'metric_16bits',	'metric_float',	'metric_reference',	'num_subgraphs',	'infer_time_core_ms',	'ddr_transfer_mb', 'perfsim_ddr_transfer_mb', 'perfsim_gmacs']

    models_info_dict = excel_to_dict(excel_file=report_file, numeric_cols=numeric_cols) 
    
    missing_models_list = [(model, model_id_artifacts_pair[model])for model in selected_models_list if not model in models_info_dict]
    if len(missing_models_list) > 0:
        print("#"*32)            
        print("perf-data missing for the models")
        for artifact_id, artifacts_name in sorted(missing_models_list):
            model_id  = artifact_id.split('_')[0]
            #print(model_id,', #', artifact_id, artifacts_name)
            print("'{}', # {:23}, {}".format(model_id, artifact_id, artifacts_name))
        print("#"*32)               

    return
    
def get_selected_models(selected_task=None):
    selected_models_list = [key for key in model_id_artifacts_pair if not key in removed_model_list]                
    selected_models_for_a_task = [model for model in selected_models_list if model.split('-')[0] == selected_task]                
    return selected_models_for_a_task


def get_artifact_name(model_id_or_artifact_id, session_name=None, guess_names=False):
    # artifact_id is model_id followed by session_name
    # either pass a model_id and a session_name
    # or directly pass the artifact_id and don't pass session_name
    if session_name is None:
        artifact_id = model_id_or_artifact_id
    else:
        model_id = model_id_or_artifact_id
        artifact_id = f'{model_id}_{session_name}'
    #

    artifact_name = None
    if artifact_id in model_id_artifacts_pair:
        artifact_name = model_id_artifacts_pair[artifact_id]
    elif guess_names:
        model_id, runtime_name = artifact_id.split('_')
        # create mapping dictionaries
        model_id_to_model_name_dict = {k.split('_')[0]:'-'.join(v.split('-')[1:]) \
                for k,v in model_id_artifacts_pair.items()}
        short_runtime_name_dict = {'tvmdlr':'TVM', 'tflitert':'TFL', 'onnxrt':'ONR'}
        # finally for the artifact name
        if runtime_name in short_runtime_name_dict and model_id in model_id_to_model_name_dict:
            artifact_name = f'{short_runtime_name_dict[runtime_name]}-{model_id_to_model_name_dict[model_id]}'
        
    return artifact_name


def get_name_key_pair_list(model_ids, session_name, remove_models=True):
    global removed_model_list
    removed_model_entries = removed_model_list.keys()
    name_key_pair_list = []
    for model_id in model_ids:
        artifact_id = f'{model_id}_{session_name}'
        artifact_name =  model_id_artifacts_pair[artifact_id] if artifact_id in model_id_artifacts_pair else None
        if artifact_name is not None and \
                (not remove_models or artifact_id not in removed_model_entries):
            name_key_pair_list.append((artifact_name, model_id))
        #
    #
    return name_key_pair_list


if __name__ == '__main__':
    generate_list_mising_models = True

    test_against_super_set()

    print("Total models : ", len(model_id_artifacts_pair))        
    print("removed models : ", len(removed_model_list))        
    print_selected_models = True

    selected_models_list = [key for key in model_id_artifacts_pair if not key in removed_model_list]
    
    print("with runtime prefix")
    print("="*64)
    if print_selected_models:
        for selected_model in sorted(selected_models_list):
            print("{}{}{}".format("\'", selected_model,"\'" ), end=',')
        print("")    

    print("without runtime prefix")
    print("="*64)
    if print_selected_models:
        for selected_model in sorted(selected_models_list):
            selected_model = '-'.join(selected_model.split('-')[0:-1])
            print("{}{}{}".format("\'", selected_model,"\'" ), end=',')
        print("")    
    print("="*64)
    selected_models_vcls = [model for model in selected_models_list if model.split('-')[0] == 'vcls']
    selected_models_vdet = [model for model in selected_models_list if model.split('-')[0] == 'vdet']
    selected_models_vseg = [model for model in selected_models_list if model.split('-')[0] == 'vseg']

    print("num_selected_models: {}, vcls:{}, vdet:{}, vseg:{}".format(len(selected_models_list), len(selected_models_vcls), len(selected_models_vdet), len(selected_models_vseg)))        

    #find which models need re-run due to lack of performance data    
    if generate_list_mising_models:    
        df = get_missing_models(report_file='./work_dirs/modelartifacts/report_20210409-155749.csv',
            selected_models_list=selected_models_list)