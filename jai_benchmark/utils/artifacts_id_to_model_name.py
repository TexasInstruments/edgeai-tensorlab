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

#ver:34 2022-04-04

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
# ONR	   object pose estimation	7200
# ONR	   depth estimation	        7300
# ONR	   OD                       8000
# ONR	   Semantic Segmentation	8500


#mapping from artifacts id to readable model names
model_id_artifacts_pair = {
    # TFLite CL
    'cl-0000_tflitert': 'TFL-CL-0000-mobileNetV1-mlperf',
    'cl-0010_tflitert': 'TFL-CL-0010-mobileNetV2',
    'cl-0020_tflitert': 'TFL-CL-0020-SqueezeNet',
    'cl-0038_tflitert': 'TFL-CL-0038-InceptionNetV1',
    'cl-0040_tflitert': 'TFL-CL-0040-InceptionNetV3',
    'cl-0050_tflitert': 'TFL-CL-0050-resNet50V1',
    'cl-0060_tflitert': 'TFL-CL-0060-resNet50V2',
    'cl-0070_tflitert': 'TFL-CL-0070-mnasNet',
    'cl-0080_tflitert': 'TFL-CL-0080-mobileNet-edgeTPU-mlperf',
    'cl-0090_tflitert': 'TFL-CL-0090-efficientNet-edgeTPU-s',
    'cl-0100_tflitert': 'TFL-CL-0100-efficientNet-edgeTPU-m',
    'cl-0130_tflitert': 'TFL-CL-0130-efficientNet-lite0',
    'cl-0140_tflitert': 'TFL-CL-0140-efficientNet-lite4',
    'cl-0150_tflitert': 'TFL-CL-0150-denseNet',
    'cl-0160_tflitert': 'TFL-CL-0160-resNet50V1p5-mlperf',
    'cl-0170_tflitert': 'TFL-CL-0170-efficientNet-lite1',
    'cl-0180_tflitert': 'TFL-CL-0180-efficientNet-lite2',
    'cl-0190_tflitert': 'TFL-CL-0190-efficientNet-edgeTPU-l',
    'cl-0200_tflitert': 'TFL-CL-0200-mobileNetV2-1p4',
    'cl-0210_tflitert': 'TFL-CL-0210-mobileNetV1',
    'cl-0218_tflitert': 'TFL-CL-0218-mobileNetV1-qat',
    'cl-0018_tflitert': 'TFL-CL-0018-mobileNetV2-qat',
    'cl-0240_tflitert': 'TFL-CL-0240-nasNet-mobile-tflite',
    'cl-0250_tflitert': 'TFL-CL-0250-xceptionNet-tflite', # mxnet model replaced with with tflite model now
    'cl-0260_tflitert': 'TFL-CL-0260-mobv3-large-minimalistic', #added in SDK8.0
    'cl-0270_tflitert': 'TFL-CL-0270-mobv3-small-minimalistic', #added in SDK8.0

    # TFLite OD
    'od-2000_tflitert': 'TFL-OD-2000-ssd-mobV1-coco-mlperf-300x300',
    'od-2010_tflitert': 'TFL-OD-2010-ssd-mobV2-coco-mlperf-300x300',
    'od-2020_tflitert': 'TFL-OD-2020-ssdLite-mobDet-DSP-coco-320x320',
    'od-2030_tflitert': 'TFL-OD-2030-ssdLite-mobDet-EdgeTPU-coco-320x320',
    'od-2040_tflitert': 'TFL-OD-2040-ssd-mobV1-fpn-coco-640x640',
    'od-2050_tflitert': 'TFL-OD-2050-ssd-mobV2-mnas-fpn-coco-320x320',
    'od-2060_tflitert': 'TFL-OD-2060-ssdlite-mobV2-coco-300x300', #name was wrong. Corrrected on 2021-08-09. During SK8.0 release

    # added in SDK8.0
    'od-2070_tflitert':'TFL-OD-2070-ssd-mobV1-fpn-coco-tpu-8-640x640',
    'od-2080_tflitert':'TFL-OD-2080-ssd-mobV2-fpnlite-coco-tpu-8-320x320',
    'od-2090_tflitert':'TFL-OD-2090-ssd-mobV2-fpnlite-coco-tpu-8-640x640',
    'od-2100_tflitert':'TFL-OD-2100-ssd-res50V1-fpn-coco-tpu-8-640x640',
    'od-2110_tflitert':'TFL-OD-2110-efficient-det-ti-lite0-coco-512x512',
    'od-2120_tflitert':'TFL-OD-2120-ssd-res50V1-fpn-coco-tpu-8-1024x1024', #'od-12-415-0'
    'od-2130_tflitert':'TFL-OD-2130-ssd-mobV2-coco-tpu-8-320x320', #'od-12-411-0'

    # TFLite SS
    'ss-2500_tflitert': 'TFL-SS-2500-deeplab-mobV2-ade20k-512x512',
    'ss-2540_tflitert': 'TFL-SS-2540-deeplabv3-mobv2-ade20k-512x512',
    'ss-2550_tflitert': 'TFL-SS-2550-deeplabv3-mobv2-cs-2048x1024',
    'ss-2580_tflitert': 'TFL-SS-2580-deeplabv3_mobv2-ade20k32-mlperf-512x512',
    'ss-2590_tflitert': 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512',
    'ss-2600_tflitert': 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512',

    # TVM- CL
    'cl-3000_tvmdlr': 'TVM-CL-3000-resNet18V2',
    'cl-3020_tvmdlr': 'TVM-CL-3020-xceptionNet-mxnet',
    'cl-3040_tvmdlr': 'TVM-CL-3040-nasNet-mobile-tflite',
    'cl-3060_tvmdlr': 'TVM-CL-3060-mobileNetV1',
    'cl-3070_tvmdlr': 'TVM-CL-3070-mobileNetV2',
    'cl-3080_tvmdlr': 'TVM-CL-3080-shuffleNetV2',
    'cl-3090_tvmdlr': 'TVM-CL-3090-mobileNetV2-tv',
    'cl-3100_tvmdlr': 'TVM-CL-3100-resNet18',
    'cl-3110_tvmdlr': 'TVM-CL-3110-resNet50',
    'cl-3120_tvmdlr': 'TVM-CL-3120-regNetX-400mf',
    'cl-3130_tvmdlr': 'TVM-CL-3130-regNetX-800mf',
    'cl-3140_tvmdlr': 'TVM-CL-3140-regNetX-1.6gf',
    'cl-3150_tvmdlr': 'TVM-CL-3150-mobileNetV2-1p4-qat',

    #512x512
    'cl-3061_tvmdlr': 'TVM-CL-3061-mobileNetV1-512x512',
    'cl-3071_tvmdlr': 'TVM-CL-3071-mobileNetV2-512x512',
    'cl-3081_tvmdlr': 'TVM-CL-3081-shuffleNetV2-512x512',
    'cl-3091_tvmdlr': 'TVM-CL-3091-mobileNetV2-tv-512x512',
    'cl-3101_tvmdlr': 'TVM-CL-3101-resNet18-512x512',
    'cl-3111_tvmdlr': 'TVM-CL-3111-resNet50-512x512',
    'cl-3121_tvmdlr': 'TVM-CL-3121-regNetX-400mf-512x512',
    'cl-3131_tvmdlr': 'TVM-CL-3131-regNetX-800mf-512x512',
    'cl-3141_tvmdlr': 'TVM-CL-3141-regNetX-1.6gf-512x512',
    'cl-3151_tvmdlr': 'TVM-CL-3151-mobileNetV2-1p4-qat-512x512',

    #1024x1024
    'cl-3062_tvmdlr': 'TVM-CL-3062-mobileNetV1-1024x1024',
    'cl-3072_tvmdlr': 'TVM-CL-3072-mobileNetV2-1024x1024',
    'cl-3082_tvmdlr': 'TVM-CL-3082-shuffleNetV2-1024x1024',
    'cl-3092_tvmdlr': 'TVM-CL-3092-mobileNetV2-tv-1024x1024',
    'cl-3102_tvmdlr': 'TVM-CL-3102-resNet18-1024x1024',
    'cl-3112_tvmdlr': 'TVM-CL-3112-resNet50-1024x1024',
    'cl-3122_tvmdlr': 'TVM-CL-3122-regNetX-400mf-1024x1024',
    'cl-3132_tvmdlr': 'TVM-CL-3132-regNetX-800mf-1024x1024',
    'cl-3142_tvmdlr': 'TVM-CL-3142-regNetX-1.6gf-1024x1024',
    'cl-3152_tvmdlr': 'TVM-CL-3152-mobileNetV2-1p4-qat-1024x1024',
    ########

    'cl-3360_tvmdlr': 'TVM-CL-3360-regNetx-200mf',
    'cl-3370_tvmdlr': 'TVM-CL-3370-vgg16',
    'cl-3078_tvmdlr': 'TVM-CL-3078-mobileNetV2-qat',
    'cl-3098_tvmdlr': 'TVM-CL-3098-mobileNetV2-tv-qat',
    'cl-3410_tvmdlr': 'TVM-CL-3410-gluoncv-mxnet-mobv2',
    'cl-3420_tvmdlr': 'TVM-CL-3420-gluoncv-mxnet-resNet50-v1',
    'cl-3430_tvmdlr': 'TVM-CL-3430-gluoncv-mxnet-xception',
    # HarD Net
    'cl-3440_tvmdlr': 'TVM-CL-3440-harDNet68',
    'cl-3450_tvmdlr': 'TVM-CL-3450-harDNet85',
    'cl-3460_tvmdlr': 'TVM-CL-3460-harDNet68ds',
    'cl-3470_tvmdlr': 'TVM-CL-3470-harDNet39ds',

    'cl-3480_tvmdlr': 'TVM-CL-3480-hrnet-w18-small-v2-gluon-mxnet',#added in SDK8.0
    'cl-3498_tvmdlr': 'TVM-CL-3498-mobv3-ti-lite-large-qat', #added in SDK8.0 #'cl-10-106-8'
    'cl-3500_tvmdlr': 'TVM-CL-3500-mobv3-ti-lite-large-x2r', #added in SDK8.0 #'cl-10-107-0'
    'cl-3510_tvmdlr': 'TVM-CL-3510-hrnet-w30-c-gluon-mxnet', #added in SDK8.0 #'cl-10-064-0'

    # TVM - OD
    'od-5000_tvmdlr': 'TVM-OD-5000-ssd1200-resNet34-mlperf-1200x1200',
    'od-5010_tvmdlr': 'TVM-OD-5010-yolov3-416x416',
    'od-5020_tvmdlr': 'TVM-OD-5020-yolov3-mobv1-gluon-mxnet-coco-416x416',

    'od-5030_tvmdlr': 'TVM-OD-5030-ssd-res50v1-gluon-mxnet-512x512', #added in SDK8.0
    'od-5040_tvmdlr': 'TVM-OD-5040-ssd-mobv1-coco-gluon-mxnet-512x512', #renamed mobv1 model in SDK8.0

    'od-5050_tvmdlr': 'TVM-OD-5050-yolo3_darknet53-coco-gluon-mxnet-512x512', #renamed mobv1 model in SDK8.0 #'od-12-063-0'
    'od-5060_tvmdlr': 'TVM-OD-5060-centernet-r18-v1b-coco-gluon-mxnet-512x512', #renamed mobv1 model in SDK8.0 'od-12-064-0'

    # TVM - SS - CS
    'ss-5500_tvmdlr': 'TVM-SS-5500-deeplabv3lite-mobv2-cs-768x384',
    'ss-5510_tvmdlr': 'TVM-SS-5510-deeplabv3lite-mobv2-cs-qat-768x384',
    'ss-5520_tvmdlr': 'TVM-SS-5520-fpnlite-aspp-mobv2-cs-768x384',
    'ss-5528_tvmdlr': 'TVM-SS-5528-fpnlite-aspp-mobv2-cs-qat-768x384',
    'ss-5540_tvmdlr': 'TVM-SS-5540-unetlite-aspp-mobv2-tv-cs-768x384',
    'ss-5548_tvmdlr': 'TVM-SS-5548-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'ss-5560_tvmdlr': 'TVM-SS-5560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'ss-5570_tvmdlr': 'TVM-SS-5570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'ss-5580_tvmdlr': 'TVM-SS-5580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'ss-5590_tvmdlr': 'TVM-SS-5590-deeplabv3-res50-cs-1040x520',
    'ss-5600_tvmdlr': 'TVM-SS-5600-fcn-res50-cs-1040x520',

    # TVM - SS - ADE20k
    'ss-5610_tvmdlr': 'TVM-SS-5610-deeplabv3lite-mobv2-ade20k32-512x512',
    'ss-5618_tvmdlr': 'TVM-SS-5618-deeplabv3lite-mobv2-ade20k32-qat-512x512',
    'ss-5630_tvmdlr': 'TVM-SS-5630-unetlite-aspp-mobv2-tv-ade20k32-512x512',
    'ss-5638_tvmdlr': 'TVM-SS-5638-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512',
    'ss-5650_tvmdlr': 'TVM-SS-5650-fpnlite-aspp-mobv2-ade20k32-512x512',
    'ss-5658_tvmdlr': 'TVM-SS-5658-fpnlite-aspp-mobv2-ade20k32-qat-512x512',
    'ss-5670_tvmdlr': 'TVM-SS-5670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512',
    'ss-5678_tvmdlr': 'TVM-SS-5678-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512',
    'ss-5690_tvmdlr': 'TVM-SS-5690-fpnlite-aspp-regnetx400mf-ade20k32-384x384',
    'ss-5700_tvmdlr': 'TVM-SS-5700-fpnlite-aspp-regnetx800mf-ade20k32-512x512',

    # TVM -SS - CocoSeg21
    'ss-5710_tvmdlr' : 'TVM-SS-5710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'ss-5720_tvmdlr' : 'TVM-SS-5720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',

    # TVM -SS - CocoSeg21
    'ss-5810_tvmdlr' : 'TVM-SS-5810-fcn-resNet101-cocoseg21-gluon-mxnet-480x480',
    'ss-5820_tvmdlr' : 'TVM-SS-5820-deeplab-resNet101-cocoseg21-gluon-mxnet-480x480',
    'ss-5830_tvmdlr' : 'TVM-SS-5830-fcn-resNet50-ade20k-gluon-mxnet-480x480',

    # ONNXRT- CL
    'cl-6000_onnxrt': 'ONR-CL-6000-resNet18V2',
    'cl-6040_onnxrt': 'ONR-CL-6040-nasNet-mobile-tflite',
    'cl-6060_onnxrt': 'ONR-CL-6060-mobileNetV1',
    'cl-6070_onnxrt': 'ONR-CL-6070-mobileNetV2',
    'cl-6080_onnxrt': 'ONR-CL-6080-shuffleNetV2',
    'cl-6090_onnxrt': 'ONR-CL-6090-mobileNetV2-tv',
    'cl-6100_onnxrt': 'ONR-CL-6100-resNet18',
    'cl-6110_onnxrt': 'ONR-CL-6110-resNet50',
    'cl-6120_onnxrt': 'ONR-CL-6120-regNetX-400mf',
    'cl-6130_onnxrt': 'ONR-CL-6130-regNetX-800mf',
    'cl-6140_onnxrt': 'ONR-CL-6140-regNetX-1.6gf',
    'cl-6150_onnxrt': 'ONR-CL-6150-mobileNetV2-1p4-qat',
    'cl-6160_onnxrt': 'ONR-CL-6160-regNetX-400mf',
    'cl-6170_onnxrt': 'ONR-CL-6170-regNetX-800mf',
    'cl-6180_onnxrt': 'ONR-CL-6180-regNetX-1.6gf',

    #512x512
    'cl-6061_onnxrt': 'ONR-CL-6061-mobileNetV1-512x512',
    'cl-6071_onnxrt': 'ONR-CL-6071-mobileNetV2-512x512',
    'cl-6081_onnxrt': 'ONR-CL-6081-shuffleNetV2-512x512',
    'cl-6091_onnxrt': 'ONR-CL-6091-mobileNetV2-tv-512x512',
    'cl-6101_onnxrt': 'ONR-CL-6101-resNet18-512x512',
    'cl-6111_onnxrt': 'ONR-CL-6111-resNet50-512x512',
    'cl-6121_onnxrt': 'ONR-CL-6121-regNetX-400mf-512x512',
    'cl-6131_onnxrt': 'ONR-CL-6131-regNetX-800mf-512x512',
    'cl-6141_onnxrt': 'ONR-CL-6141-regNetX-1.6gf-512x512',
    'cl-6151_onnxrt': 'ONR-CL-6151-mobileNetV2-1p4-qat-512x512',

    #1024x1024
    'cl-6062_onnxrt': 'ONR-CL-6062-mobileNetV1-1024x1024',
    'cl-6072_onnxrt': 'ONR-CL-6072-mobileNetV2-1024x1024',
    'cl-6082_onnxrt': 'ONR-CL-6082-shuffleNetV2-1024x1024',
    'cl-6092_onnxrt': 'ONR-CL-6092-mobileNetV2-tv-1024x1024',
    'cl-6102_onnxrt': 'ONR-CL-6102-resNet18-1024x1024',
    'cl-6112_onnxrt': 'ONR-CL-6112-resNet50-1024x1024',
    'cl-6122_onnxrt': 'ONR-CL-6122-regNetX-400mf-1024x1024',
    'cl-6132_onnxrt': 'ONR-CL-6132-regNetX-800mf-1024x1024',
    'cl-6142_onnxrt': 'ONR-CL-6142-regNetX-1.6gf-1024x1024',
    'cl-6152_onnxrt': 'ONR-CL-6152-mobileNetV2-1p4-qat-1024x1024',
    ########

    'cl-6360_onnxrt': 'ONR-CL-6360-regNetx-200mf',
    'cl-6370_onnxrt': 'ONR-CL-6370-vgg16',
    'cl-6078_onnxrt': 'ONR-CL-6078-mobileNetV2-qat',
    'cl-6098_onnxrt': 'ONR-CL-6098-mobileNetV2-tv-qat',
    'cl-6410_onnxrt': 'ONR-CL-6410-gluoncv-mxnet-mobv2',
    'cl-6420_onnxrt': 'ONR-CL-6420-gluoncv-mxnet-resNet50-v1',
    'cl-6430_onnxrt': 'ONR-CL-6430-gluoncv-mxnet-xception',

    # HarD Net
    # model id changed in SDK8.0
    'cl-6440_onnxrt': 'ONR-CL-6440-harDNet68',
    'cl-6450_onnxrt': 'ONR-CL-6450-harDNet85',
    'cl-6460_onnxrt': 'ONR-CL-6460-harDNet68ds',
    'cl-6470_onnxrt': 'ONR-CL-6470-harDNet39ds',

    #added in SDK8.0
    'cl-6480_onnxrt': 'ONR-CL-6480-mobv3-lite-small',
    'cl-6488_onnxrt': 'ONR-CL-6488-mobv3-lite-small-qat',
    'cl-6490_onnxrt': 'ONR-CL-6490-mobv3-lite-large',

    'cl-6508_onnxrt': 'ONR-CL-6508-mobv3-ti-lite-large-qat', #added in SDK8.0 #'cl-10-106-8'
    'cl-6510_onnxrt': 'ONR-CL-6510-mobv3-ti-lite-large-x2r', #added in SDK8.0 #'cl-10-107-0'

    # ONNX - OD
    'od-8000_onnxrt': 'ONR-OD-8000-ssd1200-resNet34-mlperf-coco-1200x1200',
    'od-8010_onnxrt': 'ONR-OD-8010-yolov3-coco-416x416',

    # Edgeai-MMdetection
    'od-8020_onnxrt': 'ONR-OD-8020-ssd-lite-mobv2-mmdet-coco-512x512',
    'od-8030_onnxrt': 'ONR-OD-8030-ssd-lite-mobv2-fpn-mmdet-coco-512x512',
    'od-8040_onnxrt': 'ONR-OD-8040-ssd-lite-regNetX-200mf-fpn-bgr-mmdet-coco-320x320',
    'od-8050_onnxrt': 'ONR-OD-8050-ssd-lite-regNetX-800mf-fpn-bgr-mmdet-coco-512x512',
    'od-8060_onnxrt': 'ONR-OD-8060-ssd-lite-regNetX-1.6gf-fpn-bgr-mmdet-coco-768x768',
    'od-8070_onnxrt': 'ONR-OD-8070-yolov3-d53-relu-mmdet-coco-416x416',
    'od-8080_onnxrt': 'ONR-OD-8080-yolov3-lite-regNetX-1.6gf-bgr-mmdet-coco-512x512',
    'od-8090_onnxrt': 'ONR-OD-8090-retina-lite-regNetX-800mf-fpn-bgr-mmdet-coco-512x512',
    'od-8200_onnxrt': 'ONR-OD-8200-yolox-nano-lite-mmdet-coco-416x416',
    'od-8210_onnxrt': 'ONR-OD-8210-yolox-tiny-lite-mmdet-coco-416x416',
    'od-8220_onnxrt': 'ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640',
    'od-8230_onnxrt': 'ONR-OD-8230-yolox-m-lite-mmdet-coco-640x640',
    # face detection
    'od-8410_onnxrt': 'ONR-OD-8410-yolox-tiny-lite-mmdet-widerface-416x416',
    'od-8420_onnxrt': 'ONR-OD-8420-yolox-s-lite-mmdet-widerface-640x640',
    'od-8421_onnxrt': 'ONR-OD-8421-yolox-s-lite-mmdet-widerface-1024x1024',
    'od-8430_onnxrt': 'ONR-OD-8430-yolox-m-lite-mmdet-widerface-640x640',

    # Edgeai-Yolov5
    'od-8100_onnxrt': 'ONR-OD-8100-yolov5-s6-ti-lite-coco-640x640',
    'od-8110_onnxrt': 'ONR-OD-8110-yolov5-s6-ti-lite-coco-384x384',
    'od-8120_onnxrt': 'ONR-OD-8120-yolov5-m6-ti-lite-coco-640x640',
    'od-8130_onnxrt': 'ONR-OD-8130-yolov5-l6-ti-lite-coco-640x640',
    'od-8450_onnxrt': 'ONR-OD-8450-yolov5-s6-ti-lite-widerface-640x640',

    # Edgeai-Yolox
    'od-8140_onnxrt': 'ONR-OD-8140-yolox-s-ti-lite-coco-640x640',
    'od-8150_onnxrt': 'ONR-OD-8150-yolox-m-ti-lite-coco-640x640',
    'od-8180_onnxrt': 'ONR-OD-8180-yolox-tiny-ti-lite-coco-416x416',
    'od-8190_onnxrt': 'ONR-OD-8190-yolox-nano-ti-lite-coco-416x416',

    # EdgeAI-Torchvision
    'od-8160_onnxrt': 'ONR-OD-8160-ssdlite-mobv2-fpn-lite-coco-512x512',
    'od-8170_onnxrt': 'ONR-OD-8170-ssdlite-regNetX-800mf-fpn-lite-coco-512x512',

    # ONNX - SS - CS
    'ss-8500_onnxrt': 'ONR-SS-8500-deeplabv3lite-mobv2-cs-768x384',
    'ss-8508_onnxrt': 'ONR-SS-8508-deeplabv3lite-mobv2-cs-qat-768x384',
    'ss-8520_onnxrt': 'ONR-SS-8520-fpnlite-aspp-mobv2-cs-768x384',
    'ss-8528_onnxrt': 'ONR-SS-8528-fpnlite-aspp-mobv2-cs-qat-768x384',
    'ss-8540_onnxrt': 'ONR-SS-8540-unetlite-aspp-mobv2-tv-cs-768x384',
    'ss-8548_onnxrt': 'ONR-SS-8548-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'ss-8560_onnxrt': 'ONR-SS-8560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'ss-8570_onnxrt': 'ONR-SS-8570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'ss-8580_onnxrt': 'ONR-SS-8580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'ss-8590_onnxrt': 'ONR-SS-8590-deeplabv3-res50-1040x520',
    'ss-8600_onnxrt': 'ONR-SS-8600-fcn-res50-1040x520',
    # ONNX - LIDAR - 3DOD
    '3dod-7100_onnxrt' : 'ONR-3DOD-7100-pointPillars-lidar-10000-496x432',

    # ONNX - SS - ADE20k
    'ss-8610_onnxrt': 'ONR-SS-8610-deeplabv3lite-mobv2-ade20k32-512x512',
    'ss-8618_onnxrt': 'ONR-SS-8618-deeplabv3lite-mobv2-ade20k32-qat-512x512',
    'ss-8630_onnxrt': 'ONR-SS-8630-unetlite-aspp-mobv2-tv-ade20k32-512x512',
    'ss-8638_onnxrt': 'ONR-SS-8638-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512',
    'ss-8650_onnxrt': 'ONR-SS-8650-fpnlite-aspp-mobv2-ade20k32-512x512',
    'ss-8658_onnxrt': 'ONR-SS-8658-fpnlite-aspp-mobv2-ade20k32-qat-512x512',
    'ss-8670_onnxrt': 'ONR-SS-8670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512',
    'ss-8678_onnxrt': 'ONR-SS-8678-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512',
    'ss-8690_onnxrt': 'ONR-SS-8690-fpnlite-aspp-regnetx400mf-ade20k32-384x384',
    'ss-8700_onnxrt': 'ONR-SS-8700-fpnlite-aspp-regnetx800mf-ade20k32-512x512',

    # ONNX-SS - CocoSeg21
    'ss-8710_onnxrt': 'ONR-SS-8710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'ss-8720_onnxrt': 'ONR-SS-8720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',
    'ss-8730_onnxrt': 'ONR-SS-8730-deeplabv3-mobv3-lite-large-cocoseg21-512x512', #added in SDK8.0
    'ss-8731_onnxrt': 'ONR-SS-8731-lraspp-mobV3-ti-lite-large-cocoseg21-512x512', #added in SDK8.0 #'ss-21-105-0'

    # Models for Robokit
    'ss-8810_onnxrt': 'ONR-SS-8810-deeplabv3lite-mobv2-robokit-768x432',
    'ss-8818_onnxrt': 'ONR-SS-8818-deeplabv3lite-mobv2-qat-robokit-768x432',
    'ss-5810_tvmdlr': 'TVM-SS-5810-deeplabv3lite-mobv2-robokit-768x432',
    'ss-5818_tvmdlr': 'TVM-SS-5818-deeplabv3lite-mobv2-qat-robokit-768x432',

    #ONNX key point detections
    'kd-7000_onnxrt': 'ONR-KD-7000-human-pose-ae-mobv2-fpn-spp-udp-512x512',
    'kd-7010_onnxrt': 'ONR-KD-7010-human-pose-ae-res50v2-fpn-spp-udp-512x512',
    'kd-7020_onnxrt': 'ONR-KD-7020-human-pose-ae-mobv2-pan-spp-udp-512x512',
    'kd-7030_onnxrt': 'ONR-KD-7030-human-pose-ae-res50v2-pan-spp-udp-512x512',
    'kd-7040_onnxrt': 'ONR-KD-7040-human-pose-yolov5s6-640x640',
    'kd-7050_onnxrt': 'ONR-KD-7050-human-pose-yolov5s6-mixp-640x640',
    'kd-7060_onnxrt': 'ONR-KD-7060-human-pose-yolox-s-640x640',

    'od-2150_tflitert':'TFL-OD-2150-efficientdet-lite1-relu-coco-384x384',
    'od-2170_tflitert':'TFL-OD-2170-efficientdet-lite3-relu-coco-512x512',

    # ONNX depth estimation
    'de-7300_onnxrt': 'ONR-DE-7300-depth-est-fast-depth-224x224',
    'de-7310_onnxrt': 'ONR-DE-7310-depth-est-midas-small-256x256',
}

removed_model_list = {
    'cl-3020_tvmdlr' : 'TVM-CL-3020-xceptionNet-mxnet', # This is replaced with tflite model now (that was also eventually removed)
    'cl-0230_tflitert': 'TFL-CL-0230-mobileNetV2-qat',  # QAT model is not giving good accuracy so keep only float
    'od-5000_tvmdlr': 'TVM-OD-5000-ssd1200-resNet34-mlperf-coco-1200x1200', # Using ONNX RT for this.
    'cl-0210_tflitert': 'TFL-CL-0210-mobileNetV1', #duplicate same as cl-0000 mlperf model.
    ################ CS models
    # cityscapes model not part of Model Zoo
    'ss-5500_tvmdlr': 'TVM-SS-5500-deeplabv3lite-mobv2-cs-768x384',
    'ss-5510_tvmdlr': 'TVM-SS-5510-deeplabv3lite-mobv2-cs-qat-768x384',
    'ss-5520_tvmdlr': 'TVM-SS-5520-fpnlite-aspp-mobv2-cs-768x384',
    'ss-5528_tvmdlr': 'TVM-SS-5528-fpnlite-aspp-mobv2-cs-qat-768x384',
    'ss-5540_tvmdlr': 'TVM-SS-5540-unetlite-aspp-mobv2-tv-cs-768x384',
    'ss-5548_tvmdlr': 'TVM-SS-5548-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'ss-5560_tvmdlr': 'TVM-SS-5560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'ss-5570_tvmdlr': 'TVM-SS-5570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'ss-5580_tvmdlr': 'TVM-SS-5580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'ss-5590_tvmdlr': 'TVM-SS-5590-deeplabv3-res50-cs-1040x520',
    'ss-5600_tvmdlr': 'TVM-SS-5600-fcn-res50-cs-1040x520',
    # cityscapes model not part of Model Zoo
    'ss-8500_onnxrt': 'ONR-SS-8500-deeplabv3lite-mobv2-cs-768x384',
    'ss-8508_onnxrt': 'ONR-SS-8508-deeplabv3lite-mobv2-cs-qat-768x384',
    'ss-8520_onnxrt': 'ONR-SS-8520-fpnlite-aspp-mobv2-cs-768x384',
    'ss-8528_onnxrt': 'ONR-SS-8528-fpnlite-aspp-mobv2-cs-qat-768x384',
    'ss-8540_onnxrt': 'ONR-SS-8540-unetlite-aspp-mobv2-tv-cs-768x384',
    'ss-8548_onnxrt': 'ONR-SS-8548-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'ss-8560_onnxrt': 'ONR-SS-8560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'ss-8570_onnxrt': 'ONR-SS-8570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'ss-8580_onnxrt': 'ONR-SS-8580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'ss-8590_onnxrt': 'ONR-SS-8590-deeplabv3-res50-cs-1040x520',
    'ss-8600_onnxrt': 'ONR-SS-8600-fcn-res50-cs-1040x520',
    ################
    'cl-0240_tflitert': 'TFL-CL-0240-nasNet-mobile-tflite', # Kumar removed model (Multiple sub-graphs) (Accuracy issue)
    'cl-0180_tflitert': 'TFL-CL-0180-efficientNet-lite2', # Kumar removed model  (Accuracy issue)
    #'od-2010_tflitert': 'TFL-OD-2010-ssd-mobV2-coco-300x300-mlperf',  # Kumar removed model (Duplocate). It is not duplicate. 2010 and 2060 are different models.
    'ss-2550_tflitert': 'TFL-SS-2550-deeplabv3-mobv2-cs-2048x1024',  # Kumar removed model, (not in plan)

    'cl-3370_tvmdlr': 'TVM-CL-3370-vgg16', # Kumar removed model
    'cl-3000_tvmdlr': 'TVM-CL-3000-resNet18V2', # Kumar removed model
    'ss-5590_tvmdlr': 'TVM-SS-5590-deeplabv3-res50-1040x520', # Kumar removed model, nc does not have info for this
    'ss-5600_tvmdlr': 'TVM-SS-5600-fcn-res50-1040x520', # Kumar removed model

    'cl-6370_onnxrt': 'ONR-CL-6370-vgg16', # Kumar removed model for TVM
    'cl-6000_onnxrt': 'ONR-CL-6000-resNet18V2', # Kumar removed model  for TVM
    'ss-8590_onnxrt': 'ONR-SS-8590-deeplabv3-res50-1040x520', # Kumar removed model, nc does not have info for this  for TVM
    'ss-8600_onnxrt': 'ONR-SS-8600-fcn-res50-1040x520', # Kumar removed model  for TVM

    #########################
    'ss-2500_tflitert': 'TFL-SS-2500-deeplab-mobV2-ade20k-512x512', # Manu said incorrect model ID removed. ss-17-010 is replaced with ss-18-010
    'cl-0250_tflitert': 'TFL-CL-0250-xceptionNet-tflite', # mxnet model replaced with with tflite model now. Eventually removed as size is quite big.
    'od-2040_tflitert': 'TFL-OD-2040-ssd-mobV1-FPN-coco-640x640', # does not run, import crashes. Manu
    'od-2050_tflitert': 'TFL-OD-2050-ssd-mobV2-mnas-fpn-coco-320x320', # does not run, import crashes. Manu

    'cl-3040_tvmdlr': 'TVM-CL-3040-nasNet-mobile-tflite', # not part of benchmarking script yet. tflite model with TVM.
    'od-5010_tvmdlr': 'TVM-OD-5010-yolov3-coco-416x416', # not supported yet

    'cl-6040_onnxrt': 'ONR-CL-6040-nasNet-mobile-tflite', # not part of benchmarking script yet. tflite model with TVM.
    'od-8010_onnxrt': 'ONR-OD-8010-yolov3-coco-416x416', # not supported yet

    #removed from ONNX-RT
    'cl-6410_onnxrt': 'ONR-CL-6410-gluoncv-mxnet-mobv2',
    'cl-6420_onnxrt': 'ONR-CL-6420-gluoncv-mxnet-resNet50-v1',
    'cl-6430_onnxrt': 'ONR-CL-6430-gluoncv-mxnet-xception',

    #ADE20k32 models
    'ss-5618_tvmdlr': 'TVM-SS-5618-deeplabv3lite-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'ss-5638_tvmdlr': 'TVM-SS-5638-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512', # import fails
    'ss-5658_tvmdlr': 'TVM-SS-5658-fpnlite-aspp-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'ss-5678_tvmdlr': 'TVM-SS-5678-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed

    'ss-8618_onnxrt': 'ONR-SS-8618-deeplabv3lite-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'ss-8638_onnxrt': 'ONR-SS-8638-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512', # import fails
    'ss-8658_onnxrt': 'ONR-SS-8658-fpnlite-aspp-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'ss-8678_onnxrt': 'ONR-SS-8678-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed

    # ONNX-SS-CocoSeg21 Models  are now in after SDK8.0
    #'vseg-8710_onnxrt' : 'ONR-SS-8710-deeplabv3lite-mobv2-cocoseg21-512x512',
    #'vseg-8720_onnxrt' : 'ONR-SS-8720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',

    # TVM-SS-CocoSeg21 Models are now in after SDK8.0
    #'vseg-5710_tvmdlr' : 'TVM-SS-5710-deeplabv3lite-mobv2-cocoseg21-512x512',
    #'vseg-5720_tvmdlr' : 'TVM-SS-5720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',

    # During SDK7.3 removed low accuracy models where drop is more wrt reference accuracy. This will be corrected in the future.
    'ss-2540_tflitert' : 'TFL-SS-2540-deeplabv3-mobv2-ade20k-512x512', #still bad in SDK8.2
    'ss-2590_tflitert' : 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512', #still bad in SDK8.2
    'cl-0060_tflitert' : 'TFL-CL-0060-resNet50V2', #float accuracy is around 5% lower than ref, SDK8.2
    'ss-2600_tflitert' : 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512', #still bad in SDK8.2
    'cl-0218_tflitert' : 'TFL-CL-0218-mobileNetV1-qat', #acruacy 1.6% below float
    'cl-0010_tflitert' : 'TFL-CL-0010-mobileNetV2', #sdk8.2, accuracy around 3% down compared to float
    #'cl-6080_onnxrt' : 'ONR-CL-6080-shuffleNetV2',, 50k val as well as EVM give good accuracy so not removing now
    'cl-0200_tflitert' : 'TFL-CL-0200-mobileNetV2-1p4', #sdk8.2, accuracy around 3% down compared to float

    # During SDK8.0 removed low accuracy models where drop is more wrt reference accuracy.
    'od-2100_tflitert':'TFL-OD-2100-ssd-res50V1-fpn-coco-tpu-8-640x640', #7 % drop wrt accuracy
    'od-5030_tvmdlr': 'TVM-OD-5030-ssd-res50v1-gluon-mxnet-coco-512x512',
    'cl-0270_tflitert': 'TFL-CL-0270-mobv3-small-minimalistic', #huge drop (>30%) in 8 bit accuracy wrt float
    'od-5040_tvmdlr': 'TVM-OD-5040-ssd-mobv1-coco-gluon-mxnet-512x512', #accuracy looks ok now, but EVM run fails, SDK8.2

    #'od-8070_onnxrt': 'ONR-OD-8070-yolov3-d53-relu-coco-416x416',   #large model - issue in evm, #fixed in SDK8.2
    #'od-8130_onnxrt': 'ONR-OD-8130-yolov5-l6-ti-lite-coco-640x640', #large model - issue in evm, #fixed in SDK8.2

    # models for which artifacts are not getting generated during SDK8.0
    'cl-3510_tvmdlr' :  'TVM-CL-3510-hrnet-w30-c-gluon-mxnet',
    'cl-6508_onnxrt' :  'ONR-CL-6508-mobv3-ti-lite-large-qat',
    'cl-3498_tvmdlr' :  'TVM-CL-3498-mobv3-ti-lite-large-qat',
    'cl-6510_onnxrt' :  'ONR-CL-6510-mobv3-ti-lite-large-x2r',
    'cl-3500_tvmdlr' :  'TVM-CL-3500-mobv3-ti-lite-large-x2r',
    'od-2120_tflitert': 'TFL-OD-2120-ssd-res50V1-fpn-coco-tpu-8-1024x1024',
    'od-5050_tvmdlr' :  'TVM-OD-5050-yolo3_darknet53-coco-gluon-mxnet-512x512',
    'od-5060_tvmdlr' :  'TVM-OD-5060-centernet-r18-v1b-coco-gluon-mxnet-512x512',
    'ss-5810_tvmdlr' :  'TVM-SS-5810-fcn-resNet101-cocoseg21-gluon-mxnet-480x480',
    'ss-5820_tvmdlr' :  'TVM-SS-5820-deeplab-resNet101-cocoseg21-gluon-mxnet-480x480',
    'ss-5830_tvmdlr' : 'TVM-SS-5830-fcn-resNet50-ade20k-gluon-mxnet-480x480',

    # During SDK8.1, these models were marked as removed - not running on device (may be due to large memory requirement)
    #'cl-6450_onnxrt': 'ONR-CL-6450-harDNet85', #working in SDK8.2 onwards
    #'od-8060_onnxrt': 'ONR-OD-8060-ssd-lite-regNetX-1.6gf-fpn-bgr-coco-768x768', #working in SDK8.2 onwards
    #'od-8120_onnxrt': 'ONR-OD-8120-yolov5-m6-ti-lite-coco-640x640', #working in SDK8.2 onwards
    #'od-8130_onnxrt': 'ONR-OD-8130-yolov5-l6-ti-lite-coco-640x640', #working in SDK8.2 onwards

    'od-2150_tflitert':'TFL-OD-2150-efficientdet-lite1-relu-coco-384x384',
    'od-2170_tflitert':'TFL-OD-2170-efficientdet-lite3-relu-coco-512x512',

    # High resolution models
    'cl-6151_onnxrt': 'ONR-CL-6151-mobileNetV2-1p4-qat-512x512',
    'cl-6152_onnxrt': 'ONR-CL-6152-mobileNetV2-1p4-qat-1024x1024',

    'od-8160_onnxrt': 'ONR-OD-8160-ssdlite-mobv2-fpn-lite-coco-512x512', #Manu removed it in SDK8.2
    'od-8170_onnxrt': 'ONR-OD-8170-ssdlite-regNetX-800mf-fpn-lite-coco-512x512', #Manu removed it in SDK8.2

    'cl-0038_tflitert': 'TFL-CL-0038-InceptionNetV1', #Kumar reported issue with SDK8.2
    'cl-0150_tflitert': 'TFL-CL-0150-denseNet', #Kumar reported issue with SDK8.2
    'ss-8670_onnxrt': 'ONR-SS-8670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512', #Paula reported issue with SDK8.2 TIDL-1589

    'de-7300_onnxrt': 'ONR-DE-7300-depth-est-fast-depth-224x224', #artifacts get generated. Inference did not run on PC ref, SDK8.2
    'ss-5670_tvmdlr': 'TVM-SS-5670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512', #looks like removed from the run. Artifacts were not there.SDK8.2

    'cl-3430_tvmdlr': 'TVM-CL-3430-gluoncv-mxnet-xception', #removed in SDK8.2 size 299 is not supported in SDK/gstreamer
    'cl-0040_tflitert': 'TFL-CL-0040-InceptionNetV3', #removed in SDK8.2 size 299 is not supported in SDK/gstreamer
}

removed_models_from_plots = {
    'ss-2590_tflitert': 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512', # SS on pascal has only 2 models. So removed from plots.
    'ss-2600_tflitert': 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512', # # SS on pascal has only 2 models.  So removed from plots.
}

recommended_model_list = {
    'cl-0000_tflitert': 'TFL-CL-0000-mobileNetV1-mlperf',
    'od-2020_tflitert': 'TFL-OD-2020-ssdLite-mobDet-DSP-coco-320x320',
    'ss-2580_tflitert': 'TFL-SS-2580-deeplabv3_mobv2-ade20k32-mlperf-512x512',
    'cl-3410_tvmdlr': 'TVM-CL-3410-gluoncv-mxnet-mobv2',
    'od-5020_tvmdlr': 'TVM-OD-5020-yolov3-mobv1-gluon-mxnet-coco-416x416',
    'ss-5720_tvmdlr' : 'TVM-SS-5720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',
    'cl-6360_onnxrt': 'ONR-CL-6360-regNetx-200mf',
    'od-8050_onnxrt': 'ONR-OD-8050-ssd-lite-regNetX-800mf-fpn-bgr-coco-512x512',
    'od-8220_onnxrt': 'ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640',
    'od-8420_onnxrt': 'ONR-OD-8420-yolox-s-lite-mmdet-widerface-640x640',
    'ss-8610_onnxrt': 'ONR-SS-8610-deeplabv3lite-mobv2-ade20k32-512x512',
    'kd-7060_onnxrt': 'ONR-KD-7060-human-pose-yolox-s-640x640',
}

#sampled on 29th Jul 21
super_set = [
'cl-10-010-0_tflitert',
'cl-10-011-0_tflitert',
'cl-10-012-0_tflitert',
'cl-10-030-0_onnxrt',
'cl-10-031-0_onnxrt',
'cl-10-031-1_onnxrt',
'cl-10-031-2_onnxrt',
'cl-10-032-0_onnxrt',
'cl-10-032-1_onnxrt',
'cl-10-032-2_onnxrt',
'cl-10-033-0_onnxrt',
'cl-10-033-1_onnxrt',
'cl-10-033-2_onnxrt',
'cl-10-060-0_tvmdlr',
'cl-10-061-0_tvmdlr',
'cl-10-062-0_tvmdlr',
'cl-10-063-0_tvmdlr',
'cl-10-100-0_onnxrt',
'cl-10-100-1_onnxrt',
'cl-10-100-2_onnxrt',
'cl-10-101-0_onnxrt',
'cl-10-101-1_onnxrt',
'cl-10-101-2_onnxrt',
'cl-10-101-8_onnxrt',
'cl-10-102-1_onnxrt',
'cl-10-102-2_onnxrt',
'cl-10-102-8_onnxrt',
'cl-10-105-0_onnxrt',
'cl-10-105-8_onnxrt',
'cl-10-106-0_onnxrt',
'cl-10-301-0_onnxrt',
'cl-10-302-0_onnxrt',
'cl-10-302-8_onnxrt',
'cl-10-304-0_onnxrt',
'cl-10-304-1_onnxrt',
'cl-10-304-2_onnxrt',
'cl-10-305-0_onnxrt',
'cl-10-305-1_onnxrt',
'cl-10-305-2_onnxrt',
'cl-10-330-0_onnxrt',
'cl-10-331-0_onnxrt',
'cl-10-332-0_onnxrt',
'cl-10-333-0_onnxrt',
'cl-10-400-0_tflitert',
'cl-10-400-8_tflitert',
'cl-10-401-0_tflitert',
'cl-10-402-0_tflitert',
'cl-10-403-0_tflitert',
'cl-10-404-0_tflitert',
'cl-10-405-8_tflitert',
'cl-10-406-0_tflitert',
'cl-10-407-0_tflitert',
'cl-10-409-0_tflitert',
'cl-10-410-0_tflitert',
'cl-10-411-0_tflitert',
'cl-10-412-0_tflitert',
'cl-10-430-0_tflitert',
'cl-10-431-0_tflitert',
'cl-10-434-0_tflitert',
'cl-10-440-0_tflitert',
'cl-10-441-0_tflitert',
'cl-10-442-0_tflitert',
'od-12-010-0_tflitert',
'od-12-011-0_tflitert',
'od-12-012-0_onnxrt',
'od-12-061-0_tvmdlr',
'od-12-062-0_tvmdlr',
'od-12-100-0_onnxrt',
'od-12-101-0_onnxrt',
'od-12-102-0_onnxrt',
'od-12-103-0_onnxrt',
'od-12-104-0_onnxrt',
'od-12-105-0_onnxrt',
'od-12-106-0_onnxrt',
'od-12-107-0_onnxrt',
'od-12-400-0_tflitert',
'od-12-401-0_tflitert',
'od-12-402-0_tflitert',
'od-12-410-0_tflitert',
'od-12-412-0_tflitert',
'od-12-413-0_tflitert',
'od-12-414-0_tflitert',
'od-12-420-0_tflitert',
'kd-25-101-0_onnxrt',
'kd-25-102-0_onnxrt',
'kd-25-103-0_onnxrt',
'kd-25-104-0_onnxrt',
'ss-17-400-0_tflitert',
'ss-18-010-0_tflitert',
'ss-18-100-0_onnxrt',
'ss-18-101-0_onnxrt',
'ss-18-102-0_onnxrt',
'ss-18-103-0_onnxrt',
'ss-18-110-0_onnxrt',
'ss-18-111-0_onnxrt',
'ss-19-400-0_tflitert',
'ss-19-401-0_tflitert',
'ss-21-100-0_onnxrt',
'ss-21-106-0_onnxrt',
'ss-21-110-0_onnxrt',
]

def find_compiled_artifact_in_removed_list():
    import yaml
    compiled_artifacts_file = '/data/hdd/a0875091/files/work/bitbucket/model-selection-tool/utils/cloud_eval/perf_acc_rel_8.2_202203/compiled_artifacts.yaml'
    with open(compiled_artifacts_file) as file:
        compiled_artifacts = yaml.full_load(file)
    
        for key, value in compiled_artifacts.items():
            if key in removed_model_list:
                print("{} for which artifacts are available but model part of removed model list".format(key))


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


def is_shortlisted_model(artifact_id):
    removed_model_entries = removed_model_list.keys()
    is_removed = (artifact_id in removed_model_entries)
    is_shortlisted = (not is_removed)
    return is_shortlisted


def is_recommended_model(artifact_id):
    recommended_model_entries = recommended_model_list.keys()
    is_recommended = (artifact_id in recommended_model_entries)
    return is_recommended


if __name__ == '__main__':
    find_compiled_artifact_in_removed_list()
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
        df = get_missing_models(report_file='./work_dirs/modelartifacts/report_20210727-235437.csv',
            selected_models_list=selected_models_list)