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
#ver:16 2021-07-29
model_id_artifacts_pair = {
    # TFLite CL
    'vcls-0000_tflitert': 'TFL-CL-0000-mobileNetV1-mlperf',
    'vcls-0010_tflitert': 'TFL-CL-0010-mobileNetV2',
    'vcls-0020_tflitert': 'TFL-CL-0020-SqueezeNet',
    'vcls-0038_tflitert': 'TFL-CL-0038-InceptionNetV1',
    'vcls-0040_tflitert': 'TFL-CL-0040-InceptionNetV3',
    'vcls-0050_tflitert': 'TFL-CL-0050-resNet50V1',
    'vcls-0060_tflitert': 'TFL-CL-0060-resNet50V2',
    'vcls-0070_tflitert': 'TFL-CL-0070-mnasNet',
    'vcls-0080_tflitert': 'TFL-CL-0080-mobileNet-edgeTPU-mlperf',
    'vcls-0090_tflitert': 'TFL-CL-0090-efficientNet-edgeTPU-s',
    'vcls-0100_tflitert': 'TFL-CL-0100-efficientNet-edgeTPU-m',
    'vcls-0130_tflitert': 'TFL-CL-0130-efficientNet-lite0',
    'vcls-0140_tflitert': 'TFL-CL-0140-efficientNet-lite4',
    'vcls-0150_tflitert': 'TFL-CL-0150-denseNet',
    'vcls-0160_tflitert': 'TFL-CL-0160-resNet50V1p5-mlperf',
    'vcls-0170_tflitert': 'TFL-CL-0170-efficientNet-lite1',
    'vcls-0180_tflitert': 'TFL-CL-0180-efficientNet-lite2',
    'vcls-0190_tflitert': 'TFL-CL-0190-efficientNet-edgeTPU-l',
    'vcls-0200_tflitert': 'TFL-CL-0200-mobileNetV2-1p4',
    'vcls-0210_tflitert': 'TFL-CL-0210-mobileNetV1', 
    'vcls-0218_tflitert': 'TFL-CL-0218-mobileNetV1-qat', 
    'vcls-0018_tflitert': 'TFL-CL-0018-mobileNetV2-qat', 
    'vcls-0240_tflitert': 'TFL-CL-0240-nasNet-mobile-tflite',
    'vcls-0250_tflitert': 'TFL-CL-0250-xceptionNet-tflite', # mxnet model replaced with with tflite model now
    'vcls-0260_tflitert': 'TFL-CL-0260-mobv3-large-minimalistic', #added in SDK8.0
    'vcls-0270_tflitert': 'TFL-CL-0270-mobv3-small-minimalistic', #added in SDK8.0
        
    # TFLite OD
    'vdet-2000_tflitert': 'TFL-OD-2000-ssd-mobV1-coco-mlperf-300x300',
    'vdet-2010_tflitert': 'TFL-OD-2010-ssd-mobV2-coco-mlperf-300x300', 
    'vdet-2020_tflitert': 'TFL-OD-2020-ssdLite-mobDet-DSP-coco-320x320',
    'vdet-2030_tflitert': 'TFL-OD-2030-ssdLite-mobDet-EdgeTPU-coco-320x320',
    'vdet-2040_tflitert': 'TFL-OD-2040-ssd-mobV1-fpn-coco-640x640',
    'vdet-2050_tflitert': 'TFL-OD-2050-ssd-mobV2-mnas-fpn-coco-320x320',
    'vdet-2060_tflitert': 'TFL-OD-2060-ssd-mobV2-coco-mlperf-300x300',

    # added in SDK8.0
    'vdet-2070_tflitert':'TFL-OD-2070-ssd-mobV1-fpn-coco-tpu-8-640x640',
    'vdet-2080_tflitert':'TFL-OD-2080-ssd-mobV2-fpnlite-coco-tpu-8-320x320',
    'vdet-2090_tflitert':'TFL-OD-2090-ssd-mobV2-fpnlite-coco-tpu-8-640x640',
    'vdet-2100_tflitert':'TFL-OD-2100-ssd-res50V1-fpn-coco-tpu-8-640x640',
    'vdet-2110_tflitert':'TFL-OD-2110-efficient-det-ti-lite0-512x512',
    'vdet-2120_tflitert':'TFL-OD-2120-ssd-res50V1-fpn-coco-tpu-8-1024x1024', #'vdet-12-415-0'
    'vdet-2130_tflitert':'TFL-OD-2130-ssd-mobV2-coco-tpu-8-320x320', #'vdet-12-411-0'

    # TFLite SS
    'vseg-2500_tflitert': 'TFL-SS-2500-deeplab-mobV2-ade20k-512x512',
    'vseg-2540_tflitert': 'TFL-SS-2540-deeplabv3-mobv2-ade20k-512x512',
    'vseg-2550_tflitert': 'TFL-SS-2550-deeplabv3-mobv2-cs-2048x1024',
    'vseg-2580_tflitert': 'TFL-SS-2580-deeplabv3_mobv2-ade20k32-mlperf-512x512',
    'vseg-2590_tflitert': 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512',
    'vseg-2600_tflitert': 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512',

    # TVM- CL
    'vcls-3000_tvmdlr': 'TVM-CL-3000-resNet18V2',
    'vcls-3020_tvmdlr': 'TVM-CL-3020-xceptionNet-mxnet',
    'vcls-3040_tvmdlr': 'TVM-CL-3040-nasNet-mobile-tflite',
    'vcls-3060_tvmdlr': 'TVM-CL-3060-mobileNetV1',
    'vcls-3070_tvmdlr': 'TVM-CL-3070-mobileNetV2',
    'vcls-3080_tvmdlr': 'TVM-CL-3080-shuffleNetV2',
    'vcls-3090_tvmdlr': 'TVM-CL-3090-mobileNetV2-tv',
    'vcls-3100_tvmdlr': 'TVM-CL-3100-resNet18',
    'vcls-3110_tvmdlr': 'TVM-CL-3110-resNet50',
    'vcls-3120_tvmdlr': 'TVM-CL-3120-regNetX-400mf',
    'vcls-3130_tvmdlr': 'TVM-CL-3130-regNetX-800mf',
    'vcls-3140_tvmdlr': 'TVM-CL-3140-regNetX-1.6gf',
    'vcls-3150_tvmdlr': 'TVM-CL-3150-mobileNetV2-1p4-qat',

    #512x512
    'vcls-3061_tvmdlr': 'TVM-CL-3061-mobileNetV1-512x512',
    'vcls-3071_tvmdlr': 'TVM-CL-3071-mobileNetV2-512x512',
    'vcls-3081_tvmdlr': 'TVM-CL-3081-shuffleNetV2-512x512',
    'vcls-3091_tvmdlr': 'TVM-CL-3091-mobileNetV2-tv-512x512',
    'vcls-3101_tvmdlr': 'TVM-CL-3101-resNet18-512x512',
    'vcls-3111_tvmdlr': 'TVM-CL-3111-resNet50-512x512',
    'vcls-3121_tvmdlr': 'TVM-CL-3121-regNetX-400mf-512x512',
    'vcls-3131_tvmdlr': 'TVM-CL-3131-regNetX-800mf-512x512',
    'vcls-3141_tvmdlr': 'TVM-CL-3141-regNetX-1.6gf-512x512',
    'vcls-3151_tvmdlr': 'TVM-CL-3151-mobileNetV2-1p4-qat-512x512',

    #1024x1024
    'vcls-3062_tvmdlr': 'TVM-CL-3062-mobileNetV1-1024x1024',
    'vcls-3072_tvmdlr': 'TVM-CL-3072-mobileNetV2-1024x1024',
    'vcls-3082_tvmdlr': 'TVM-CL-3082-shuffleNetV2-1024x1024',
    'vcls-3092_tvmdlr': 'TVM-CL-3092-mobileNetV2-tv-1024x1024',
    'vcls-3102_tvmdlr': 'TVM-CL-3102-resNet18-1024x1024',
    'vcls-3112_tvmdlr': 'TVM-CL-3112-resNet50-1024x1024',
    'vcls-3122_tvmdlr': 'TVM-CL-3122-regNetX-400mf-1024x1024',
    'vcls-3132_tvmdlr': 'TVM-CL-3132-regNetX-800mf-1024x1024',
    'vcls-3142_tvmdlr': 'TVM-CL-3142-regNetX-1.6gf-1024x1024',
    'vcls-3152_tvmdlr': 'TVM-CL-3152-mobileNetV2-1p4-qat-1024x1024',
    ########
    
    'vcls-3360_tvmdlr': 'TVM-CL-3360-regNetx-200mf',
    'vcls-3370_tvmdlr': 'TVM-CL-3370-vgg16',
    'vcls-3078_tvmdlr': 'TVM-CL-3078-mobileNetV2-qat',
    'vcls-3098_tvmdlr': 'TVM-CL-3098-mobileNetV2-tv-qat',
    'vcls-3410_tvmdlr': 'TVM-CL-3410-gluoncv-mxnet-mobv2',
    'vcls-3420_tvmdlr': 'TVM-CL-3420-gluoncv-mxnet-resNet50-v1',
    'vcls-3430_tvmdlr': 'TVM-CL-3430-gluoncv-mxnet-xception',
    # HarD Net
    'vcls-3440_tvmdlr': 'TVM-CL-3440-harDNet68',
    'vcls-3450_tvmdlr': 'TVM-CL-3450-harDNet85',
    'vcls-3460_tvmdlr': 'TVM-CL-3460-harDNet68ds',
    'vcls-3470_tvmdlr': 'TVM-CL-3470-harDNet39ds',
    
    'vcls-3480_tvmdlr': 'TVM-CL-3480-hrnet-w18-small-v2-gluon-mxnet',#added in SDK8.0
    'vcls-3490_tvmdlr': 'TVM-CL-3498-mobv3-ti-lite-large-qat', #added in SDK8.0 #'vcls-10-106-8'
    'vcls-3500_tvmdlr': 'TVM-CL-3500-mobv3-ti-lite-large-x2r', #added in SDK8.0 #'vcls-10-107-0'
    'vcls-3510_tvmdlr': 'TVM-CL-3510-hrnet-w30-c-gluon-mxnet', #added in SDK8.0 #'vcls-10-064-0'

    
    # TVM - OD
    'vdet-5000_tvmdlr': 'TVM-OD-5000-ssd1200-resNet34-mlperf-1200x1200',
    'vdet-5010_tvmdlr': 'TVM-OD-5010-yolov3-416x416',
    'vdet-5020_tvmdlr': 'TVM-OD-5020-yolov3-mobv1-gluon-mxnet-416x416',

    'vdet-5030_tvmdlr': 'TVM-OD-5030-ssd-res50v1-gluon-mxnet-512x512', #added in SDK8.0
    'vdet-5040_tvmdlr': 'TVM-OD-5040-ssd-mobv1-coco-gluon-mxnet-512x512', #renamed mobv1 model in SDK8.0

    'vdet-5050_tvmdlr': 'TVM-OD-5050-yolo3_darknet53-coco-gluon-mxnet-512x512', #renamed mobv1 model in SDK8.0 #'vdet-12-063-0'
    'vdet-5060_tvmdlr': 'TVM-OD-5060-centernet-r18-v1b-coco-gluon-mxnet-512x512', #renamed mobv1 model in SDK8.0 'vdet-12-064-0'

    # TVM - SS - CS
    'vseg-5500_tvmdlr': 'TVM-SS-5500-deeplabv3lite-mobv2-cs-768x384',
    'vseg-5510_tvmdlr': 'TVM-SS-5510-deeplabv3lite-mobv2-cs-qat-768x384',
    'vseg-5520_tvmdlr': 'TVM-SS-5520-fpnlite-aspp-mobv2-cs-768x384',
    'vseg-5528_tvmdlr': 'TVM-SS-5528-fpnlite-aspp-mobv2-cs-qat-768x384',
    'vseg-5540_tvmdlr': 'TVM-SS-5540-unetlite-aspp-mobv2-tv-cs-768x384',
    'vseg-5548_tvmdlr': 'TVM-SS-5548-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'vseg-5560_tvmdlr': 'TVM-SS-5560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'vseg-5570_tvmdlr': 'TVM-SS-5570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'vseg-5580_tvmdlr': 'TVM-SS-5580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'vseg-5590_tvmdlr': 'TVM-SS-5590-deeplabv3-res50-cs-1040x520',
    'vseg-5600_tvmdlr': 'TVM-SS-5600-fcn-res50-cs-1040x520',

    # TVM - SS - ADE20k
    'vseg-5610_tvmdlr': 'TVM-SS-5610-deeplabv3lite-mobv2-ade20k32-512x512',
    'vseg-5618_tvmdlr': 'TVM-SS-5618-deeplabv3lite-mobv2-ade20k32-qat-512x512', 
    'vseg-5630_tvmdlr': 'TVM-SS-5630-unetlite-aspp-mobv2-tv-ade20k32-512x512',
    'vseg-5638_tvmdlr': 'TVM-SS-5638-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512',
    'vseg-5650_tvmdlr': 'TVM-SS-5650-fpnlite-aspp-mobv2-ade20k32-512x512',
    'vseg-5658_tvmdlr': 'TVM-SS-5658-fpnlite-aspp-mobv2-ade20k32-qat-512x512', 
    'vseg-5670_tvmdlr': 'TVM-SS-5670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512',
    'vseg-5678_tvmdlr': 'TVM-SS-5678-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512',
    'vseg-5690_tvmdlr': 'TVM-SS-5690-fpnlite-aspp-regnetx400mf-ade20k32-384x384',
    'vseg-5700_tvmdlr': 'TVM-SS-5700-fpnlite-aspp-regnetx800mf-ade20k32-512x512',

    # TVM -SS - CocoSeg21
    'vseg-5710_tvmdlr' : 'TVM-SS-5710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'vseg-5711_tvmdlr' : 'TVM-SS-5720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',

    # ONNXRT- CL
    'vcls-6000_onnxrt': 'ONR-CL-6000-resNet18V2',
    'vcls-6040_onnxrt': 'ONR-CL-6040-nasNet-mobile-tflite',
    'vcls-6060_onnxrt': 'ONR-CL-6060-mobileNetV1',
    'vcls-6070_onnxrt': 'ONR-CL-6070-mobileNetV2',
    'vcls-6080_onnxrt': 'ONR-CL-6080-shuffleNetV2',
    'vcls-6090_onnxrt': 'ONR-CL-6090-mobileNetV2-tv',
    'vcls-6100_onnxrt': 'ONR-CL-6100-resNet18',
    'vcls-6110_onnxrt': 'ONR-CL-6110-resNet50',
    'vcls-6120_onnxrt': 'ONR-CL-6120-regNetX-400mf',
    'vcls-6130_onnxrt': 'ONR-CL-6130-regNetX-800mf',
    'vcls-6140_onnxrt': 'ONR-CL-6140-regNetX-1.6gf',
    'vcls-6150_onnxrt': 'ONR-CL-6150-mobileNetV2-1p4-qat',
    
    #512x512
    'vcls-6061_onnxrt': 'ONR-CL-6061-mobileNetV1-512x512',
    'vcls-6071_onnxrt': 'ONR-CL-6071-mobileNetV2-512x512',
    'vcls-6081_onnxrt': 'ONR-CL-6081-shuffleNetV2-512x512',
    'vcls-6091_onnxrt': 'ONR-CL-6091-mobileNetV2-tv-512x512',
    'vcls-6101_onnxrt': 'ONR-CL-6101-resNet18-512x512',
    'vcls-6111_onnxrt': 'ONR-CL-6111-resNet50-512x512',
    'vcls-6121_onnxrt': 'ONR-CL-6121-regNetX-400mf-512x512',
    'vcls-6131_onnxrt': 'ONR-CL-6131-regNetX-800mf-512x512',
    'vcls-6141_onnxrt': 'ONR-CL-6141-regNetX-1.6gf-512x512',
    'vcls-6151_onnxrt': 'ONR-CL-6151-mobileNetV2-1p4-qat-512x512',

    #1024x1024
    'vcls-6062_onnxrt': 'ONR-CL-6062-mobileNetV1-1024x1024',
    'vcls-6072_onnxrt': 'ONR-CL-6072-mobileNetV2-1024x1024',
    'vcls-6082_onnxrt': 'ONR-CL-6082-shuffleNetV2-1024x1024',
    'vcls-6092_onnxrt': 'ONR-CL-6092-mobileNetV2-tv-1024x1024',
    'vcls-6102_onnxrt': 'ONR-CL-6102-resNet18-1024x1024',
    'vcls-6112_onnxrt': 'ONR-CL-6112-resNet50-1024x1024',
    'vcls-6122_onnxrt': 'ONR-CL-6122-regNetX-400mf-1024x1024',
    'vcls-6132_onnxrt': 'ONR-CL-6132-regNetX-800mf-1024x1024',
    'vcls-6142_onnxrt': 'ONR-CL-6142-regNetX-1.6gf-1024x1024',
    'vcls-6152_onnxrt': 'ONR-CL-6152-mobileNetV2-1p4-qat-1024x1024',
    ########
    
    'vcls-6360_onnxrt': 'ONR-CL-6360-regNetx-200mf',
    'vcls-6370_onnxrt': 'ONR-CL-6370-vgg16',
    'vcls-6078_onnxrt': 'ONR-CL-6078-mobileNetV2-qat',
    'vcls-6098_onnxrt': 'ONR-CL-6098-mobileNetV2-tv-qat',
    'vcls-6410_onnxrt': 'ONR-CL-6410-gluoncv-mxnet-mobv2',
    'vcls-6420_onnxrt': 'ONR-CL-6420-gluoncv-mxnet-resNet50-v1',
    'vcls-6430_onnxrt': 'ONR-CL-6430-gluoncv-mxnet-xception',

    # HarD Net 
    # model id changed in SDK8.0
    'vcls-6440_onnxrt': 'ONR-CL-6440-harDNet68',
    'vcls-6450_onnxrt': 'ONR-CL-6450-harDNet85',
    'vcls-6460_onnxrt': 'ONR-CL-6460-harDNet68ds',
    'vcls-6470_onnxrt': 'ONR-CL-6470-harDNet39ds',

    #added in SDK8.0
    'vcls-6480_onnxrt': 'ONR-CL-6480-mobv3-lite-small',
    'vcls-6488_onnxrt': 'ONR-CL-6488-mobv3-lite-small-qat',
    'vcls-6490_onnxrt': 'ONR-CL-6490-mobv3-lite-large',

    'vcls-6508_onnxrt': 'ONR-CL-6508-mobv3-ti-lite-large-qat', #added in SDK8.0 #'vcls-10-106-8'
    'vcls-6510_onnxrt': 'ONR-CL-6510-mobv3-ti-lite-large-x2r', #added in SDK8.0 #'vcls-10-107-0'
    
    # ONNX - OD
    'vdet-8000_onnxrt': 'ONR-OD-8000-ssd1200-resNet34-mlperf-1200x1200',
    'vdet-8010_onnxrt': 'ONR-OD-8010-yolov3-416x416',

    #Added in SDK8.0
    'vdet-8020_onnxrt': 'ONR-OD-8020-ssd-lite-mobv2-coco-512x512',
    'vdet-8030_onnxrt': 'ONR-OD-8030-ssd-lite-mobv2-fpn-coco-512x512',
    'vdet-8040_onnxrt': 'ONR-OD-8040-ssd-lite-regNetX-200mf-fpn-bgr-coco-320x320',
    'vdet-8050_onnxrt': 'ONR-OD-8050-ssd-lite-regNetX-800mf-fpn-bgr-coco-512x512',
    'vdet-8060_onnxrt': 'ONR-OD-8060-ssd-lite-regNetX-1.6gf-bifpn168x4-bgr-coco-768x768',
    'vdet-8070_onnxrt': 'ONR-OD-8070-yolov3-d53-relu-coco-416x416',
    'vdet-8080_onnxrt': 'ONR-OD-8080-yolov3-lite-regNetX-1.6gf-bgr-coco-512x512',
    'vdet-8090_onnxrt': 'ONR-OD-8090-retina-lite-regNetX-800mf-fpn-bgr-coco-512x512',

    #Yolov5 series added in SDK8.0
    'vdet-8100_onnxrt': 'ONR-OD-8100-yolov5-s6-ti-lite-coco-640x640',
    'vdet-8110_onnxrt': 'ONR-OD-8110-yolov5-s6-ti-lite-coco-384x384',
    'vdet-8120_onnxrt': 'ONR-OD-8120-yolov5-m6-ti-lite-coco-640x640',
    'vdet-8130_onnxrt': 'ONR-OD-8130-yolov5-l6-ti-lite-coco-640x640',

    # ONNX - SS - CS
    'vseg-8500_onnxrt': 'ONR-SS-8500-deeplabv3lite-mobv2-cs-768x384',
    'vseg-8508_onnxrt': 'ONR-SS-8508-deeplabv3lite-mobv2-cs-qat-768x384',
    'vseg-8520_onnxrt': 'ONR-SS-8520-fpnlite-aspp-mobv2-cs-768x384',
    'vseg-8528_onnxrt': 'ONR-SS-8528-fpnlite-aspp-mobv2-cs-qat-768x384',
    'vseg-8540_onnxrt': 'ONR-SS-8540-unetlite-aspp-mobv2-tv-cs-768x384',
    'vseg-8548_onnxrt': 'ONR-SS-8548-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'vseg-8560_onnxrt': 'ONR-SS-8560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'vseg-8570_onnxrt': 'ONR-SS-8570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'vseg-8580_onnxrt': 'ONR-SS-8580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'vseg-8590_onnxrt': 'ONR-SS-8590-deeplabv3-res50-1040x520',
    'vseg-8600_onnxrt': 'ONR-SS-8600-fcn-res50-1040x520',

    # ONNX - SS - ADE20k
    'vseg-8610_onnxrt': 'ONR-SS-8610-deeplabv3lite-mobv2-ade20k32-512x512',
    'vseg-8618_onnxrt': 'ONR-SS-8618-deeplabv3lite-mobv2-ade20k32-qat-512x512', 
    'vseg-8630_onnxrt': 'ONR-SS-8630-unetlite-aspp-mobv2-tv-ade20k32-512x512',
    'vseg-8638_onnxrt': 'ONR-SS-8638-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512',
    'vseg-8650_onnxrt': 'ONR-SS-8650-fpnlite-aspp-mobv2-ade20k32-512x512',
    'vseg-8658_onnxrt': 'ONR-SS-8658-fpnlite-aspp-mobv2-ade20k32-qat-512x512', 
    'vseg-8670_onnxrt': 'ONR-SS-8670-fpnlite-aspp-mobv2-1p4-ade20k32-512x512',
    'vseg-8678_onnxrt': 'ONR-SS-8678-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512',
    'vseg-8690_onnxrt': 'ONR-SS-8690-fpnlite-aspp-regnetx400mf-ade20k32-384x384',
    'vseg-8700_onnxrt': 'ONR-SS-8700-fpnlite-aspp-regnetx800mf-ade20k32-512x512',

    # ONNX-SS - CocoSeg21
    'vseg-8710_onnxrt': 'ONR-SS-8710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'vseg-8720_onnxrt': 'ONR-SS-8720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',
    'vseg-8730_onnxrt': 'ONR-SS-8730-deeplabv3-mobv3-lite-large-cocoseg21-512x512', #added in SDK8.0
    'vseg-8731_onnxrt': 'ONR-SS-8731-lraspp-mobV3-ti-lite-large-cocoseg21-512x512', #added in SDK8.0 #'vseg-21-105-0'

    #ONNX key point detections
    'vkpdet-7000_onnxrt': 'ONR-KD-7000-human-pose-ae-mobv2-fpn-spp-udp-512x512',
    'vkpdet-7010_onnxrt': 'ONR-KD-7010-human-pose-ae-res50v2-fpn-spp-udp-512x512',
    'vkpdet-7020_onnxrt': 'ONR-KD-7020-human-pose-ae-mobv2-pan-spp-udp-512x512',
    'vkpdet-7030_onnxrt': 'ONR-KD-7030-human-pose-ae-res50v2-pan-spp-udp-512x512',
}

removed_model_list = {
    'vcls-3020_tvmdlr' : 'TVM-CL-3020-xceptionNet-mxnet', # this is replaced with tflite model now (that was also eventually removed)
    'vcls-0230_tflitert': 'TFL-CL-0230-mobileNetV2-qat',  # QAT model is not giving good accuracy so keep only float
    'vdet-5000_tvmdlr': 'TVM-OD-5000-ssd1200-resNet34-mlperf-1200x1200', # Not working with TVM. Will need to park it till ONNX RT OD support is available.
    #'vdet-12-012-0_onnxrt': 'ONR-OD-800-ssd1200-resNet34-mlperf-1200x1200', #ONNX does not support OD yet
    ################ CS models
    # cityscapes model not part of Model Zoo
    'vseg-5500_tvmdlr': 'TVM-SS-5500-deeplabv3lite-mobv2-cs-768x384',
    'vseg-5510_tvmdlr': 'TVM-SS-5510-deeplabv3lite-mobv2-cs-qat-768x384',
    'vseg-5520_tvmdlr': 'TVM-SS-5520-fpnlite-aspp-mobv2-cs-768x384',
    'vseg-5528_tvmdlr': 'TVM-SS-5528-fpnlite-aspp-mobv2-cs-qat-768x384',
    'vseg-5540_tvmdlr': 'TVM-SS-5540-unetlite-aspp-mobv2-tv-cs-768x384',
    'vseg-5548_tvmdlr': 'TVM-SS-5548-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'vseg-5560_tvmdlr': 'TVM-SS-5560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'vseg-5570_tvmdlr': 'TVM-SS-5570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'vseg-5580_tvmdlr': 'TVM-SS-5580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'vseg-5590_tvmdlr': 'TVM-SS-5590-deeplabv3-res50-cs-1040x520',
    'vseg-5600_tvmdlr': 'TVM-SS-5600-fcn-res50-cs-1040x520',
    # cityscapes model not part of Model Zoo
    'vseg-8500_onnxrt': 'ONR-SS-8500-deeplabv3lite-mobv2-cs-768x384',
    'vseg-8508_onnxrt': 'ONR-SS-8508-deeplabv3lite-mobv2-cs-qat-768x384',
    'vseg-8520_onnxrt': 'ONR-SS-8520-fpnlite-aspp-mobv2-cs-768x384',
    'vseg-8528_onnxrt': 'ONR-SS-8528-fpnlite-aspp-mobv2-cs-qat-768x384',
    'vseg-8540_onnxrt': 'ONR-SS-8540-unetlite-aspp-mobv2-tv-cs-768x384',
    'vseg-8548_onnxrt': 'ONR-SS-8548-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'vseg-8560_onnxrt': 'ONR-SS-8560-fpnlite-aspp-regNetx800mf-cs-768x384',
    'vseg-8570_onnxrt': 'ONR-SS-8570-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'vseg-8580_onnxrt': 'ONR-SS-8580-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'vseg-8590_onnxrt': 'ONR-SS-8590-deeplabv3-res50-cs-1040x520',
    'vseg-8600_onnxrt': 'ONR-SS-8600-fcn-res50-cs-1040x520',
    ################
    'vcls-0240_tflitert': 'TFL-CL-0240-nasNet-mobile-tflite', # Kumar removed model (Multiple sub-graphs) (Accuracy issue)
    'vcls-0180_tflitert': 'TFL-CL-0180-efficientNet-lite2', # Kumar removed model  (Accuracy issue)
    'vdet-2010_tflitert': 'TFL-OD-2010-ssd-mobV2-coco-300x300-mlperf',  # Kumar removed model (Duplocate)
    'vseg-2550_tflitert': 'TFL-SS-2550-deeplabv3-mobv2-cs-2048x1024',  # Kumar removed model, (not in plan)

    'vcls-3370_tvmdlr': 'TVM-CL-3370-vgg16', # Kumar removed model
    'vcls-3000_tvmdlr': 'TVM-CL-3000-resNet18V2', # Kumar removed model
    'vseg-5590_tvmdlr': 'TVM-SS-5590-deeplabv3-res50-1040x520', # Kumar removed model, nc does not have info for this
    'vseg-5600_tvmdlr': 'TVM-SS-5600-fcn-res50-1040x520', # Kumar removed model

    'vcls-6370_onnxrt': 'ONR-CL-6370-vgg16', # Kumar removed model for TVM 
    'vcls-6000_onnxrt': 'ONR-CL-6000-resNet18V2', # Kumar removed model  for TVM 
    'vseg-8590_onnxrt': 'ONR-SS-8590-deeplabv3-res50-1040x520', # Kumar removed model, nc does not have info for this  for TVM 
    'vseg-8600_onnxrt': 'ONR-SS-8600-fcn-res50-1040x520', # Kumar removed model  for TVM 
    
    #########################
    'vseg-2500_tflitert': 'TFL-SS-2500-deeplab-mobV2-ade20k-512x512', # Manu said incorrect model ID removed. vseg-17-010 is replaced with vseg-18-010
    'vcls-0250_tflitert': 'TFL-CL-0250-xceptionNet-tflite', # mxnet model replaced with with tflite model now. Eventually removed as size is quite big.
    'vdet-2040_tflitert': 'TFL-OD-2040-ssd-mobV1-FPN-coco-640x640', # does not run, import crashes. Manu
    'vdet-2050_tflitert': 'TFL-OD-2050-ssd-mobV2-mnas-fpn-coco-320x320', # does not run, import crashes. Manu

    'vcls-3040_tvmdlr': 'TVM-CL-3040-nasNet-mobile-tflite', # not part of benchmarking script yet. tflite model with TVM.
    'vdet-5010_tvmdlr': 'TVM-OD-5010-yolov3-416x416', # not supported yet

    'vcls-6040_onnxrt': 'ONR-CL-6040-nasNet-mobile-tflite', # not part of benchmarking script yet. tflite model with TVM.
    'vdet-8010_onnxrt': 'ONR-OD-8010-yolov3-416x416', # not supported yet
    'vcls-0150_tflitert': 'TFL-CL-0150-denseNet', # too far from optimal pareto line
    'vdet-5020_tvmdlr': 'TVM-OD-5020-yolov3-mobv1-gluon-mxnet-416x416', #segmentation fault while running import

    #removed from ONNX-RT
    'vcls-6410_onnxrt': 'ONR-CL-6410-gluoncv-mxnet-mobv2',
    'vcls-6420_onnxrt': 'ONR-CL-6420-gluoncv-mxnet-resNet50-v1',
    'vcls-6430_onnxrt': 'ONR-CL-6430-gluoncv-mxnet-xception',

    #ADE20k32 models
    'vseg-5618_tvmdlr': 'TVM-SS-5618-deeplabv3lite-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-5638_tvmdlr': 'TVM-SS-5638-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512', # import fails
    'vseg-5658_tvmdlr': 'TVM-SS-5658-fpnlite-aspp-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-5678_tvmdlr': 'TVM-SS-5678-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed

    'vseg-8618_onnxrt': 'ONR-SS-8618-deeplabv3lite-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-8638_onnxrt': 'ONR-SS-8638-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512', # import fails
    'vseg-8658_onnxrt': 'ONR-SS-8658-fpnlite-aspp-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-8678_onnxrt': 'ONR-SS-8678-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed

    #512x512 (Only for performance)
    'vcls-3061_tvmdlr': 'TVM-CL-3061-mobileNetV1-512x512',
    'vcls-3071_tvmdlr': 'TVM-CL-3071-mobileNetV2-512x512',
    'vcls-3081_tvmdlr': 'TVM-CL-3081-shuffleNetV2-512x512',
    'vcls-3091_tvmdlr': 'TVM-CL-3091-mobileNetV2-tv-512x512',
    'vcls-3101_tvmdlr': 'TVM-CL-3101-resNet18-512x512',
    'vcls-3111_tvmdlr': 'TVM-CL-3111-resNet50-512x512',
    'vcls-3121_tvmdlr': 'TVM-CL-3121-regNetX-400mf-512x512',
    'vcls-3131_tvmdlr': 'TVM-CL-3131-regNetX-800mf-512x512',
    'vcls-3141_tvmdlr': 'TVM-CL-3141-regNetX-1.6gf-512x512',
    'vcls-3151_tvmdlr': 'TVM-CL-3151-mobileNetV2-1p4-qat-512x512',
    
    'vcls-6061_onnxrt': 'ONR-CL-6061-mobileNetV1-512x512',
    'vcls-6071_onnxrt': 'ONR-CL-6071-mobileNetV2-512x512',
    'vcls-6081_onnxrt': 'ONR-CL-6081-shuffleNetV2-512x512',
    'vcls-6091_onnxrt': 'ONR-CL-6091-mobileNetV2-tv-512x512',
    'vcls-6101_onnxrt': 'ONR-CL-6101-resNet18-512x512',
    'vcls-6111_onnxrt': 'ONR-CL-6111-resNet50-512x512',
    'vcls-6121_onnxrt': 'ONR-CL-6121-regNetX-400mf-512x512',
    'vcls-6131_onnxrt': 'ONR-CL-6131-regNetX-800mf-512x512',
    'vcls-6141_onnxrt': 'ONR-CL-6141-regNetX-1.6gf-512x512',
    'vcls-6151_onnxrt': 'ONR-CL-6151-mobileNetV2-1p4-qat-512x512',

    #1024x1024  (Only for performance)
    'vcls-3062_tvmdlr': 'TVM-CL-3062-mobileNetV1-1024x1024',
    'vcls-3072_tvmdlr': 'TVM-CL-3072-mobileNetV2-1024x1024',
    'vcls-3082_tvmdlr': 'TVM-CL-3082-shuffleNetV2-1024x1024',
    'vcls-3092_tvmdlr': 'TVM-CL-3092-mobileNetV2-tv-1024x1024',
    'vcls-3102_tvmdlr': 'TVM-CL-3102-resNet18-1024x1024',
    'vcls-3112_tvmdlr': 'TVM-CL-3112-resNet50-1024x1024',
    'vcls-3122_tvmdlr': 'TVM-CL-3122-regNetX-400mf-1024x1024',
    'vcls-3132_tvmdlr': 'TVM-CL-3132-regNetX-800mf-1024x1024',
    'vcls-3142_tvmdlr': 'TVM-CL-3142-regNetX-1.6gf-1024x1024',
    'vcls-3152_tvmdlr': 'TVM-CL-3152-mobileNetV2-1p4-qat-1024x1024',

    'vcls-6062_onnxrt': 'ONR-CL-6062-mobileNetV1-1024x1024',
    'vcls-6072_onnxrt': 'ONR-CL-6072-mobileNetV2-1024x1024',
    'vcls-6082_onnxrt': 'ONR-CL-6082-shuffleNetV2-1024x1024',
    'vcls-6092_onnxrt': 'ONR-CL-6092-mobileNetV2-tv-1024x1024',
    'vcls-6102_onnxrt': 'ONR-CL-6102-resNet18-1024x1024',
    'vcls-6112_onnxrt': 'ONR-CL-6112-resNet50-1024x1024',
    'vcls-6122_onnxrt': 'ONR-CL-6122-regNetX-400mf-1024x1024',
    'vcls-6132_onnxrt': 'ONR-CL-6132-regNetX-800mf-1024x1024',
    'vcls-6142_onnxrt': 'ONR-CL-6142-regNetX-1.6gf-1024x1024',
    'vcls-6152_onnxrt': 'ONR-CL-6152-mobileNetV2-1p4-qat-1024x1024',

    # ONNX-SS - CocoSeg21 Models are not added yet
    'vseg-8710_onnxrt' : 'ONR-SS-8710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'vseg-8720_onnxrt' : 'ONR-SS-8720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',

    # TVM-SS-CocoSeg21 Models are not added yet
    'vseg-5710_tvmdlr' : 'TVM-SS-5710-deeplabv3lite-mobv2-cocoseg21-512x512',
    'vseg-5720_tvmdlr' : 'TVM-SS-5720-deeplabv3lite-regnetx800mf-cocoseg21-512x512',

    #removed low accuracy models where drop is more wrt reference accuracy. This will be corrected in the next release.
    'vseg-2540_tflitert' : 'TFL-SS-2540-deeplabv3-mobv2-ade20k-512x512',
    'vseg-2590_tflitert' : 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512',
    'vcls-0060_tflitert' : 'TFL-CL-0060-resNet50V2',
    'vseg-2600_tflitert' : 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512',
    'vcls-0218_tflitert' : 'TFL-CL-0218-mobileNetV1-qat',
    'vcls-0010_tflitert' : 'TFL-CL-0010-mobileNetV2',
    'vcls-6080_onnxrt' : 'ONR-CL-6080-shuffleNetV2',
    'vcls-0200_tflitert' : 'TFL-CL-0200-mobileNetV2-1p4',
    'vcls-3430_tvmdlr' : 'TVM-CL-3430-gluoncv-mxnet-xception',
}

removed_models_from_plots = {
    'vseg-2590_tflitert': 'TFL-SS-2590-deeplabv3_mobv2-dm05-pascal-trainaug-512x512', # SS on pascal has only 2 models. So removed from plots.
    'vseg-2600_tflitert': 'TFL-SS-2600-deeplabv3_mobv2-pascal-trainaug-512x512', # # SS on pascal has only 2 models.  So removed from plots.
}

#sampled on 29th Jul 21
super_set = [
'vcls-10-010-0_tflitert',
'vcls-10-011-0_tflitert',
'vcls-10-012-0_tflitert',
'vcls-10-030-0_onnxrt',
'vcls-10-031-0_onnxrt',
'vcls-10-031-1_onnxrt',
'vcls-10-031-2_onnxrt',
'vcls-10-032-0_onnxrt',
'vcls-10-032-1_onnxrt',
'vcls-10-032-2_onnxrt',
'vcls-10-033-0_onnxrt',
'vcls-10-033-1_onnxrt',
'vcls-10-033-2_onnxrt',
'vcls-10-060-0_tvmdlr',
'vcls-10-061-0_tvmdlr',
'vcls-10-062-0_tvmdlr',
'vcls-10-063-0_tvmdlr',
'vcls-10-100-0_onnxrt',
'vcls-10-100-1_onnxrt',
'vcls-10-100-2_onnxrt',
'vcls-10-101-0_onnxrt',
'vcls-10-101-1_onnxrt',
'vcls-10-101-2_onnxrt',
'vcls-10-101-8_onnxrt',
'vcls-10-102-1_onnxrt',
'vcls-10-102-2_onnxrt',
'vcls-10-102-8_onnxrt',
'vcls-10-105-0_onnxrt',
'vcls-10-105-8_onnxrt',
'vcls-10-106-0_onnxrt',
'vcls-10-301-0_onnxrt',
'vcls-10-302-0_onnxrt',
'vcls-10-302-8_onnxrt',
'vcls-10-304-0_onnxrt',
'vcls-10-304-1_onnxrt',
'vcls-10-304-2_onnxrt',
'vcls-10-305-0_onnxrt',
'vcls-10-305-1_onnxrt',
'vcls-10-305-2_onnxrt',
'vcls-10-330-0_onnxrt',
'vcls-10-331-0_onnxrt',
'vcls-10-332-0_onnxrt',
'vcls-10-333-0_onnxrt',
'vcls-10-400-0_tflitert',
'vcls-10-400-8_tflitert',
'vcls-10-401-0_tflitert',
'vcls-10-402-0_tflitert',
'vcls-10-403-0_tflitert',
'vcls-10-404-0_tflitert',
'vcls-10-405-8_tflitert',
'vcls-10-406-0_tflitert',
'vcls-10-407-0_tflitert',
'vcls-10-409-0_tflitert',
'vcls-10-410-0_tflitert',
'vcls-10-411-0_tflitert',
'vcls-10-412-0_tflitert',
'vcls-10-430-0_tflitert',
'vcls-10-431-0_tflitert',
'vcls-10-434-0_tflitert',
'vcls-10-440-0_tflitert',
'vcls-10-441-0_tflitert',
'vcls-10-442-0_tflitert',
'vdet-12-010-0_tflitert',
'vdet-12-011-0_tflitert',
'vdet-12-012-0_onnxrt',
'vdet-12-061-0_tvmdlr',
'vdet-12-062-0_tvmdlr',
'vdet-12-100-0_onnxrt',
'vdet-12-101-0_onnxrt',
'vdet-12-102-0_onnxrt',
'vdet-12-103-0_onnxrt',
'vdet-12-104-0_onnxrt',
'vdet-12-105-0_onnxrt',
'vdet-12-106-0_onnxrt',
'vdet-12-107-0_onnxrt',
'vdet-12-400-0_tflitert',
'vdet-12-401-0_tflitert',
'vdet-12-402-0_tflitert',
'vdet-12-410-0_tflitert',
'vdet-12-412-0_tflitert',
'vdet-12-413-0_tflitert',
'vdet-12-414-0_tflitert',
'vdet-12-420-0_tflitert',
'vkpdet-25-101-0_onnxrt',
'vkpdet-25-102-0_onnxrt',
'vkpdet-25-103-0_onnxrt',
'vkpdet-25-104-0_onnxrt',
'vseg-17-400-0_tflitert',
'vseg-18-010-0_tflitert',
'vseg-18-100-0_onnxrt',
'vseg-18-101-0_onnxrt',
'vseg-18-102-0_onnxrt',
'vseg-18-103-0_onnxrt',
'vseg-18-110-0_onnxrt',
'vseg-18-111-0_onnxrt',
'vseg-19-400-0_tflitert',
'vseg-19-401-0_tflitert',
'vseg-21-100-0_onnxrt',
'vseg-21-106-0_onnxrt',
'vseg-21-110-0_onnxrt',
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
        df = get_missing_models(report_file='./work_dirs/modelartifacts/report_20210727-235437.csv',
            selected_models_list=selected_models_list)