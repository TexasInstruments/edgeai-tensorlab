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
#ver:12-2021-03-29
model_id_artifacts_pair = {
    # TFLite CL
    'vcls-10-010-0_tflitert': 'TFL-CL-000-mobileNetV1-mlperf',
    'vcls-10-401-0_tflitert': 'TFL-CL-001-mobileNetV2',
    'vcls-10-403-0_tflitert': 'TFL-CL-002-SqueezeNet',
    'vcls-10-405-8_tflitert': 'TFL-CL-003-InceptionNetV1',
    'vcls-10-406-0_tflitert': 'TFL-CL-004-InceptionNetV3',
    'vcls-10-409-0_tflitert': 'TFL-CL-005-resNet50V1',
    'vcls-10-410-0_tflitert': 'TFL-CL-006-resNet50V2',
    'vcls-10-407-0_tflitert': 'TFL-CL-007-mnasNet',
    'vcls-10-011-0_tflitert': 'TFL-CL-008-mobileNet-edgeTPU-mlperf',
    'vcls-10-440-0_tflitert': 'TFL-CL-009-efficientNet-edgeTPU-s',
    'vcls-10-441-0_tflitert': 'TFL-CL-010-efficientNet-edgeTPU-m',
    'vcls-10-430-0_tflitert': 'TFL-CL-013-efficientNet-lite0',
    'vcls-10-434-0_tflitert': 'TFL-CL-014-efficientNet-lite4',
    'vcls-10-404-0_tflitert': 'TFL-CL-015-denseNet',
    'vcls-10-012-0_tflitert': 'TFL-CL-016-resNet50V1p5-mlperf',
    'vcls-10-431-0_tflitert': 'TFL-CL-017-efficientNet-lite1',
    'vcls-10-432-0_tflitert': 'TFL-CL-018-efficientNet-lite2',
    'vcls-10-442-0_tflitert': 'TFL-CL-019-efficientNet-edgeTPU-l',
    'vcls-10-402-0_tflitert': 'TFL-CL-020-mobileNetV2-1p4',
    'vcls-10-400-0_tflitert': 'TFL-CL-021-mobileNetV1', 
    'vcls-10-400-8_tflitert': 'TFL-CL-022-mobileNetV1-qat', 
    'vcls-10-401-8_tflitert': 'TFL-CL-023-mobileNetV2-qat', 
    'vcls-10-408-0_tflitert': 'TFL-CL-024-nasNet-mobile-tflite',
    'vcls-10-450-0_tflitert': 'TFL-CL-025-xceptionNet-tflite', # mxnet model replaced with with tflite model now

    # TFLite OD
    'vdet-12-010-0_tflitert': 'TFL-OD-200-ssd-mobV1-coco-mlperf-300x300',
    'vdet-12-011-0_tflitert': 'TFL-OD-201-ssd-mobV2-coco-mlperf-300x300', 
    'vdet-12-400-0_tflitert': 'TFL-OD-202-ssdLite-mobDet-DSP-coco-320x320',
    'vdet-12-401-0_tflitert': 'TFL-OD-203-ssdLite-mobDet-EdgeTPU-coco-320x320',
    'vdet-12-404-0_tflitert': 'TFL-OD-204-ssd-mobV1-FPN-coco-640x640',
    'vdet-12-403-0_tflitert': 'TFL-OD-205-ssd-mobV2-mnas-fpn-coco-320x320',
    'vdet-12-402-0_tflitert': 'TFL-OD-206-ssd-mobV2-coco-mlperf-300x300',

    # TFLite SS
    'vseg-17-010-0_tflitert': 'TFL-SS-250-deeplab-mobV2-ade20k-512x512',
    'vseg-17-400-0_tflitert': 'TFL-SS-254-deeplabv3-mobv2-ade20k-512x512',
    'vseg-16-400-0_tflitert': 'TFL-SS-255-deeplabv3-mobv2_cs-2048x1024',
    'vseg-18-010-0_tflitert': 'TFL-SS-258-deeplabv3_mobv2-ade20k32-mlperf-512x512',
    'vseg-19-400-0_tflitert': 'TFL-SS-259-deeplabv3_mobv2-dm05-pascal-trainaug-512x512',
    'vseg-19-401-0_tflitert': 'TFL-SS-260-deeplabv3_mobv2-pascal-trainaug-512x512',

    # TVM- CL
    'vcls-10-020-0_tvmdlr': 'TVM-CL-300-resNet18V2',
    'vcls-10-450-0_tvmdlr': 'TVM-CL-302-xceptionNet-mxnet',
    'vcls-10-408-0_tvmdlr': 'TVM-CL-304-nasNet-mobile-tflite',
    'vcls-10-100-0_tvmdlr': 'TVM-CL-306-mobileNetV1',
    'vcls-10-101-0_tvmdlr': 'TVM-CL-307-mobileNetV2',
    'vcls-10-301-0_tvmdlr': 'TVM-CL-308-shuffleNetV2',
    'vcls-10-302-0_tvmdlr': 'TVM-CL-309-mobileNetV2-tv',
    'vcls-10-304-0_tvmdlr': 'TVM-CL-310-resNet18',
    'vcls-10-305-0_tvmdlr': 'TVM-CL-311-resNet50',
    'vcls-10-031-0_tvmdlr': 'TVM-CL-312-regNetX-400mf',
    'vcls-10-032-0_tvmdlr': 'TVM-CL-313-regNetX-800mf',
    'vcls-10-033-0_tvmdlr': 'TVM-CL-314-regNetX-1.6gf',
    'vcls-10-102-8_tvmdlr': 'TVM-CL-315-mobileNetV2-1p4-qat',
    
    #512x512
    'vcls-10-100-1_tvmdlr': 'TVM-CL-316-mobileNetV1-512x512',
    'vcls-10-101-1_tvmdlr': 'TVM-CL-317-mobileNetV2-512x512',
    'vcls-10-301-1_tvmdlr': 'TVM-CL-318-shuffleNetV2-512x512',
    'vcls-10-302-1_tvmdlr': 'TVM-CL-319-mobileNetV2-tv-512x512',
    'vcls-10-304-1_tvmdlr': 'TVM-CL-320-resNet18-512x512',
    'vcls-10-305-1_tvmdlr': 'TVM-CL-321-resNet50-512x512',
    'vcls-10-031-1_tvmdlr': 'TVM-CL-322-regNetX-400mf-512x512',
    'vcls-10-032-1_tvmdlr': 'TVM-CL-323-regNetX-800mf-512x512',
    'vcls-10-033-1_tvmdlr': 'TVM-CL-324-regNetX-1.6gf-512x512',
    'vcls-10-102-1_tvmdlr': 'TVM-CL-325-mobileNetV2-1p4-qat-512x512',

    #1024x1024
    'vcls-10-100-2_tvmdlr': 'TVM-CL-326-mobileNetV1-1024x1024',
    'vcls-10-101-2_tvmdlr': 'TVM-CL-327-mobileNetV2-1024x1024',
    'vcls-10-301-2_tvmdlr': 'TVM-CL-328-shuffleNetV2-1024x1024',
    'vcls-10-302-2_tvmdlr': 'TVM-CL-329-mobileNetV2-tv-1024x1024',
    'vcls-10-304-2_tvmdlr': 'TVM-CL-330-resNet18-1024x1024',
    'vcls-10-305-2_tvmdlr': 'TVM-CL-331-resNet50-1024x1024',
    'vcls-10-031-2_tvmdlr': 'TVM-CL-332-regNetX-400mf-1024x1024',
    'vcls-10-032-2_tvmdlr': 'TVM-CL-333-regNetX-800mf-1024x1024',
    'vcls-10-033-2_tvmdlr': 'TVM-CL-334-regNetX-1.6gf-1024x1024',
    'vcls-10-102-2_tvmdlr': 'TVM-CL-335-mobileNetV2-1p4-qat-1024x1024',
    ########
    
    'vcls-10-030-0_tvmdlr': 'TVM-CL-336-regNetx-200mf',
    'vcls-10-306-0_tvmdlr': 'TVM-CL-337-vgg16',
    'vcls-10-101-8_tvmdlr': 'TVM-CL-338-mobileNetV2-qat',
    'vcls-10-302-8_tvmdlr': 'TVM-CL-340-mobileNetV2-tv-qat',
    'vcls-10-060-0_tvmdlr': 'TVM-CL-341-gluoncv-mxnet-mobv2',
    'vcls-10-061-0_tvmdlr': 'TVM-CL-342-gluoncv-mxnet-resNet50-v1',
    'vcls-10-062-0_tvmdlr': 'TVM-CL-343-gluoncv-xception',
    # HarD Net
    'vcls-10-900-0_tvmdlr': 'TVM-CL-344-harDNet68',
    'vcls-10-901-0_tvmdlr': 'TVM-CL-345-harDNet85',
    'vcls-10-902-0_tvmdlr': 'TVM-CL-346-harDNet68ds',
    'vcls-10-903-0_tvmdlr': 'TVM-CL-347-harDNet39ds',
    
    # TVM - OD
    'vdet-12-012-0_tvmdlr': 'TVM-OD-500-ssd1200-resNet34-mlperf-1200x1200',
    'vdet-12-020-0_tvmdlr': 'TVM-OD-501-yolov3-416x416',
    'vdet-12-060-0_tvmdlr': 'TVM-OD-502-yolov3-mobv1-gluon-mxnet-416x416',
    'vdet-12-061-0_tvmdlr': 'TVM-OD-503-ssd-mobv1-gluon-mxnet-512x512',
    # TVM - SS - CS
    'vseg-16-100-0_tvmdlr': 'TVM-SS-550-deeplabv3lite-mobv2-cs-768x384',
    'vseg-16-100-8_tvmdlr': 'TVM-SS-551-deeplabv3lite-mobv2-cs-qat-768x384',
    'vseg-16-101-0_tvmdlr': 'TVM-SS-552-fpnlite-aspp-mobv2-cs-768x384',
    'vseg-16-101-8_tvmdlr': 'TVM-SS-553-fpnlite-aspp-mobv2-cs-qat-768x384',
    'vseg-16-102-0_tvmdlr': 'TVM-SS-554-unetlite-aspp-mobv2-tv-cs-768x384',
    'vseg-16-102-8_tvmdlr': 'TVM-SS-555-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'vseg-16-103-0_tvmdlr': 'TVM-SS-556-fpnlite-aspp-regNetx800mf-cs-768x384',
    'vseg-16-104-0_tvmdlr': 'TVM-SS-557-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'vseg-16-105-0_tvmdlr': 'TVM-SS-558-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'vseg-16-300-0_tvmdlr': 'TVM-SS-559-deeplabv3-res50-1040x520',
    'vseg-16-301-0_tvmdlr': 'TVM-SS-560-fcn-res50-1040x520',

    # TVM - SS - ADE20k
    'vseg-18-100-0_tvmdlr': 'TVM-SS-561-deeplabv3lite-mobv2-ade20k32-512x512',
    'vseg-18-100-8_tvmdlr': 'TVM-SS-562-deeplabv3lite-mobv2-ade20k32-qat-512x512', 
    'vseg-18-101-0_tvmdlr': 'TVM-SS-563-unetlite-aspp-mobv2-tv-ade20k32-512x512',
    'vseg-18-101-8_tvmdlr': 'TVM-SS-564-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512',
    'vseg-18-102-0_tvmdlr': 'TVM-SS-565-fpnlite-aspp-mobv2-ade20k32-512x512',
    'vseg-18-102-8_tvmdlr': 'TVM-SS-566-fpnlite-aspp-mobv2-ade20k32-qat-512x512', 
    'vseg-18-103-0_tvmdlr': 'TVM-SS-567-fpnlite-aspp-mobv2-1p4-ade20k32-512x512',
    'vseg-18-103-8_tvmdlr': 'TVM-SS-568-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512',
    'vseg-18-110-0_tvmdlr': 'TVM-SS-569-fpnlite-aspp-regnetx400mf-ade20k32-384x384',
    'vseg-18-111-0_tvmdlr': 'TVM-SS-570-fpnlite-aspp-regnetx800mf-ade20k32-512x512',


    # ONNXRT- CL
    'vcls-10-020-0_onnxrt': 'ONR-CL-600-resNet18V2',
    'vcls-10-450-0_onnxrt': 'ONR-CL-602-xceptionNet-mxnet',
    'vcls-10-408-0_onnxrt': 'ONR-CL-604-nasNet-mobile-tflite',
    'vcls-10-100-0_onnxrt': 'ONR-CL-606-mobileNetV1',
    'vcls-10-101-0_onnxrt': 'ONR-CL-607-mobileNetV2',
    'vcls-10-301-0_onnxrt': 'ONR-CL-608-shuffleNetV2',
    'vcls-10-302-0_onnxrt': 'ONR-CL-609-mobileNetV2-tv',
    'vcls-10-304-0_onnxrt': 'ONR-CL-610-resNet18',
    'vcls-10-305-0_onnxrt': 'ONR-CL-611-resNet50',
    'vcls-10-031-0_onnxrt': 'ONR-CL-612-regNetX-400mf',
    'vcls-10-032-0_onnxrt': 'ONR-CL-613-regNetX-800mf',
    'vcls-10-033-0_onnxrt': 'ONR-CL-614-regNetX-1.6gf',
    'vcls-10-102-8_onnxrt': 'ONR-CL-615-mobileNetV2-1p4-qat',
    
    #512x512
    'vcls-10-100-1_onnxrt': 'ONR-CL-616-mobileNetV1-512x512',
    'vcls-10-101-1_onnxrt': 'ONR-CL-617-mobileNetV2-512x512',
    'vcls-10-301-1_onnxrt': 'ONR-CL-618-shuffleNetV2-512x512',
    'vcls-10-302-1_onnxrt': 'ONR-CL-619-mobileNetV2-tv-512x512',
    'vcls-10-304-1_onnxrt': 'ONR-CL-620-resNet18-512x512',
    'vcls-10-305-1_onnxrt': 'ONR-CL-621-resNet50-512x512',
    'vcls-10-031-1_onnxrt': 'ONR-CL-622-regNetX-400mf-512x512',
    'vcls-10-032-1_onnxrt': 'ONR-CL-623-regNetX-800mf-512x512',
    'vcls-10-033-1_onnxrt': 'ONR-CL-624-regNetX-1.6gf-512x512',
    'vcls-10-102-1_onnxrt': 'ONR-CL-625-mobileNetV2-1p4-qat-512x512',

    #1024x1024
    'vcls-10-100-2_onnxrt': 'ONR-CL-626-mobileNetV1-1024x1024',
    'vcls-10-101-2_onnxrt': 'ONR-CL-627-mobileNetV2-1024x1024',
    'vcls-10-301-2_onnxrt': 'ONR-CL-628-shuffleNetV2-1024x1024',
    'vcls-10-302-2_onnxrt': 'ONR-CL-629-mobileNetV2-tv-1024x1024',
    'vcls-10-304-2_onnxrt': 'ONR-CL-630-resNet18-1024x1024',
    'vcls-10-305-2_onnxrt': 'ONR-CL-631-resNet50-1024x1024',
    'vcls-10-031-2_onnxrt': 'ONR-CL-632-regNetX-400mf-1024x1024',
    'vcls-10-032-2_onnxrt': 'ONR-CL-633-regNetX-800mf-1024x1024',
    'vcls-10-033-2_onnxrt': 'ONR-CL-634-regNetX-1.6gf-1024x1024',
    'vcls-10-102-2_onnxrt': 'ONR-CL-635-mobileNetV2-1p4-qat-1024x1024',
    ########
    
    'vcls-10-030-0_onnxrt': 'ONR-CL-636-regNetx-200mf',
    'vcls-10-306-0_onnxrt': 'ONR-CL-637-vgg16',
    'vcls-10-101-8_onnxrt': 'ONR-CL-638-mobileNetV2-qat',
    'vcls-10-302-8_onnxrt': 'ONR-CL-640-mobileNetV2-tv-qat',
    'vcls-10-060-0_onnxrt': 'ONR-CL-641-gluoncv-mxnet-mobv2',
    'vcls-10-061-0_onnxrt': 'ONR-CL-642-gluoncv-mxnet-resNet50-v1',
    'vcls-10-062-0_onnxrt': 'ONR-CL-643-gluoncv-xception',
    # HarD Net
    'vcls-10-900-0_onnxrt': 'ONR-CL-644-harDNet68',
    'vcls-10-901-0_onnxrt': 'ONR-CL-645-harDNet85',
    'vcls-10-902-0_onnxrt': 'ONR-CL-646-harDNet68ds',
    'vcls-10-903-0_onnxrt': 'ONR-CL-647-harDNet39ds',
    
    # ONNX - OD
    'vdet-12-012-0_onnxrt': 'ONR-OD-800-ssd1200-resNet34-mlperf-1200x1200',
    'vdet-12-020-0_onnxrt': 'ONR-OD-801-yolov3-416x416',
    'vdet-12-060-0_onnxrt': 'ONR-OD-802-yolov3-mobv1-gluon-mxnet-416x416',
    'vdet-12-061-0_onnxrt': 'ONR-OD-803-ssd-mobv1-gluon-mxnet-512x512',
    # ONNX - SS - CS
    'vseg-16-100-0_onnxrt': 'ONR-SS-850-deeplabv3lite-mobv2-cs-768x384',
    'vseg-16-100-8_onnxrt': 'ONR-SS-851-deeplabv3lite-mobv2-cs-qat-768x384',
    'vseg-16-101-0_onnxrt': 'ONR-SS-852-fpnlite-aspp-mobv2-cs-768x384',
    'vseg-16-101-8_onnxrt': 'ONR-SS-853-fpnlite-aspp-mobv2-cs-qat-768x384',
    'vseg-16-102-0_onnxrt': 'ONR-SS-854-unetlite-aspp-mobv2-tv-cs-768x384',
    'vseg-16-102-8_onnxrt': 'ONR-SS-855-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'vseg-16-103-0_onnxrt': 'ONR-SS-856-fpnlite-aspp-regNetx800mf-cs-768x384',
    'vseg-16-104-0_onnxrt': 'ONR-SS-857-fpnlite-aspp-regNetx1.6gf-cs-1024x512',
    'vseg-16-105-0_onnxrt': 'ONR-SS-858-fpnlite-aspp-regNetx3.2gf-cs-1536x768',
    'vseg-16-300-0_onnxrt': 'ONR-SS-859-deeplabv3-res50-1040x520',
    'vseg-16-301-0_onnxrt': 'ONR-SS-860-fcn-res50-1040x520',

    # ONNX - SS - ADE20k
    'vseg-18-100-0_onnxrt': 'ONR-SS-861-deeplabv3lite-mobv2-ade20k32-512x512',
    'vseg-18-100-8_onnxrt': 'ONR-SS-862-deeplabv3lite-mobv2-ade20k32-qat-512x512', 
    'vseg-18-101-0_onnxrt': 'ONR-SS-863-unetlite-aspp-mobv2-tv-ade20k32-512x512',
    'vseg-18-101-8_onnxrt': 'ONR-SS-864-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512',
    'vseg-18-102-0_onnxrt': 'ONR-SS-865-fpnlite-aspp-mobv2-ade20k32-512x512',
    'vseg-18-102-8_onnxrt': 'ONR-SS-866-fpnlite-aspp-mobv2-ade20k32-qat-512x512', 
    'vseg-18-103-0_onnxrt': 'ONR-SS-867-fpnlite-aspp-mobv2-1p4-ade20k32-512x512',
    'vseg-18-103-8_onnxrt': 'ONR-SS-868-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512',
    'vseg-18-110-0_onnxrt': 'ONR-SS-869-fpnlite-aspp-regnetx400mf-ade20k32-384x384',
    'vseg-18-111-0_onnxrt': 'ONR-SS-870-fpnlite-aspp-regnetx800mf-ade20k32-512x512',
}

removed_model_list = {
    'vcls-10-450-0_tvmdlr' : 'TVM-CL-302-xceptionNet-mxnet', # this is replaced with tflite model now (that was also eventually removed)
    'vcls-10-401-8_tflitert': 'TFL-CL-023-mobileNetV2-qat',  # QAT model is not giving good accuracy so keep only float
    'vdet-12-012-0_tvmdlr': 'TVM-OD-500-ssd1200-resNet34-1200x1200-mlperf', # Not working with TVM. Will need to park it till ONNX RT OD support is available.
    ################ CS models
    'vseg-16-100-0_tvmdlr': 'TVM-SS-550-deeplabv3lite-mobv2-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-100-8_tvmdlr': 'TVM-SS-551-deeplabv3lite-mobv2-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-101-0_tvmdlr': 'TVM-SS-552-fpnlite-aspp-mobv2-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-101-8_tvmdlr': 'TVM-SS-553-fpnlite-aspp-mobv2-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-102-0_tvmdlr': 'TVM-SS-554-unetlite-aspp-mobv2-tv-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-102-8_tvmdlr': 'TVM-SS-555-unetlite-aspp-mobv2-tv-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-103-0_tvmdlr': 'TVM-SS-556-fpnlite-aspp-regNetx800mf-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-104-0_tvmdlr': 'TVM-SS-557-fpnlite-aspp-regNetx1.6gf-cs-1024x512', # cityscapes model not part of Model Zoo
    'vseg-16-105-0_tvmdlr': 'TVM-SS-558-fpnlite-aspp-regNetx3.2gf-cs-1536x768', # cityscapes model not part of Model Zoo

    'vseg-16-100-0_onnxrt': 'ONR-SS-850-deeplabv3lite-mobv2-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-100-8_onnxrt': 'ONR-SS-851-deeplabv3lite-mobv2-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-101-0_onnxrt': 'ONR-SS-852-fpnlite-aspp-mobv2-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-101-8_onnxrt': 'ONR-SS-853-fpnlite-aspp-mobv2-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-102-0_onnxrt': 'ONR-SS-854-unetlite-aspp-mobv2-tv-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-102-8_onnxrt': 'ONR-SS-855-unetlite-aspp-mobv2-tv-cs-qat-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-103-0_onnxrt': 'ONR-SS-856-fpnlite-aspp-regNetx800mf-cs-768x384', # cityscapes model not part of Model Zoo
    'vseg-16-104-0_onnxrt': 'ONR-SS-857-fpnlite-aspp-regNetx1.6gf-cs-1024x512', # cityscapes model not part of Model Zoo
    'vseg-16-105-0_onnxrt': 'ONR-SS-858-fpnlite-aspp-regNetx3.2gf-cs-1536x768', # cityscapes model not part of Model Zoo
    ################
    'vcls-10-408-0_tflitert': 'TFL-CL-024-nasNet-mobile-tflite', # Kumar removed model (Multiple sub-graphs)
    'vcls-10-432-0_tflitert': 'TFL-CL-018-efficientNet-lite2', # Kumar removed model
    'vdet-12-011-0_tflitert': 'TFL-OD-201-ssd-mobV2-coco-300x300-mlperf',  # Kumar removed model
    'vseg-16-400-0_tflitert': 'TFL-SS-255-deeplabv3-mobv2_cs-2048x1024',  # Kumar removed model, 

    'vcls-10-306-0_tvmdlr': 'TVM-CL-337-vgg16', # Kumar removed model
    'vcls-10-020-0_tvmdlr': 'TVM-CL-300-resNet18V2', # Kumar removed model
    'vseg-16-300-0_tvmdlr': 'TVM-SS-559-deeplabv3-res50-1040x520', # Kumar removed model, nc does not have info for this
    'vseg-16-301-0_tvmdlr': 'TVM-SS-560-fcn-res50-1040x520', # Kumar removed model

    'vcls-10-306-0_onnxrt': 'ONR-CL-637-vgg16', # Kumar removed model
    'vcls-10-020-0_onnxrt': 'ONR-CL-600-resNet18V2', # Kumar removed model
    'vseg-16-300-0_onnxrt': 'ONR-SS-859-deeplabv3-res50-1040x520', # Kumar removed model, nc does not have info for this
    'vseg-16-301-0_onnxrt': 'ONR-SS-860-fcn-res50-1040x520', # Kumar removed model
    
    #########################
    'vseg-17-010-0_tflitert': 'TFL-SS-250-deeplab-mobV2-ade20k-512x512', # Manu said incorrect model ID removed. vseg-17-010 is replaced with vseg-18-010
    'vcls-10-450-0_tflitert': 'TFL-CL-025-xceptionNet-tflite', # mxnet model replaced with with tflite model now. Eventually removed as size is quite big.
    'vdet-12-404-0_tflitert': 'TFL-OD-204-ssd-mobV1-FPN-coco-640x640', # does not run, import crashes. Manu
    'vdet-12-403-0_tflitert': 'TFL-OD-205-ssd-mobV2-mnas-fpn-coco-320x320', # does not run, import crashes. Manu

    'vcls-10-408-0_tvmdlr': 'TVM-CL-304-nasNet-mobile-tflite', # not part of benchmarking script yet. tflite model with TVM.
    'vdet-12-020-0_tvmdlr': 'TVM-OD-501-yolov3-416x416', # not supported yet

    'vcls-10-408-0_onnxrt': 'ONR-CL-604-nasNet-mobile-tflite', # not part of benchmarking script yet. tflite model with TVM.
    'vdet-12-020-0_onnxrt': 'ONR-OD-801-yolov3-416x416', # not supported yet

    #ADE20k32 models
    'vseg-18-100-8_tvmdlr': 'TVM-SS-562-deeplabv3lite-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-18-101-8_tvmdlr': 'TVM-SS-564-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512', # import fails
    'vseg-18-102-8_tvmdlr': 'TVM-SS-566-fpnlite-aspp-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-18-103-8_tvmdlr': 'TVM-SS-568-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed

    'vseg-18-100-8_onnxrt': 'ONR-SS-862-deeplabv3lite-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-18-101-8_onnxrt': 'ONR-SS-864-unetlite-aspp-mobv2-tv-ade20k32-qat-512x512', # import fails
    'vseg-18-102-8_onnxrt': 'ONR-SS-866-fpnlite-aspp-mobv2-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed
    'vseg-18-103-8_onnxrt': 'ONR-SS-868-fpnlite-aspp-mobv2-1p4-ade20k32-qat-512x512', # PTQ itself is good,  QAT not needed

    #512x512 (Only for performance)
    'vcls-10-100-1_tvmdlr': 'TVM-CL-316-mobileNetV1-512x512',
    'vcls-10-101-1_tvmdlr': 'TVM-CL-317-mobileNetV2-512x512',
    'vcls-10-301-1_tvmdlr': 'TVM-CL-318-shuffleNetV2-512x512',
    'vcls-10-302-1_tvmdlr': 'TVM-CL-319-mobileNetV2-tv-512x512',
    'vcls-10-304-1_tvmdlr': 'TVM-CL-320-resNet18-512x512',
    'vcls-10-305-1_tvmdlr': 'TVM-CL-321-resNet50-512x512',
    'vcls-10-031-1_tvmdlr': 'TVM-CL-322-regNetX-400mf-512x512',
    'vcls-10-032-1_tvmdlr': 'TVM-CL-323-regNetX-800mf-512x512',
    'vcls-10-033-1_tvmdlr': 'TVM-CL-324-regNetX-1.6gf-512x512',
    'vcls-10-102-1_tvmdlr': 'TVM-CL-325-mobileNetV2-1p4-qat-512x512',
    
    'vcls-10-100-1_onnxrt': 'ONR-CL-616-mobileNetV1-512x512',
    'vcls-10-101-1_onnxrt': 'ONR-CL-617-mobileNetV2-512x512',
    'vcls-10-301-1_onnxrt': 'ONR-CL-618-shuffleNetV2-512x512',
    'vcls-10-302-1_onnxrt': 'ONR-CL-619-mobileNetV2-tv-512x512',
    'vcls-10-304-1_onnxrt': 'ONR-CL-620-resNet18-512x512',
    'vcls-10-305-1_onnxrt': 'ONR-CL-621-resNet50-512x512',
    'vcls-10-031-1_onnxrt': 'ONR-CL-622-regNetX-400mf-512x512',
    'vcls-10-032-1_onnxrt': 'ONR-CL-623-regNetX-800mf-512x512',
    'vcls-10-033-1_onnxrt': 'ONR-CL-624-regNetX-1.6gf-512x512',
    'vcls-10-102-1_onnxrt': 'ONR-CL-625-mobileNetV2-1p4-qat-512x512',


    #1024x1024  (Only for performance)
    'vcls-10-100-2_tvmdlr': 'TVM-CL-326-mobileNetV1-1024x1024',
    'vcls-10-101-2_tvmdlr': 'TVM-CL-327-mobileNetV2-1024x1024',
    'vcls-10-301-2_tvmdlr': 'TVM-CL-328-shuffleNetV2-1024x1024',
    'vcls-10-302-2_tvmdlr': 'TVM-CL-329-mobileNetV2-tv-1024x1024',
    'vcls-10-304-2_tvmdlr': 'TVM-CL-330-resNet18-1024x1024',
    'vcls-10-305-2_tvmdlr': 'TVM-CL-331-resNet50-1024x1024',
    'vcls-10-031-2_tvmdlr': 'TVM-CL-332-regNetX-400mf-1024x1024',
    'vcls-10-032-2_tvmdlr': 'TVM-CL-333-regNetX-800mf-1024x1024',
    'vcls-10-033-2_tvmdlr': 'TVM-CL-334-regNetX-1.6gf-1024x1024',
    'vcls-10-102-2_tvmdlr': 'TVM-CL-335-mobileNetV2-1p4-qat-1024x1024',

    'vcls-10-100-2_onnxrt': 'ONR-CL-626-mobileNetV1-1024x1024',
    'vcls-10-101-2_onnxrt': 'ONR-CL-627-mobileNetV2-1024x1024',
    'vcls-10-301-2_onnxrt': 'ONR-CL-628-shuffleNetV2-1024x1024',
    'vcls-10-302-2_onnxrt': 'ONR-CL-629-mobileNetV2-tv-1024x1024',
    'vcls-10-304-2_onnxrt': 'ONR-CL-630-resNet18-1024x1024',
    'vcls-10-305-2_onnxrt': 'ONR-CL-631-resNet50-1024x1024',
    'vcls-10-031-2_onnxrt': 'ONR-CL-632-regNetX-400mf-1024x1024',
    'vcls-10-032-2_onnxrt': 'ONR-CL-633-regNetX-800mf-1024x1024',
    'vcls-10-033-2_onnxrt': 'ONR-CL-634-regNetX-1.6gf-1024x1024',
    'vcls-10-102-2_onnxrt': 'ONR-CL-635-mobileNetV2-1p4-qat-1024x1024',
}

removed_models_from_plots = {
    'vseg-19-400-0_tflitert': 'TFL-SS-259-deeplabv3_mobv2-dm05-pascal-trainaug-512x512', # SS on pascal has only 2 models. So removed from plots.
    'vseg-19-401-0_tflitert': 'TFL-SS-260-deeplabv3_mobv2-pascal-trainaug-512x512', # # SS on pascal has only 2 models.  So removed from plots.
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
    import pandas as pd
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


def get_artifact_name(artifact_id):
    artifact_name = None
    if artifact_id in model_id_artifacts_pair:
        artifact_name = model_id_artifacts_pair[artifact_id]
    else:
        model_id, runtime_name = artifact_id.split('_')
        # create mapping dictionaries
        model_id_to_model_name_dict = {k.split('_')[0]:'-'.join(v.split('-')[1:]) \
                for k,v in model_id_artifacts_pair.items()}
        short_runtime_name_dict = {'tvmdlr':'TVM', 'tflitert':'TFL', 'onnxrt':'ONR'}
        # finally for the artifact name
        if runtime_name in short_runtime_name_dict and model_id in model_id_to_model_name_dict:
            artifact_name = f'{short_runtime_name_dict[runtime_name]}-{model_id_to_model_name_dict[model_id]}'
        
    return artifact_name


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
        df = get_missing_models(report_file='./work_dirs/benchmark_accuracy/report_20210401-102803.csv', 
            selected_models_list=selected_models_list)