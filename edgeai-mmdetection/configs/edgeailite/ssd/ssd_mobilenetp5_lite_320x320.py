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

######################################################
input_size = (320,320)                          #(512,512) #(768,768) #(1024,1024)
# input_size = (416, 416)
dataset_type = 'CocoDataset'
num_classes_dict = {'CocoDataset':80, 'VOCDataset':20, 'CityscapesDataset':8, 'WIDERFaceDataset':1}
num_classes = num_classes_dict[dataset_type]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #imagenet mean/std

_base_ = [
    f'../_xbase_/datasets/{dataset_type.lower()}.py',
    '../_xbase_/hyper_params/common_config.py',
    '../_xbase_/hyper_params/ssd_config.py',
    '../_xbase_/hyper_params/common_schedule.py',
]

######################################################
# settings for qat or calibration - uncomment after doing floating point training
# also change dataset_repeats in the dataset config to 1 for fast learning
quantize = False #'training' #'calibration'
initial_learning_rate = 4e-2 #8e-2
samples_per_gpu = 16
if quantize:
  load_from = './work_dirs/ssd_mobilenet_lite/latest.pth'
  optimizer = dict(type='SGD', lr=initial_learning_rate/100.0, momentum=0.9, weight_decay=4e-5) #1e-4 => 4e-5
  total_epochs = 1 if quantize == 'calibration' else 12
else:
  optimizer = dict(type='SGD', lr=initial_learning_rate, momentum=0.9, weight_decay=4e-5) #1e-4 => 4e-5
#

######################################################
backbone_type = 'MobileNetV2P5Lite' #'MobileNetV2Lite' #'MobileNetV1Lite'
mobilenetv2_pretrained = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/edgeai-tv/mobilenet_v2p5_20230201_checkpoint.pth'
mobilenetv1_pretrained='https://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/edgeai-tv/mobilenet_v1_20190906_checkpoint.pth'
pretrained=(mobilenetv2_pretrained if backbone_type == 'MobileNetV2P5Lite' else mobilenetv1_pretrained)

backbone_out_channels=(48, 160, 256, 128, 128, 128) if backbone_type == 'MobileNetV2P5Lite' else (512, 1024, 512, 256, 256, 256)
backbone_out_indices = (1, 2, 3, 4)
basesize_ratio_range = (0.1, 0.9)

conv_cfg = None
norm_cfg = dict(type='BN')
convert_to_lite_model = dict(group_size_dw=1)

model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type=backbone_type,
        strides=(2, 2, 2, 2, 2),
        depth=None,
        with_last_pool=False,
        ceil_mode=True,
        extra_channels=backbone_out_channels[2:],
        out_indices=backbone_out_indices[-2:],
        out_feature_indices=None,
        l2_norm_scale=None,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        in_channels=backbone_out_channels,
        num_classes=num_classes,
        conv_cfg=conv_cfg,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=basesize_ratio_range,
            strides=[16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18) if not quantize else dict(type='Bypass'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=input_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=input_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=0,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))


