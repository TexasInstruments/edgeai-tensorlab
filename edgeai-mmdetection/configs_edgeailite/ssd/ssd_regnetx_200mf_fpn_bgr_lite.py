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
_base_ = [
    '../../configs/_base_/datasets/coco_detection.py',
    '../../configs/_base_/schedules/schedule_1x.py', '../../configs/_base_/default_runtime.py'
]

input_size = 320 
num_classes = 80
img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False) #imagenet mean used in pycls (bgr)
#

######################################################
# settings for qat or calibration - uncomment after doing floating point training
# also change dataset_repeats in the dataset config to 1 for fast learning
quantize = False #'training' #'calibration'
initial_learning_rate = 8e-2
samples_per_gpu = 16
if quantize:
  load_from = './work_dirs/ssd_regnet_fpn_bgr_lite/latest.pth'
  optimizer = dict(type='SGD', lr=initial_learning_rate/100.0, momentum=0.9, weight_decay=4e-5) #1e-4 => 4e-5
  total_epochs = 1 if quantize == 'calibration' else 12
else:
  optimizer = dict(type='SGD', lr=initial_learning_rate, momentum=0.9, weight_decay=4e-5) #1e-4 => 4e-5
#
  
#######################################################
backbone_type = 'RegNet'
backbone_arch = 'regnetx_200mf'                  # 'regnetx_200mf' #'regnetx_800mf' #'regnetx_1.6gf' #'regnetx_3.2gf'
to_rgb = False                                   # pycls regnet backbones are trained with bgr

decoder_fpn_type = 'FPN'
fpn_width_fact = 4
decoder_width_fact = 4
decoder_depth_fact = 4

regnet_settings = {
    'regnetx_200mf': {'backbone_out_channels': [32, 56, 152, 368], 'group_size_dw': 8,
                      'fpn_intermediate_channels': min(28*fpn_width_fact,256),
                      'fpn_out_channels': min(28*decoder_width_fact,256),
                      'fpn_num_blocks': decoder_depth_fact,
                      # TODO: the following pretrained checkpoint is not correct - it is just a placeholder.
                      # TODO: run tools/model_converters/regnet2mmdet.py on pycls checkpoint and provide the output checkpoint here.
                      'pretrained': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905981/RegNetX-200MF_dds_8gpu.pyth'},
    'regnetx_400mf': {'backbone_out_channels': [32, 64, 160, 384], 'group_size_dw': 16,
                      'fpn_intermediate_channels': min(32*fpn_width_fact,256),
                      'fpn_out_channels': min(32*decoder_width_fact,256),
                      'fpn_num_blocks': decoder_depth_fact,
                      'pretrained': 'open-mmlab://regnetx_400mf'},
    'regnetx_800mf':{'backbone_out_channels':[64, 128, 288, 672], 'group_size_dw':16,
                     'fpn_intermediate_channels':min(64*fpn_width_fact,256),
                     'fpn_out_channels':min(64*decoder_width_fact,256),
                     'fpn_num_blocks':decoder_depth_fact,
                     'pretrained':'open-mmlab://regnetx_800mf'},
    'regnetx_1.6gf':{'backbone_out_channels':[72, 168, 408, 912], 'group_size_dw':24,
                     'fpn_intermediate_channels':min(84*fpn_width_fact,264),
                     'fpn_out_channels':min(84*decoder_width_fact,264),
                     'fpn_num_blocks':decoder_depth_fact,
                     'pretrained':'open-mmlab://regnetx_1.6gf'},
    'regnetx_3.2gf':{'backbone_out_channels':[96, 192, 432, 1008], 'group_size_dw':48,
                     'fpn_intermediate_channels':min(96*fpn_width_fact,288),
                     'fpn_out_channels':min(96*decoder_width_fact,288),
                     'fpn_num_blocks':decoder_depth_fact,
                     'pretrained': 'open-mmlab://regnetx_3.2gf'}
}

######################################################
regnet_cfg = regnet_settings[backbone_arch]
pretrained=regnet_cfg['pretrained']
backbone_out_channels=regnet_cfg['backbone_out_channels']
backbone_out_indices = (0, 1, 2, 3)

fpn_in_channels = backbone_out_channels[-len(backbone_out_indices):]
fpn_out_channels = regnet_cfg['fpn_out_channels']
fpn_start_level = 1
fpn_num_outs = 6
fpn_upsample_mode = 'bilinear' #'nearest' #'bilinear'
fpn_upsample_cfg = dict(scale_factor=2, mode=fpn_upsample_mode)
fpn_num_blocks = regnet_cfg['fpn_num_blocks']
fpn_intermediate_channels = regnet_cfg['fpn_intermediate_channels']
fpn_add_extra_convs = 'on_input'

basesize_ratio_range = (0.1, 0.9)
input_size_divisor = 32

conv_cfg = None
norm_cfg = dict(type='BN')
convert_to_lite_model = dict(group_size_dw=regnet_cfg['group_size_dw'], model_surgery=1)

# model settings
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    pad_size_divisor=1)
model = dict(
    type='SingleStageDetector',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=backbone_type,
        arch=backbone_arch,
        out_indices=backbone_out_indices,
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type=decoder_fpn_type,
        in_channels=fpn_in_channels,
        out_channels=fpn_out_channels,
        start_level=fpn_start_level,
        num_outs=fpn_num_outs,
        add_extra_convs=fpn_add_extra_convs,
        upsample_cfg=fpn_upsample_cfg,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='SSDHead',
        in_channels=[fpn_out_channels for _ in range(fpn_num_outs)],
        num_classes=num_classes,
        conv_cfg=conv_cfg,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=basesize_ratio_range,
            strides=[8, 16, 32, 64, 128, 256],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
env_cfg = dict(cudnn_benchmark=True)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=data_preprocessor['mean'],
        to_rgb=data_preprocessor['bgr_to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='Pad', size_divisor=input_size_divisor),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=24,
    num_workers=4,
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type={{_base_.dataset_type}},
            data_root={{_base_.data_root}},
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)))
val_dataloader = dict(batch_size=8, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# training schedule
max_epochs = 120
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=max_epochs,
        end=max_epochs,
        by_epoch=True,
        eta_min=0)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=4.0e-5))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (24 samples per GPU)
auto_scale_lr = dict(base_batch_size=192)
