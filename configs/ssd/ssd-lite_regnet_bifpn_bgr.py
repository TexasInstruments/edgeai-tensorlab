######################################################
input_size = (512,512)                          #(512,512) #(768,768) #(1024,1024)
dataset_type = 'CocoDataset'
num_classes_dict = {'CocoDataset':80, 'VOCDataset':20, 'CityscapesDataset':8}
num_classes = num_classes_dict[dataset_type]
img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False) #imagenet mean used in pycls (bgr)

_base_ = [
    f'../_xbase_/datasets/{dataset_type.lower()}.py',
    '../_xbase_/hyper_params/common_config.py',
    '../_xbase_/hyper_params/ssd_config.py',
    '../_xbase_/hyper_params/schedule.py',
]

######################################################
# settings for qat or calibration - uncomment after doing floating point training
# also change dataset_repeats in the dataset config to 1 for fast learning
quantize = False #'training' #'calibration'
initial_learning_rate = 2e-2
if quantize:
  load_from = './work_dirs/ssd-lite_regnet_bifpn_bgr/latest.pth'
  optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=4e-5) #1e-4 => 4e-5
  total_epochs = 1 if quantize == 'calibration' else 6
else:
  optimizer = dict(type='SGD', lr=initial_learning_rate, momentum=0.9, weight_decay=4e-5) #1e-4 => 4e-5
#

######################################################
decoder_fpn_type = 'BiFPNLite'                  # 'FPNLite' #'BiFPNLite' #'FPN'
decoder_conv_type = 'ConvDWSep'                 # 'ConvDWSep' #'ConvDWTripletRes' #'ConvDWTripletAlwaysRes'
decoder_width_fact = (2 if decoder_fpn_type == 'BiFPNLite' else 4)
decoder_depth_fact = 4

backbone_type = 'RegNet'
backbone_arch = 'regnetx_800mf'                  # 'regnetx_800mf' #'regnetx_1.6gf'
to_rgb = False                                   # pycls regnet backbones are trained with bgr

regnet_settings = {
    'regnetx_800mf':{'bacbone_out_channels':[64, 128, 288, 672], 'group_size_dw':16,
                      'fpn_out_channels':min(64*decoder_width_fact,256),
                      'fpn_num_blocks':decoder_depth_fact, 'pretrained':'open-mmlab://regnetx_800mf'},
    'regnetx_1.6gf':{'bacbone_out_channels':[72, 168, 408, 912], 'group_size_dw':24,
                     'fpn_out_channels':min(96*decoder_width_fact,256),
                     'fpn_num_blocks':decoder_depth_fact, 'pretrained':'open-mmlab://regnetx_1.6gf'}}

regnet_cfg = regnet_settings[backbone_arch]
pretrained=regnet_cfg['pretrained']
bacbone_out_channels=regnet_cfg['bacbone_out_channels']
backbone_out_indices = (0, 1, 2, 3)

fpn_in_channels = bacbone_out_channels[-len(backbone_out_indices):]
fpn_out_channels = regnet_cfg['fpn_out_channels']
fpn_start_level = 1
fpn_num_outs = 6
fpn_upsample_mode = 'bilinear' #'nearest' #'bilinear'
fpn_upsample_cfg = dict(scale_factor=2, mode=fpn_upsample_mode)
fpn_num_blocks = regnet_cfg['fpn_num_blocks']
fpn_bifpn_cfg = dict(num_blocks=fpn_num_blocks) if decoder_fpn_type == 'BiFPNLite' else dict()

basesize_ratio_range = (0.1, 0.9)
input_size_divisor = 128 if decoder_fpn_type == 'BiFPNLite' else 32

conv_cfg = dict(type=decoder_conv_type, group_size_dw=regnet_cfg['group_size_dw']) #None
norm_cfg = dict(type='BN')

model = dict(
    type='SingleStageDetector',
    pretrained=pretrained,
    backbone=dict(
        type=backbone_type,
        arch=backbone_arch,
        out_indices=backbone_out_indices,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type=decoder_fpn_type,
        in_channels=fpn_in_channels,
        out_channels=fpn_out_channels,
        start_level=fpn_start_level,
        num_outs=fpn_num_outs,
        add_extra_convs='on_output',
        upsample_cfg=fpn_upsample_cfg,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        **fpn_bifpn_cfg),
    bbox_head=dict(
        type='SSDLiteHead',
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
    dict(type='Pad', size_divisor=input_size_divisor),
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
            dict(type='Pad', size_divisor=input_size_divisor),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))


