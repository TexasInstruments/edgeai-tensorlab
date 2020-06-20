_base_ = [
    '../_jacinto_ai_base_/hyper_params/common_config.py',
    '../_jacinto_ai_base_/hyper_params/ssd_config.py',
]

dataset_type = 'VOCDataset'

if dataset_type == 'VOCDataset':
    _base_ += ['../_jacinto_ai_base_/datasets/voc0712_det.py']
    num_classes = 20
elif dataset_type == 'CocoDataset':
    _base_ += ['../_jacinto_ai_base_/datasets/coco_det.py']
    num_classes = 80
elif dataset_type == 'CityscapesDataset':
    _base_ += ['../_jacinto_ai_base_/datasets/cityscapes_det.py']
    num_classes = 8
else:
    assert False, f'Unknown dataset_type: {dataset_type}'


input_size = (768,384)          #(1536,768) #(1024,512) #(768,384) #(512,512)

backbone_type = 'RegNet'
backbone_arch = 'regnetx_800mf' #'regnetx_800mf' #'regnetx_1.6gf'
to_rgb = False                  #pycls regnet backbones are trained with bgr

regnet_settings = {
    'regnetx_800mf':{'regnet_base_channels':32, 'bacbone_out_channels':[64, 128, 288, 672],
                      'group_size_dw':16, 'fpn_out_channels':256,
                      'pretrained':'open-mmlab://regnetx_800mf'},
    'regnetx_1.6gf':{'regnet_base_channels':32, 'bacbone_out_channels':[72, 168, 408, 912],
                     'group_size_dw':24, 'fpn_out_channels':264,
                     'pretrained':'open-mmlab://regnetx_1.6gf'}}

regnet_cfg = regnet_settings[backbone_arch]
pretrained=regnet_cfg['pretrained']
regnet_base_channels=regnet_cfg['regnet_base_channels']
bacbone_out_channels=regnet_cfg['bacbone_out_channels']
backbone_out_indices = (0, 1, 2, 3)

fpn_in_channels = bacbone_out_channels[-len(backbone_out_indices):]
fpn_out_channels = regnet_cfg['fpn_out_channels']
fpn_start_level = 1
fpn_num_outs = 6
basesize_ratio_range = (0.1, 0.9)

conv_cfg = dict(type='ConvDWSep', group_size_dw=regnet_cfg['group_size_dw'])
norm_cfg = dict(type='BN')
img_norm_cfg = dict(mean=[128.0, 128.0, 128.0], std=[64.0, 64.0, 64.0], to_rgb=to_rgb)

model = dict(
    type='SingleStageDetector',
    pretrained=pretrained,
    backbone=dict(
        type=backbone_type,
        arch=backbone_arch,
        base_channels=regnet_base_channels,
        out_indices=backbone_out_indices,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='JaiFPN',
        in_channels=fpn_in_channels,
        out_channels=fpn_out_channels,
        start_level=fpn_start_level,
        num_outs=fpn_num_outs,
        add_extra_convs='on_output',
        upsample_cfg=dict(scale_factor=2,mode='bilinear'),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='JaiSSDHead',
        in_channels=[fpn_out_channels for _ in range(6)],
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
        hue_delta=18),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=3,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))


# settings for qat or calibration - uncomment after doing floating point training
# also change dataset_repeats in the dataset config to 1 for fast learning
quantize = False #'training' #'calibration'
if quantize:
  load_from = './work_dirs/ssd_regnet_fpn_bgr/latest.pth'
  optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=1e-4)
  lr_config = dict(policy='CosineAnealing', min_lr_ratio=1e-3, warmup='linear', warmup_iters=100, warmup_ratio=1e-4)
  total_epochs = 1 if quantize == 'calibration' else 5
#
