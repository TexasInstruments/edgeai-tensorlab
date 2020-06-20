_base_ = [
    '../_jacinto_ai_base_/hyper_params/common_config.py',
    '../_jacinto_ai_base_/hyper_params/ssd_config.py',
    '../_jacinto_ai_base_/hyper_params/schedule_1x.py',
]

dataset_type = 'CocoDataset'

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


input_size = (768,384)          # (1536,768) #(1024,512) #(768,384) #(512,512)

backbone_type = 'ResNet'
backbone_depth = 50
pretrained='torchvision://resnet50'
bacbone_out_channels=[256, 512, 1024, 2048]
backbone_out_indices = (0, 1, 2, 3)

fpn_in_channels = bacbone_out_channels
fpn_out_channels = 256
fpn_start_level = 1
fpn_num_outs = 5

#retinanet_base_stride = (8 if fpn_start_level==1 else (4 if fpn_start_level==0 else None))

# for multi-scale training
input_size_ms = [(input_size[0], (input_size[1]*8)//10),
                 (input_size[0], (input_size[1]*9)//10),
                 input_size]

conv_cfg = None
norm_cfg = None

# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #for mmdet trained
img_norm_cfg = dict(mean=[128.0, 128.0, 128.0], std=[64.0, 64.0, 64.0], to_rgb=True)

model = dict(
    type='RetinaNet',
    pretrained=pretrained,
    backbone=dict(
        type='ResNet',
        depth=backbone_depth,
        num_stages=4,
        out_indices=backbone_out_indices,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=fpn_in_channels,
        out_channels=fpn_out_channels,
        start_level=fpn_start_level,
        num_outs=fpn_num_outs,
        add_extra_convs='on_input', #'on_output',
        upsample_cfg=dict(scale_factor=2, mode='bilinear'), #,mode='nearest'),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=num_classes,
        in_channels=fpn_out_channels,
        stacked_convs=4,
        feat_channels=fpn_out_channels,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=input_size_ms,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
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
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
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

#optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=1e-4)
#load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
#resume_from = './work_dirs/retinanet_resnet_fpn/latest.pth'