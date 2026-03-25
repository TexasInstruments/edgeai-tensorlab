_base_ = ['../../configs/_base_/schedules/schedule_1x.py', '../../configs/_base_/default_runtime.py']
input_size = (512,512) 

resize_with_scale_factor = False
######################################################
backbone_type = 'RegNet'
backbone_arch = 'regnetx_1.6gf'                  # 'regnetx_800mf' #'regnetx_1.6gf' #'regnetx_3.2gf'
to_rgb = False                                   # pycls regnet backbones are trained with bgr
decoder_conv_type = 'ConvDWSep'                 # 'ConvDWSep' #'ConvDWTripletRes' #'ConvDWTripletAlwaysRes'

regnet_settings = {
    'regnetx_200mf': {'bacbone_out_channels': [32, 56, 152, 368], 'group_size_dw': 8,
                      'pretrained': './checkpoints/RegNetX-200MF_dds_8gpu_mmdet-converted.pyth'},
    'regnetx_400mf': {'bacbone_out_channels': [32, 64, 160, 384], 'group_size_dw': 16,
                      'pretrained': 'open-mmlab://regnetx_400mf'},
    'regnetx_800mf':{'bacbone_out_channels':[64, 128, 288, 672], 'group_size_dw':16,
                     'pretrained':'open-mmlab://regnetx_800mf'},
    'regnetx_1.6gf':{'bacbone_out_channels':[72, 168, 408, 912], 'group_size_dw':24,
                     'pretrained':'open-mmlab://regnetx_1.6gf'},
    'regnetx_3.2gf':{'bacbone_out_channels':[96, 192, 432, 1008], 'group_size_dw':48,
                     'pretrained': 'open-mmlab://regnetx_3.2gf'}
}

######################################################
regnet_cfg = regnet_settings[backbone_arch]
pretrained=regnet_cfg['pretrained']
bacbone_out_channels=regnet_cfg['bacbone_out_channels'][1:][::-1]
backbone_out_indices = (1, 2, 3)
group_size_dw=regnet_cfg['group_size_dw']
fpn_in_channels = bacbone_out_channels
fpn_out_channels = regnet_cfg['bacbone_out_channels'][:-1][::-1]

conv_cfg = None
norm_cfg = dict(type='BN')
act_cfg = dict(type='ReLU')
convert_to_lite_model = dict(group_size_dw=group_size_dw, model_surgery=1)

# model settings
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    pad_size_divisor=32)
model = dict(
    type='YOLOV3',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=backbone_type,
        arch=backbone_arch,
        out_indices=backbone_out_indices,
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=fpn_in_channels,
        out_channels=fpn_out_channels,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=80,
        in_channels=fpn_out_channels,
        out_channels=fpn_in_channels,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # `mean` and `to_rgb` should be the same with the `preprocess_cfg`
    dict(type='Expand', mean=[0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='RandomResize', scale=input_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=273, val_interval=7)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2000),
    dict(type='MultiStepLR', by_epoch=True, milestones=[218, 246], gamma=0.1)
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=7))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
