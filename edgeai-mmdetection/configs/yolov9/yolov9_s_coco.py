_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

# training settings
max_epochs = 300
num_last_epochs = 15
interval = 5

img_scale = (640, 640)
batch_size = 16

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)
model = dict(
    type='YOLOV9',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='YOLOV9Backbone',
        stem_channels=[32, 64],
        expand_list=[128, 192, 256],
        pool_kernel_size=2,
        pool_type='avg',
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/WongKinYiu/yolov9mit/releases/download/v1.0-alpha/v9-s.pt')
        ),
    neck=dict(
        type='YOLOV9Neck',
        in_channels=[128, 192, 256],
        pool_kernel_size=2,
        pool_type='avg',
        csp_arg = {"repeat_num": 3}
        ),
    bbox_head=dict(
        type='YOLOV9Head',
        num_classes=80,
        in_channels=[128, 192, 256],
        strides=(8, 16, 32),
        reg_max=16,
        loss_yolo=dict(
            type='YOLOV9Loss',
                loss_cfg = dict(
                    objective=dict(
                        BCELoss=0.5,
                        BoxLoss=7.5,
                        DFLoss=1.5
                    ),
                    aux=0.25,
                    matcher=dict(
                        iou='CIoU',
                        topk=10,
                        factor=dict(
                            iou=6.0,
                            cls=0.5
                        )
                    )
                ),
                class_num=80,
        ),
        ),
        aux_head=dict(
        type='YOLOV9AuxHead',
        in_channels=[128, 192, 256],
        csp_arg = {"repeat_num": 3},
        num_classes=80,
        reg_max=16,
        ),
    # training and testing settings
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)
        ))
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
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    # dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
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

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 4e-3
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning policy
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2000),
#     dict(type='MultiStepLR', by_epoch=True, milestones=[218, 246], gamma=0.1)
# ]
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]


default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=20))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=batch_size)
