
# modified from: https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox

img_scale = (320,320)
# input_size = img_scale
samples_per_gpu = 8

# dataset settings
dataset_type = 'CocoDataset'
num_classes_dict = {'CocoDataset':80, 'VOCDataset':20, 'CityscapesDataset':8, 'WIDERFaceDataset':1}
dataset_root_dict = {'CocoDataset':'data/coco/', 'VOCDataset':'data/VOCdevkit/', 'CityscapesDataset':'data/cityscapes/', 'WIDERFaceDataset':'data/WIDERFace/'}
num_classes = num_classes_dict[dataset_type]
data_root = dataset_root_dict[dataset_type]
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)

# replace complex activation functions with ReLU.
# Also replace regular convolutions with depthwise-separable convolutions.
# edgeai_torchmodelopt needs to be installed from edgeai-modeloptimization
convert_to_lite_model = dict(group_size_dw=None)

_base_ = [
    # f'../_xbase_/datasets/{dataset_type.lower()}.py',
    f'../_xbase_/datasets/cocodataset.py',
    '../_xbase_/hyper_params/yolox_config.py',
    '../_xbase_/hyper_params/yolox_schedule.py', 
]

# settings for qat or calibration - set to True after doing floating point training
quantize = False #'training' #'calibration'
if quantize:
    load_from = './work_dirs/yolox_s_lite/latest.pth'
    max_epochs = (1 if quantize == 'calibration' else 12)
    initial_learning_rate = 1e-4
    num_last_epochs = max_epochs//2
    interval = 10
    resume_from = None
else:
    load_from = None #'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
    max_epochs = 300
    initial_learning_rate = 0.01
    num_last_epochs = 15
    interval = 1
    resume_from = None
#

# model settings
model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)
    ]),
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.1875,act_cfg=dict(type='ReLU')),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[48,96,192],
        out_channels=48,
        num_csp_blocks=1,act_cfg=dict(type='ReLU')),
    bbox_head=dict(
        type='YOLOXHead', num_classes=num_classes, in_channels=48, feat_channels=48,act_cfg=dict(type='ReLU')),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))


train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
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
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

backend_args = None

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
    dict(type='Resize', scale=img_scale, keep_ratio=True),
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
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=8,
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

val_cfg = dict(type='ValLoop')
train_cfg = dict( type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=interval)
test_cfg = dict(type='TestLoop')

# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
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

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        # num_last_epochs=num_last_epochs,
        # interval=interval,
        priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        # resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]


# optimizer = dict(
#     type='SGD',lr=initial_learning_rate)

# lr_config = dict(
#     num_last_epochs=num_last_epochs)
