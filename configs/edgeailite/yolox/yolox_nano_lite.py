
# modified from: https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox

img_scale = (416, 416)
input_size = img_scale
samples_per_gpu = 16

# dataset settings
dataset_type = 'CocoDataset'
num_classes_dict = {'CocoDataset':80, 'VOCDataset':20, 'CityscapesDataset':8, 'WIDERFaceDataset':1}
dataset_root_dict = {'CocoDataset':'data/coco/', 'VOCDataset':'data/VOCdevkit/', 'CityscapesDataset':'data/cityscapes/', 'WIDERFaceDataset':'data/WIDERFace/'}
num_classes = num_classes_dict[dataset_type]
data_root = dataset_root_dict[dataset_type]
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)

# replace complex activation functions with ReLU.
# Also replace regular convolutions with depthwise-separable convolutions.
# torchvision.edgeailite requires edgeai-torchvision to be installed
convert_to_lite_model = dict(group_size_dw=None)

_base_ = [
    f'../_xbase_/datasets/{dataset_type.lower()}.py',
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
    interval = 10
    resume_from = None
#

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(10, 20),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.25, use_depthwise=False),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=False),
    bbox_head=dict(
        type='YOLOXHead', num_classes=num_classes, in_channels=64, feat_channels=64, use_depthwise=False),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            filter_empty_gt=False,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ]
        ),
        pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

optimizer = dict(
    type='SGD',lr=initial_learning_rate)

lr_config = dict(
    num_last_epochs=num_last_epochs)
