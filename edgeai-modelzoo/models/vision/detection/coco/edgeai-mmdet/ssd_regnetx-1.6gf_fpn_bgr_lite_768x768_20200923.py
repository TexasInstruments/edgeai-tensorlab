dataset_type = 'CocoDataset'
data_root = 'data/coco/'
dataset_repeats = 1
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='CocoDataset',
            ann_file='data/coco/annotations/instances_train2017.json',
            img_prefix='data/coco/train2017/',
            pipeline=[
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
                    mean=[103.53, 116.28, 123.675],
                    to_rgb=False,
                    ratio_range=(1, 4)),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                    min_crop_size=0.3),
                dict(type='Resize', img_scale=(768, 768), keep_ratio=False),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[57.375, 57.12, 58.395],
                    to_rgb=False),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(768, 768),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[57.375, 57.12, 58.395],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(768, 768),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[57.375, 57.12, 58.395],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
cudnn_benchmark = True
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './work_dirs/20200914-115249_ssd-lite_regnetx-1.6gf_fpn_bgr_768x768_epochs120_bs16x5_lr8e-2_(36.2%)/latest.pth'
resume_from = None
workflow = [('train', 1)]
print_model_complexity = True
fp16 = dict(loss_scale=512.0)
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.0,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.0,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
warmup_cfg = dict(warmup='linear', warmup_iters=1000, warmup_ratio=0.001)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.0001,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)
total_epochs = 120
input_size = (768, 768)
num_classes_dict = dict(CocoDataset=80, VOCDataset=20, CityscapesDataset=8)
num_classes = 80
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)
quantize = False
initial_learning_rate = 0.08
samples_per_gpu = 16
backbone_type = 'RegNet'
backbone_arch = 'regnetx_1.6gf'
to_rgb = False
decoder_fpn_type = 'FPNLite'
decoder_conv_type = 'ConvDWSep'
fpn_width_fact = 4
decoder_width_fact = 4
decoder_depth_fact = 4
regnet_settings = dict({
    'regnetx_800mf':
    dict(
        bacbone_out_channels=[64, 128, 288, 672],
        group_size_dw=16,
        fpn_intermediate_channels=256,
        fpn_out_channels=256,
        fpn_num_blocks=4,
        pretrained='open-mmlab://regnetx_800mf'),
    'regnetx_1.6gf':
    dict(
        bacbone_out_channels=[72, 168, 408, 912],
        group_size_dw=24,
        fpn_intermediate_channels=264,
        fpn_out_channels=264,
        fpn_num_blocks=4,
        pretrained='open-mmlab://regnetx_1.6gf'),
    'regnetx_3.2gf':
    dict(
        bacbone_out_channels=[96, 192, 432, 1008],
        group_size_dw=48,
        fpn_intermediate_channels=288,
        fpn_out_channels=288,
        fpn_num_blocks=4,
        pretrained='open-mmlab://regnetx_3.2gf')
})
regnet_cfg = dict(
    bacbone_out_channels=[72, 168, 408, 912],
    group_size_dw=24,
    fpn_intermediate_channels=264,
    fpn_out_channels=264,
    fpn_num_blocks=4,
    pretrained='open-mmlab://regnetx_1.6gf')
pretrained = 'open-mmlab://regnetx_1.6gf'
bacbone_out_channels = [72, 168, 408, 912]
backbone_out_indices = (0, 1, 2, 3)
fpn_in_channels = [72, 168, 408, 912]
fpn_out_channels = 264
fpn_start_level = 1
fpn_num_outs = 6
fpn_upsample_mode = 'bilinear'
fpn_upsample_cfg = dict(scale_factor=2, mode='bilinear')
fpn_num_blocks = 4
fpn_intermediate_channels = 264
fpn_bifpn_cfg = dict()
fpn_add_extra_convs = 'on_input'
basesize_ratio_range = (0.1, 0.9)
input_size_divisor = 32
conv_cfg = dict(type='ConvDWSep', group_size_dw=24)
norm_cfg = dict(type='BN')
model = dict(
    type='SingleStageDetector',
    pretrained='open-mmlab://regnetx_1.6gf',
    backbone=dict(
        type='RegNet',
        arch='regnetx_1.6gf',
        out_indices=(0, 1, 2, 3),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='FPNLite',
        in_channels=[72, 168, 408, 912],
        out_channels=264,
        start_level=1,
        num_outs=6,
        add_extra_convs='on_input',
        upsample_cfg=dict(scale_factor=2, mode='bilinear'),
        conv_cfg=dict(type='ConvDWSep', group_size_dw=24),
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='SSDLiteHead',
        in_channels=[264, 264, 264, 264, 264, 264],
        num_classes=80,
        conv_cfg=dict(type='ConvDWSep', group_size_dw=24),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=(768, 768),
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2])))
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
        mean=[103.53, 116.28, 123.675],
        to_rgb=False,
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(768, 768), keep_ratio=False),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(768, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
work_dir = './work_dirs/ssd-lite_regnet_fpn_bgr'
gpu_ids = range(0, 1)
