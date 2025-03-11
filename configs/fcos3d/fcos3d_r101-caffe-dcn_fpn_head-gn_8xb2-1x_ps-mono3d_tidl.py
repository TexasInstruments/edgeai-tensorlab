_base_ = [
    '../_base_/datasets/pandaset-mono3d.py', '../_base_/models/fcos3d.py',
    '../_base_/schedules/mmdet-schedule-1x.py', '../_base_/default_runtime.py'
]
# model settings
custom_imports = dict(imports=['projects.FCOS3D.fcos3d'])

# model settings
model = dict(
    save_onnx_model=False,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        dcn=dict(type='DCNv2_tidl', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    bbox_head=dict(
        type='CustomFCOSMono3DHead',
        dcn_on_last_conv=True,
        num_classes=27, 
        num_attrs=22,),
    test_cfg=dict(
        use_rotate_nms=False,
        # score_thr=0.0025,
        ))

env_cfg = dict(
    dist_cfg = dict(timeout=3600)
)

train_cfg = dict(max_epochs=5)

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize3D', scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    # dict(type='mmdet.Resize', scale=(1600, 900), keep_ratio=True),
    dict(type='Resize3D', scale_factor=1.0),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=2, num_workers=4, dataset=dict(pipeline=train_pipeline, ))
test_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.002),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
# TODO remove this
load_from = './checkpoints/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_ps-mono3d_finetune_20210717_095645-8d806dc2_adjusted.pth'

find_unused_parameters = True
