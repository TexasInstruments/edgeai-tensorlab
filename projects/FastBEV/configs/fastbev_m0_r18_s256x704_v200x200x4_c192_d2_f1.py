# -*- coding: utf-8 -*-
_base_ = [
    'mmdet3d::_base_/datasets/nus-3d.py',
    'mmdet3d::_base_/default_runtime.py',
]

custom_imports = dict(imports=['projects.FastBEV.fastbev'])

n_times = 1
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True)

model = dict(
    type='FastBEV',
    style="v1",
    save_onnx_model=False,
    num_temporal_feats=n_times-1,
    feats_size = [6, 64, 64, 176],
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        **img_norm_cfg,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        style='pytorch'
    ),
    neck=dict(
        type='mmdet.FPN',
        norm_cfg=dict(type='BN', requires_grad=True),
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=[256], out_channels=[64]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=64*4,
        out_channels=192,
        num_layers=2,
        stride=2,
        is_transpose=False,
        fuse=dict(in_channels=64*4*n_times, out_channels=64*4),
        norm_cfg=dict(type='BN', requires_grad=True)),
    seg_head=None,
    bbox_head=dict(
        type='CustomFreeAnchor3DHead',
        is_transpose=True,
        num_classes=10,
        in_channels=192,
        feat_channels=192,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            # scales=[1, 2, 4],
            sizes=[
                [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    multi_scale_id=[0],
    n_voxels=[[200, 200, 4]],
    voxel_size=[[0.5, 0.5, 1.5]],
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='mmdet.MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500,
        use_scale_nms=True,
        # Normal-NMS
        nms_across_levels=False,
        use_rotate_nms=True,
        nms_thr=0.2,
        # Scale-NMS
        nms_type_list=[
            'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'circle'],
        nms_thr_list=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2],
        nms_radius_thr_list=[4, 12, 10, 10, 12, 0.85, 0.85, 0.175, 0.175, 1],
        nms_rescale_factor=[1.0, 0.7, 0.55, 0.4, 0.7, 1.0, 1.0, 4.5, 9.0, 1.0],
    )
)

# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

data_config = {
    'src_size': (900, 1600),
    'input_size': (256, 704),
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    # test-aug
    'test_input_size': (256, 704),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    # top, right, bottom, left
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=6, n_times=n_times, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D',
         with_bbox=True,
         with_label=True,
         with_bev_seg=True),
    #dict(
    #    type='LoadPointsFromFile',
    #    dummy=True,
    #    coord_type='LIDAR',
    #    load_dim=5,
    #    use_dim=5),
    dict(
        type='RandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05],
        update_img2lidar=True),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ResetPointOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes', 'gt_labels',
                                 'gt_bboxes_3d', 'gt_labels_3d',
                                 'gt_bev_seg'])]
test_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=6, n_times=n_times, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='ResetPointOrigin', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img'])
    ]


metainfo = dict(classes=class_names)
data_prefix = dict(
    pts='',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            data_prefix=data_prefix,
            test_mode=False,
            with_box2d=True,
            box_type_3d='LiDAR',
            ann_file='nuscenes_infos_train_fastbev.pkl',
            sequential=True,
            n_times=n_times,
            train_adj_ids=[1, 3, 5],
            speed_mode='abs_velo',
            max_interval=10,
            min_interval=0,
            fix_direction=True,
            prev_only=True,
            test_adj='prev',
            test_adj_ids=[1, 3, 5],
            test_time_id=None,
        )))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        with_box2d=True,
        box_type_3d='LiDAR',
        ann_file='nuscenes_infos_val.pkl',
        sequential=True,
        n_times=n_times,
        train_adj_ids=[1, 3, 5], # not needed
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],  # not needed
        test_time_id=None,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CustomNuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='mAP',
    backend_args=None)
test_evaluator = val_evaluator


# Optimizer
optim_wrapper = dict(
    #optimizer=dict(type='AdamW2', lr=0.0004, weight_decay=0.01),
    optimizer=dict(type='AdamW', lr=0.0004, weight_decay=0.01),
    #paramwise_cfg=dict(custom_keys={
    #    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    #}),
    clip_grad=dict(max_norm=35, norm_type=2))


total_epochs = 1

train_cfg = dict(max_epochs=total_epochs, val_interval=total_epochs)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=4, save_last=True))


load_from = 'pretrained/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'
resume_from = None

# fp16 settings, the loss scale is specifically tuned to avoid Nan
#fp16 = dict(loss_scale='dynamic')
