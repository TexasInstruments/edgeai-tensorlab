# -*- coding: utf-8 -*-
_base_ = [
    'mmdet3d::_base_/datasets/pandaset-3d-3classes.py',
    'mmdet3d::_base_/default_runtime.py',
]

custom_imports = dict(imports=['projects.FastBEV.fastbev'])

# sequential = False for n_times = 1
n_times = 1
sequential=True

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
        num_classes=3,
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
            #sizes=[
            #    [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
            #    [0.5774, 1.7321, 1.],  # 1/sqrt(3)
            #    [1., 1., 1.],
            #    [0.4, 0.4, 1],
            #],
            sizes=[
                [2.5981, 0.8660, 1.],  # 1.5/sqrt(3)
                [1.7321, 0.5774, 1.],  # 1/sqrt(3)
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
        use_rotate_nms=False,
        nms_thr=0.2,
        # Scale-NMS
        nms_type_list=[
            'circle', 'circle', 'circle',],
        nms_thr_list=[0.2, 0.5, 0.2],
        nms_radius_thr_list=[4, 0.175, 1.0],
        nms_rescale_factor=[1.0, 9.0, 1.0],
    )
)

# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = [
    'Car','Pedestrian','Temporary Construction Barriers'
]
dataset_type = 'CustomPandaSetDataset'
data_root = './data/pandaset/'

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

data_config = {
    'src_size': (1080, 1920),
    'input_size': (256, 704),
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    #'resize': (0.0, 0.0),
    #'crop': (0.0, 0.0),
    #'rot': (0.0, 0.0),
    #'flip': False,
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
    dict(type='MultiViewPipeline', sequential=sequential, n_images=6, n_times=n_times, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         #with_bev_seg=False
         ),
    dict(
        type='CustomLoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='CustomRandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        flip_box3d=True,
        #flip_ratio_bev_horizontal=0.0,
        #flip_ratio_bev_vertical=0.0,
        #flip_box3d=False,
        update_img2lidar=True),
    dict(
        type='CustomGlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05],
        #rot_range=[0.0, 0.0],
        #scale_ratio_range=[1.0, 1.0],
        #translation_std=[0.0, 0.0, 0.0],
        update_img2lidar=True),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ResetPointOrigin', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img',
                                        #'gt_bboxes', 'gt_labels',
                                       'gt_bboxes_3d', 'gt_labels_3d'])]

test_pipeline = [
    dict(type='MultiViewPipeline', sequential=sequential, n_images=6, n_times=n_times, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='ResetPointOrigin', point_cloud_range=point_cloud_range),
    dict(type='CustomPack3DDetInputs', keys=['img'])
    ]

class_mapping = [
    0,2,2,1,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,1,2,2,2,
]
metainfo = dict(classes=class_names, class_mapping=class_mapping)
data_prefix = dict(
    pts='',
    front_camera='camera/front_camera',
    front_left_camera='camera/front_left_camera',
    front_right_camera='camera/front_right_camera',
    back_camera='camera/back_camera',
    left_camera='camera/left_camera',
    right_camera='camera/right_camera')

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
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
            with_box2d=False,
            box_type_3d='LiDAR',
            ann_file='pandaset_infos_train_fastbev.pkl',
            sequential=sequential,
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
            #max_dist_thr=[50,50]
        ))
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        ann_file='pandaset_infos_val.pkl',
        sequential=sequential,
        n_times=n_times,
        train_adj_ids=[1, 3, 5], # not needed
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],  # not needed
        test_time_id=None,
        #max_dist_thr=[50,50]
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CustomPandaSetMetric',
    data_root=data_root,
    max_dists=[50, 50],
    ann_file=data_root + 'pandaset_infos_val.pkl',
    metric='mAP',
    backend_args=None)
test_evaluator = val_evaluator


# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    }),
    clip_grad=dict(max_norm=35, norm_type=2))

total_epochs = 24

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=total_epochs,
        T_max=total_epochs,
        eta_min_ratio=1e-3)
]

train_cfg = dict(by_epoch=True, max_epochs=total_epochs, val_interval=total_epochs)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=4, save_last=True))


#load_from = 'pretrained/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'
load_from = 'checkpoints/fastbev/edgeai_fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f1.pth'
resume_from = None

# fp16 settings, the loss scale is specifically tuned to avoid Nan
#fp16 = dict(loss_scale='dynamic')
