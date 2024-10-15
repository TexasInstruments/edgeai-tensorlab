# Copyright (c) Phigent Robotics. All rights reserved.

# mAP: 0.2828
# mATE: 0.7734
# mASE: 0.2884
# mAOE: 0.6976
# mAVE: 0.8637
# mAAE: 0.2908
# NDS: 0.3500
#
# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.517	0.533	0.161	0.123	0.909	0.235
# truck	0.226	0.745	0.232	0.222	0.848	0.268
# bus	0.305	0.797	0.220	0.192	1.982	0.355
# trailer	0.101	1.107	0.230	0.514	0.536	0.068
# construction_vehicle	0.039	1.105	0.501	1.402	0.119	0.386
# pedestrian	0.318	0.805	0.305	1.341	0.826	0.650
# motorcycle	0.216	0.783	0.286	0.977	1.224	0.273
# bicycle	0.203	0.712	0.304	1.354	0.465	0.090
# traffic_cone	0.499	0.547	0.347	nan	nan	nan
# barrier	0.404	0.599	0.297	0.153	nan	nan

_base_ = ['../../../configs/_base_/datasets/nus-3d.py',
          '../../../configs/_base_/default_runtime.py']

custom_imports = dict(imports=['projects.BEVDet.bevdet'])

# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)


ida_aug_conf = {
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

model = dict(
    type='BEVDet',
    save_onnx_model=False,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675,  116.280, 103.530],
        std=[58.395, 57.120, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
		init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=ida_aug_conf['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pts_bbox_head=dict(
        type='BEVDetHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=class_names),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    
    # model training and testing settings
    # Do we need here?
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['circle'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

# Data
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
#file_client_args = dict(backend='disk')
backend_args = None

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=6,
        backend_args=backend_args),
    dict(type='ImageAug',
         ida_aug_conf=ida_aug_conf,
         is_train=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='CustomPack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]


test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=6,
        backend_args=backend_args),
    dict(type='ImageAug',
         ida_aug_conf=ida_aug_conf,
         is_train=False),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(
        type='CustomMultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='CustomPack3DDetInputs', keys=['points', 'img'])
        ])
]


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
            ann_file='nuscenes_infos_train.pkl',
            data_prefix=dict(
                CAM_FRONT='samples/CAM_FRONT',
                CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                CAM_BACK='samples/CAM_BACK',
                CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
                CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
            pipeline=train_pipeline,
            box_type_3d='LiDAR',
            metainfo=metainfo,
            test_mode=False,
            modality=input_modality,
            use_valid_flag=True,
            backend_args=backend_args)))



val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))


test_dataloader = val_dataloader


val_evaluator = dict(
    type='CustomNuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='mAP',
    backend_args=backend_args)
test_evaluator = val_evaluator


# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-2),
    #paramwise_cfg=dict(custom_keys={
    #    'img_backbone': dict(lr_mult=0.1),
    #}),
    clip_grad=dict(max_norm=5, norm_type=2))


# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=7000),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=24,
        T_max=24,
        eta_min=1e-6)
]


train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=2, max_keep_ckpts=4, save_last=True))

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

#load_from='checkpoints/BEVDet/epoch_18_mmdet3d.pth'
#fp16 = dict(loss_scale='dynamic')