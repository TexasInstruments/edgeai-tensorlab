# Copyright (c) Phigent Robotics. All rights reserved.

# mAP: 0.2843
# mATE: 0.7585
# mASE: 0.2807
# mAOE: 0.6717
# mAVE: 0.9227
# mAAE: 0.2875
# NDS: 0.3500
# Eval time: 80.6s
# 
# Per-class results:
# Object Class            AP      ATE     ASE     AOE     AVE     AAE   
# car                     0.516   0.556   0.160   0.109   0.967   0.227 
# truck                   0.240   0.754   0.224   0.162   0.799   0.258 
# bus                     0.316   0.813   0.214   0.212   2.054   0.354 
# trailer                 0.111   1.140   0.225   0.529   0.495   0.049 
# construction_vehicle    0.039   1.053   0.496   1.352   0.115   0.424 
# pedestrian              0.281   0.734   0.300   1.407   0.967   0.642 
# motorcycle              0.267   0.743   0.258   0.795   1.519   0.243 
# bicycle                 0.239   0.674   0.302   1.340   0.466   0.102 
# traffic_cone            0.419   0.544   0.334   nan     nan     nan   
# barrier                 0.417   0.573   0.293   0.139   nan     nan   


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
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675,  116.280, 103.530],
        std=[58.395, 57.120, 57.375],
        #mean=[0.0,  0.0, 0.0],
        #std=[1.0, 1.0, 1.0],
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
    #dict(type='Pack3DDetInputs', keys=['points', 'img_input'])
]


train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
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
        backend_args=backend_args))



val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
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
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-07),
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

#runner = dict(type='EpochBasedRunner', max_epochs=24)
train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=12)
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

#load_from='checkpoints/BEVDet/epoch_1.pth'
#fp16 = dict(loss_scale='dynamic')
