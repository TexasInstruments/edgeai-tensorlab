_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/cyclic-20e.py',
    # '../../../configs/_base_/models/centerpoint_pillar02_second_secfpn_nus.py',
    # '../../../configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
custom_imports = dict(imports=['projects.Sparse4D.sparse4d'])

point_cloud_range = [-55, -55, -5.0, 55, 55, 3.0]
voxel_size = [0.2, 0.2, 8]
data_root = './data/nuscenes/'
class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone'
]
metainfo = dict(classes=class_names)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False
)
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
model = dict(
    type='Sparse4D',
    use_grid_mask=True,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        frozen_stages=1,
        norm_eval=True,
        style='caffe',
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        stage_with_dcn=(False, False, True, True),
        norm_cfg=dict(type='BN2d', requires_grad=False),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
    ),
    img_neck=dict(
        type='mmdet.FPN',
        num_outs=4,
        start_level=1,
        out_channels=embed_dims,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        num_decoder=num_decoder,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor=data_root + "nuscenes_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            embed_dims=embed_dims,
            vel_dims=3,
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            dropout=0.1,
        ),
        norm_layer=dict(type='LN', normalized_shape=embed_dims),
        ffn=dict(
            type="FFN",
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 2,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=4,
            num_cams=6,
            proj_drop=0.1,
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [1.0] * 7,
            cls_wise_reg_weights={
                class_names.index("traffic_cone"): [
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0
                ],
            },
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(type='mmdet.L1Loss', loss_weight=0.25),
        gt_cls_key="gt_labels_3d",
        gt_reg_key="gt_bboxes_3d",
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            fix_scale=[
                [0, 0, 0],
                [0.45, 0, 0],
                [-0.45, 0, 0],
                [0, 0.45, 0],
                [0, -0.45, 0],
                [0, 0, 0.45],
                [0, 0, -0.45],
            ],
        ),
        depth_module=dict(
            type="DepthReweightModule",
            embed_dims=embed_dims,
        ),
    ),
)
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

dataset_type = 'CustomNuScenesDataset'

backend_args = None

file_client_args = dict(backend='disk')
ida_aug_conf = {
        "resize_lim": (0.38, 0.55),
        "final_dim": (256, 704),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True, with_2d=False),
    # dict(type="CustomCropMultiViewImage", crop_range=img_crop_range),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True,
            ),
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(
    #     type='CircleObjectRangeFilter',
    #     class_dist_thred=[55] * len(class_names)
    # ),
    # dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='NuScenesSparse4DAdaptorV1'),
    dict(
        type='CustomPack3DDetInputs',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img',],#"timestamp", "projection_mat","image_wh",],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d', 'sample_idx',
                   "timestamp", "T_global", "T_global_inv", "timestamp", "projection_mat","image_wh"] + collect_keys)
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False,  with_2d=False),
    # dict(type="CustomCropMultiViewImage", crop_range=img_crop_range),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    # dict(type='GlobalRotScaleTransImage',
    #         rot_range=[-0.3925, 0.3925],
    #         translation_std=[0, 0, 0],
    #         scale_ratio_range=[0.95, 1.05],
    #         reverse_angle=True,
    #         training=True,
            # ),
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(
    #     type='CircleObjectRangeFilter',
    #     class_dist_thred=[55] * len(class_names)
    # ),
    # dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='NuScenesSparse4DAdaptorV1'),
    dict(
        type='CustomPack3DDetInputs',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img',"timestamp", "projection_mat","image_wh",],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d', 'sample_idx',
                   "timestamp", "T_global", "T_global_inv"] + collect_keys)
]

batch_size = 1

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    # sampler=dict(type='GroupSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train_sparse4d_mini.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        #collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        # collect_keys=collect_keys + ['img', 'prev_exists'],
        # queue_length=queue_length,
        filter_empty_gt=False,
        pipeline=train_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=False,
        modality=input_modality,
        use_valid_flag=True,
        batch_size=batch_size, # Needed for GroupSampler
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    # sampler=dict(type='GroupSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val_sparse4d_mini.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        #collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        # collect_keys=collect_keys + ['img', 'prev_exists'],
        # queue_length=queue_length,
        filter_empty_gt=False,
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True,
        batch_size=batch_size, # Needed for GroupSampler
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    # sampler=dict(type='GroupSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val_sparse4d_mini.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        #collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        # collect_keys=collect_keys + ['img', 'prev_exists'],
        # queue_length=queue_length,
        filter_empty_gt=False,
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True,
        batch_size=batch_size, # Needed for GroupSampler
        backend_args=backend_args))