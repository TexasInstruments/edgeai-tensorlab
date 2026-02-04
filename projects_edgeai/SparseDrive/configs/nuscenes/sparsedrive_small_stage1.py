_base_ = [
    'mmdet3d::_base_/datasets/nus-3d.py',
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/schedules/cyclic-20e.py'
]

custom_imports = dict(imports=['projects_edgeai.SparseDrive.sparsedrive',
                               'projects_edgeai.edgeai_mmdet3d'])

# ================ base config ===================

#total_batch_size = 64
#num_gpus = 8
#batch_size = total_batch_size // num_gpus
#num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
#checkpoint_epoch_interval = 20
num_epochs = 100
batch_size = 4

#checkpoint_config = dict(
#    interval=num_iters_per_epoch * checkpoint_epoch_interval
#)
#log_config = dict(
#    interval=51,
#    hooks=[
#        dict(type="TextLoggerHook", by_epoch=False),
#        dict(type="TensorboardLoggerHook"),
#    ],
#)
#workflow = [("train", 1)]
#fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)


# ================== model ========================
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True
)

class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
map_class_names = [
    'ped_crossing',
    'divider',
    'boundary',
]

metainfo = dict(classes=class_names)

num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)

num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4 # history + current

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
# use_deformable_func should be True for training to save memory
# use_deformable_func should be False for ONNX export while inferencing
# For use_deformable_func=True, mmdet3d_plugin/ops/setup.py needs to be executed
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True

task_config = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=False,
)

dataset_type = "SparseDriveNuScenesDataset"
data_root = "data/nuscenes/"
anno_root = "data/nuscenes/ad_infos/"
backend_args = None

model = dict(
    type="SparseDrive",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    save_onnx_model=False,
    onnx_subnets=False,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        #**img_norm_cfg,
        mean=[0.0, 0.0, 0.0],
        std =[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    img_backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="pretrained/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="mmdet.FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    pts_bbox_head=dict(
        type="SparseDriveHead",
        task_config=task_config,
        det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor="data/nuscenes/kmeans/kmeans_det_900.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal else -1,
                confidence_decay=0.6,
                feat_grad=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder)
            )[2:],
            temp_graph_model=dict(
                #type="MultiheadFlashAttention",
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal
            else None,
            graph_model=dict(
                #type="MultiheadFlashAttention",
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="Sparse4DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
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
                refine_yaw=True,
                with_quality_estimation=with_quality_estimation,
            ),
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    class_names.index("traffic_cone"): [
                        2.0,
                        2.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                },
            ),
            loss_cls=dict(
                type="mmdet.FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="mmdet.L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="mmdet.GaussianFocalLoss"),
                cls_allow_reverse=[class_names.index("barrier")],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),
        map_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn_map,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor="data/nuscenes/kmeans/kmeans_map_100.npy",
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.6,
                feat_grad=True,
            ),
            anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=num_sample,
            ),
            num_single_frame_decoder=num_single_frame_decoder_map,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder_map
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder_map)
            )[:],
            temp_graph_model=dict(
                #type="MultiheadFlashAttention",
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal_map
            else None,
            graph_model=dict(
                #type="MultiheadFlashAttention",
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="Sparse4DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=num_sample,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023, # ground height in lidar frame
                ),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=num_sample,
                num_cls=num_map_classes,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            loss_cls=dict(
                type="mmdet.FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(
                    type='LinesL1Loss',
                    loss_weight=10.0,
                    beta=0.01,
                ),
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            decoder=dict(type="SparsePoint3DDecoder"),
            reg_weights=[1.0] * 40,
            gt_cls_key="gt_map_labels",
            gt_reg_key="gt_map_pts",
            gt_id_key="map_instance_id",
            with_instance_id=False,
            task_prefix='map',
        ),
        motion_plan_head=dict(
            type='MotionPlanningHead',
            fut_ts=fut_ts,
            fut_mode=fut_mode,
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
            motion_anchor=f'data/nuscenes/kmeans/kmeans_motion_{fut_mode}.npy',
            plan_anchor=f'data/nuscenes/kmeans/kmeans_plan_{ego_fut_mode}.npy',
            embed_dims=embed_dims,
            decouple_attn=decouple_attn_motion,
            instance_queue=dict(
                type="InstanceQueue",
                embed_dims=embed_dims,
                queue_length=queue_length,
                tracking_threshold=0.2,
                feature_map_scale=(input_shape[1]/strides[-1], input_shape[0]/strides[-1]),
            ),
            operation_order=(
                [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "cross_gnn",
                    "norm",
                    "ffn",
                    "norm",
                ] * 3 +
                [
                    "refine",
                ]
            ),
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            graph_model=dict(
                #type="MultiheadFlashAttention",
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            cross_graph_model=dict(
                #type="MultiheadFlashAttention",
                type="MultiheadAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 2,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            refine_layer=dict(
                type="MotionPlanningRefinementModule",
                embed_dims=embed_dims,
                fut_ts=fut_ts,
                fut_mode=fut_mode,
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),
            motion_sampler=dict(
                type="MotionTarget",
            ),
            motion_loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2
            ),
            motion_loss_reg=dict(type='mmdet.L1Loss', loss_weight=0.2),
            planning_sampler=dict(
                type="PlanningTarget",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),
            plan_loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.5,
            ),
            plan_loss_reg=dict(type='mmdet.L1Loss', loss_weight=1.0),
            plan_loss_status=dict(type='mmdet.L1Loss', loss_weight=1.0),
            motion_decoder=dict(type="SparseBox3DMotionDecoder"),
            planning_decoder=dict(
                type="HierarchicalPlanningDecoder",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
                use_rescore=True,
            ),
            num_det=50,
            num_map=10,
        ),
    ),
)

# ================== data ========================

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    # Please confirm if it is required!
    dict(type='Sparse4DLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type="ResizeCropFlipImage", data_aug_conf=data_aug_conf, training=True),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation", data_aug_conf=data_aug_conf, training=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=num_sample,
        permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type='Pack3DDetInputs',
         keys=['img', 'gt_bboxes_3d','gt_labels_3d'],
         meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
                    'lidar2ego', 'lidar2img', 'ego2global', 'lidar2global',
                    'extrinsics', 'ori_cam2img',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                    'scene_token', 'sample_idx',
                    'gt_bboxes_3d','gt_labels_3d',
                    'projection_mat', 'image_wh', 'gt_depth',
                    'gt_map_labels', 'gt_map_pts', 'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                    'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status',
                    'T_global', 'T_global_inv',
                    'cam_intrinsic', 'focal', 'timestamp', 'instance_id']),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage", data_aug_conf=data_aug_conf, training=False),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type='Pack3DDetInputs',
         keys=['img'],
         meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
                    'lidar2ego', 'lidar2img', 'ego2global', 'lidar2global',
                    'extrinsics', 'ori_cam2img',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 
                    'scene_token', 'sample_idx',
                    'projection_mat', 'image_wh', 'ego_status', 'gt_ego_fut_cmd',
                    'T_global', 'T_global_inv',
                    'cam_intrinsic', 'focal', 'timestamp']),
]
eval_pipeline = [
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=True,
        normalize=False,
    ),
    dict(type='Pack3DDetInputs',
         keys=['gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
                    'lidar2ego', 'lidar2img', 'ego2global', 'lidar2global',
                    'extrinsics', 'ori_cam2img',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 
                    'scene_token', 'sample_idx',
                    'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                    'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                    'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 
                    'fut_boxes', 'token', 'cam_intrinsic', 'focal', 'timestamp']),
]

input_modality = dict(
    use_lidar=True,  # Should be True for training
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_prefix=dict(
    pts='samples/LIDAR_TOP',
    sweeps='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT'
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    sampler=dict(type='GroupEachSampleInBatchSampler',
                 shuffle=True, sequence_flip_prob=0.1),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        modality=input_modality,
        metainfo=metainfo,
        ann_file='ad_infos/nuscenes_sparsedrive_ad_infos_train.pkl',
        data_prefix=data_prefix,
        with_seq_flag=True,
        sequences_split_num=2,
        test_mode=False,
        pipeline=train_pipeline,
        batch_size=batch_size, # Needed for GroupSampler
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        modality=input_modality,
        metainfo=metainfo,
        ann_file='ad_infos/nuscenes_sparsedrive_ad_infos_val.pkl',
        data_prefix=data_prefix,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_dataloader = test_dataloader



# ================== training ========================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=num_epochs,
        T_max=num_epochs,
        by_epoch=True)
]

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.5),
    }),
    clip_grad=dict(max_norm=25, norm_type=2))

train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=num_epochs)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=4, save_last=True))


# ================== eval ========================
eval_config = dict(
    batch_size=1,
    num_workers=1,
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        modality=input_modality,
        metainfo=metainfo,
        map_classes=map_class_names,
        ann_file='ad_infos/nuscenes_sparsedrive_ad_infos_val.pkl',
        data_prefix=data_prefix,
        test_mode=True,
        pipeline=eval_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='SparseDriveNuScenesMetric',
    data_root=data_root,
    ann_file=anno_root + 'nuscenes_sparsedrive_ad_infos_val.pkl',
    metric='bbox',
    with_det=True, # Assume always true
    with_tracking=True,
    with_map=True,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshold=0.2,
    eval_config=eval_config,
    backend_args=backend_args)
test_evaluator = val_evaluator


#load_from = 'checkpoints/sparsedrive/sparsedrive_stage2_update.pth'
load_from = None
resume = False
