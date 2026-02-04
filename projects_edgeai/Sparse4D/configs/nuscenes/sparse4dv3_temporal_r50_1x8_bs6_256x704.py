_base_ = [
    'mmdet3d::_base_/datasets/nus-3d.py',
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/schedules/cyclic-20e.py'
]

custom_imports = dict(imports=['projects_edgeai.Sparse4D.sparse4d',
                               'projects_edgeai.edgeai_mmdet3d'])

# ================ base config ===================

#total_batch_size = 48
#num_gpus = 8
#batch_size = total_batch_size // num_gpus
#num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
#checkpoint_epoch_interval = 20
num_epochs = 100
batch_size = 2

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

tracking_test = True
tracking_threshold = 0.2

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

metainfo = dict(classes=class_names)

num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
# use_deformable_func should be True for training to save memory
# use_deformable_func should be False for ONNX export while inferencing
# For use_deformable_func=True, mmdet3d_plugin/ops/setup.py needs to be executed
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
decouple_attn = True
with_quality_estimation = True

dataset_type = "Sparse4DNuScenesDataset"
data_root = "data/nuscenes/"
backend_args = None

model = dict(
    type="Sparse4D",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    save_onnx_model=False,
    onnx_subnets=False,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        **img_norm_cfg,
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
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor=data_root+"nuscenes_kmeans900.npy",
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
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        )
        if temporal
        else None,
        graph_model=dict(
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
            num_dn_groups=5,
            num_temp_dn_groups=3,
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
)

# ================== data ========================
data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
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
    dict(type='Sparse4DLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type="ResizeCropFlipImage", data_aug_conf=data_aug_conf, training=True),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation", data_aug_conf=data_aug_conf, training=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type='Pack3DDetInputs',
         keys=['img', 'gt_bboxes_3d','gt_labels_3d'],
         meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
                    'lidar2ego', 'lidar2img', 'ego2global', 'lidar2global',
                    'extrinsics', 'ori_cam2img',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                    'scene_token', 'sample_idx',
                    'gt_bboxes_3d','gt_labels_3d',
                    'projection_mat', 'image_wh', 'gt_depth', 'T_global', 'T_global_inv',
                    'cam_intrinsic', 'focal', 'timestamp', 'instance_id']),

]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage", data_aug_conf=data_aug_conf, training=False),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type='Pack3DDetInputs',
         keys=['img'],
         meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
                    'lidar2ego', 'lidar2img', 'ego2global', 'lidar2global',
                    'extrinsics', 'ori_cam2img',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 
                    'scene_token', 'sample_idx',
                    'projection_mat', 'image_wh', 'T_global', 'T_global_inv',
                    'cam_intrinsic', 'focal', 'timestamp']),
]

input_modality = dict(
    use_lidar=True,  # Should be True for Training 
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
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
        ann_file='nuscenes_sparse4d_infos_train.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
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
        ann_file='nuscenes_sparse4d_infos_val.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
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
#vis_pipeline = [
#    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
#    dict(
#        type="Collect",
#        keys=["img"],
#        meta_keys=["timestamp", "lidar2img"],
#    ),
#]

#evaluation = dict(
#    interval=num_iters_per_epoch * checkpoint_epoch_interval,
#    pipeline=vis_pipeline,
#    # out_dir="./vis",  # for visualization
#)

val_evaluator = dict(
    type='Sparse4DNuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_sparse4d_infos_val.pkl',
    metric='bbox',
    tracking=tracking_test,
    tracking_threshold=tracking_threshold,
    backend_args=backend_args)
test_evaluator = val_evaluator


#load_from = 'checkpoints/sparse4d/sparse4dv3_r50_statedict_update.pth'
load_from = None
resume = False
