_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/cyclic-20e.py'
]

custom_imports = dict(imports=['projects.PETR.petr'])

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

metainfo = dict(classes=class_names, version='v1.0-mini')

batch_size = 1
num_epochs = 24

queue_length = 1
num_frame_losses = 1
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
depthnet_config = {'type': 0, 'hidden_dim': 256, 'num_depth_bins': 50, 'depth_min': 1e-1, 'depth_max': 110, 'stride': 8}

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

model = dict(
    type='Far3D',
    save_onnx_model=True,
    use_grid_mask=True,
    stride=[8, 16, 32, 64],
    position_level=[0, 1, 2, 3],
    data_preprocessor=dict(
        type='Petr3DDataPreprocessor',
        #mean=[103.530, 116.280, 123.675],
        #std=[57.375, 57.120, 58.395],
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    img_backbone=dict(
        type='VoVNet', ###use checkpoint to save memory
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage2','stage3','stage4','stage5',)),
    img_neck=dict(
        type='mmdet.FPN',  ###remove unused parameters 
        start_level=1,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        num_outs=4),
    img_roi_head=dict(
        type='YOLOXHeadCustom',
        num_classes=10,
        in_channels=256,
        strides=[8, 16, 32, 64],
        train_cfg=dict(assigner=dict(type='mmdet.SimOTAAssigner', 
                                     center_radius=2.5, 
                                     iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
        pred_with_depth=True,
        depthnet_config=depthnet_config,
        reg_depth_level='p3',
        pred_depth_var=False,    # note 2d depth uncertainty
        loss_depth2d=dict(type='mmdet.L1Loss', loss_weight=1.0),
        sample_with_score=True,  # note threshold
        threshold_score=0.1,
        topk_proposal=None,
        return_context_feat=True,
    ),
    pts_bbox_head=dict(
        type='FarHead',
        num_classes=10,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        scalar=10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        offset=0.5,
        offset_p=0.0,
        num_smp_per_gt=3,
        with_dn=True,
        with_ego_pos=True,
        add_query_from_2d=True,
        pred_box_var=False,  # note add box uncertainty
        depthnet_config=depthnet_config,
        train_use_gt_depth=True,
        add_multi_depth_proposal=True,
        multi_depth_config={'topk': 1, 'range_min': 30,},  # 'bin_unit': 1, 'step_num': 4,
        return_bbox2d_scores=True,
        return_context_feat=True,
        code_size=10,
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='Detr3DTransformer',
            decoder=dict(
                type='Detr3DTransformerDecoder',
                embed_dims=256,
                num_layers=6,
                transformerlayers=dict(
                    type='Detr3DTemporalDecoderLayer',
                    batch_first=True,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='DeformableFeatureAggregationCuda', 
                            embed_dims=256,
                            num_groups=8,
                            num_levels=4,
                            num_cams=6,
                            dropout=0.1,
                            num_pts=13,
                            bias=2.),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
            model='Far3D'), 
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
                model='Far3D'))))


dataset_type = 'StreamNuScenesDataset'
data_root = './data/nuscenes/'
backend_args = None

file_client_args = dict(backend='disk')

ida_aug_conf = {
        "resize_lim": (0.47, 0.55),
        "final_dim": (640, 960),
        "final_dim_f": (640, 720),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }
train_pipeline = [
    dict(type='AV2LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='StreamPETRLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='AV2ResizeCropFlipRotImageV2', data_aug_conf=ida_aug_conf),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='AV2PadMultiViewImage', size='same2max'),
    dict(type='AV2DownsampleQuantizeInstanceDepthmap', downsample=depthnet_config['stride'], depth_config=depthnet_config),
    dict(
        type='CustomPack3DDetInputs',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_bboxes_labels', 'centers_2d',
              'depths'],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d', 'sample_idx',
                   'prev_exists', 'gt_bboxes', 'gt_bboxes_labels', 'centers_2d', 'depths'] + collect_keys)
]
test_pipeline = [
    #dict(type='AV2LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    #dict(type='AV2ResizeCropFlipRotImageV2', data_aug_conf=ida_aug_conf),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    #dict(type='AV2PadMultiViewImage', size='same2max'),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='CustomMultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Pack3DDetInputs',  keys=['img'], meta_keys=['filename', 'ori_shape',
                 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 
                 'scene_token', 'sample_idx'] + collect_keys)
        ])
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='nuscenes_strpetr_infos_train.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        #collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        collect_keys=collect_keys + ['img', 'prev_exists'],
        queue_length=queue_length,
        filter_empty_gt=False,
        pipeline=train_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=False,
        use_valid_flag=False,
        modality=input_modality,
        backend_args=backend_args))

test_dataloader = dict(
    # Inference does not support batch_size > 1
    batch_size=1,
    num_workers=1,
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file='nuscenes_mini_strpetr_infos_val.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        num_frame_losses=num_frame_losses,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        #filter_empty_gt=False,
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        use_valid_flag=True,
        modality=input_modality,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file='nuscenes_mini_strpetr_infos_val.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        num_frame_losses=num_frame_losses,
        #seq_split_num=2, # streaming video training
        #seq_mode=True, # streaming video training
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        #filter_empty_gt=False,
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        use_valid_flag=True,
        modality=input_modality,
        backend_args=backend_args))


val_evaluator = dict(
    type='CustomNuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_mini_strpetr_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator


optim_wrapper = dict(
    # TODO Add Amp
    # type='AmpOptimWrapper',
    # loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=1000,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        # TODO Figure out what T_max
        begin=0,
        end=num_epochs,
        T_max=num_epochs,
        by_epoch=True,
    )
]

#evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
train_cfg = dict(max_epochs=num_epochs, val_interval=num_epochs)
find_unused_parameters = True #### when use checkpoint, find_unused_parameters must be False
#checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
#runner = dict(
#    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=4, save_last=True))

load_from='pretrained/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from = None
