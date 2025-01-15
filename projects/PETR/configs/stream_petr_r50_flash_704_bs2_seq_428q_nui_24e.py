_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/cyclic-20e.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
custom_imports = dict(imports=['projects.PETR.petr'])

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
# To double check
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

metainfo = dict(classes=class_names)

batch_size = 4
num_epochs = 24

queue_length = 1
num_frame_losses = 1
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='PETR3D',
    save_onnx_model=False,
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    data_preprocessor=dict(
        type='Petr3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    use_grid_mask=True,
    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint="pretrained/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth",
            prefix='backbone.'),
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    img_roi_head=dict(
        type='FocalHead',
        num_classes=10,
        in_channels=256,
        loss_cls2d=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='mmdet.L1Loss', loss_weight=10.0),
        train_cfg=dict(
        assigner2d=dict(
            type='HungarianAssigner2D',
            cls_cost=dict(type='FocalLossCost', weight=2.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))
        ),
    pts_bbox_head=dict(
        type='StreamPETRHead',
        num_classes=10,
        in_channels=256,
        num_query=300,
        memory_len=512,
        topk_proposals=128,
        num_propagated=128,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            #type='PETRMultiheadFlashAttention',
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
            model='PETR3D'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
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
                model='PETR3D'))))


dataset_type = 'StreamNuScenesDataset'
data_root = './data/nuscenes/'
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
    dict(type='StreamPETRLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True,
            ),
    dict(
        type='CustomPack3DDetInputs',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_bboxes_labels', 'centers_2d',
              'depths'],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d', 'sample_idx',
                   'prev_exists', 'gt_bboxes', 'gt_bboxes_labels', 'centers_2d', 'depths'] + collect_keys)
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
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
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))

test_dataloader = dict(
    # Inference does not support batch_size > 1
    batch_size=1,
    num_workers=1,
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file='nuscenes_strpetr_infos_val.pkl',
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
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file='nuscenes_strpetr_infos_val.pkl',
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
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))


val_evaluator = dict(
    type='CustomNuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_strpetr_infos_val.pkl',
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

#load_from = './checkpoints/streampetr/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth'
resume_from = None
