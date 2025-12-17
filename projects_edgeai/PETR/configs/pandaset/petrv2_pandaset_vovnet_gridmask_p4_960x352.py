_base_ = [
    'mmdet3d::_base_/datasets/pandaset-3d-3classes.py',
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/schedules/cyclic-20e.py'
]

backbone_norm_cfg = dict(type='LN', requires_grad=True)
custom_imports = dict(imports=['projects_edgeai.PETR.petr',
                               'projects_edgeai.edgeai_mmdet3d'])


randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
# Not used. Can be removed
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    bgr_to_rgb=False)
# 3 classes
class_names = [
    'Car','Pedestrian','Temporary Construction Barriers'
]

class_mapping = [
    0,2,2,1,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,1,2,2,2,
]
metainfo = dict(classes=class_names, class_mapping=class_mapping)

# If True, reuse image feature map from the previous frame
# ONNX export is supported for optimized_inference = True
optimized_inference=True

sweep_range=[1,2]

input_modality = dict(use_camera=True)
model = dict(
    type='PETR',
    version='v2',
    img_feat_size = [[6, 256, 22, 60], [6, 256, 11, 30]],
    save_onnx_model=False,
    optimized_inference=optimized_inference,
    sweep_range=sweep_range,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        **img_norm_cfg,
        pad_size_divisor=32),
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=(
            'stage4',
            'stage5',
        )),
    img_neck=dict(
        type='CPFPN', in_channels=[768, 1024], out_channels=256, num_outs=2),
    pts_bbox_head=dict(
        type='PETRv2Head',
        num_classes=3,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        with_fpe=True,
        with_time=True,
        with_multi=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
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
            num_classes=3),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
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
                iou_cost=dict(
                    type='IoUCost', weight=0.0
                ),  # Fake cost. Just to be compatible with DETR head.
                pc_range=point_cloud_range))))

dataset_type = 'PETRv2PandaSetDataset'
data_root = 'data/pandaset/'
backend_args = None

"""
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'pandaset_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points={
            'Car':5,
            'Pedestrian':5,
            'Temporary Construction Barriers':5
            }),
    classes=class_names,
    sample_groups={
            'Car':2,
            'Pedestrian':2,
            'Temporary Construction Barriers':2
            },
    points_loader=dict(
        type='LoadPointsFromFile', # TODO PKL FILE ADDING
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args),
    backend_args=backend_args)
"""

ida_aug_conf = {
    'resize_lim': (0.47, 0.625),
    'final_dim': (352, 960),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1080,
    'W': 1920,
    'rand_flip': True,
}
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=6,
        backend_args=backend_args),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=1,
        to_float32=True,
        pad_empty_sweeps=True,
        test_mode=False,
        sensors=['front_camera', 'front_right_camera', 'front_left_camera', 'back_camera', 'left_camera', 'right_camera'],
        sweep_range=sweep_range),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=False,
        training=True),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d'], 
        meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img', 'cam2img', 'lidar2cam',
            'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
            'img_norm_cfg', 'sample_idx', 'timestamp', 'delta_timestamp'])
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        backend_args=backend_args),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=1,
        to_float32=True,
        pad_empty_sweeps=True,
        optimized_inference=optimized_inference,
        sweep_range=sweep_range,
        sensors=['front_camera', 'front_right_camera', 'front_left_camera', 'back_camera', 'left_camera', 'right_camera']
        ),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf,
        training=False),
    dict(type='Pack3DDetInputs', keys=['img'], meta_keys=['filename', 'ori_shape', 'img_shape', 'cam2img', 'lidar2cam',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'scene_token',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'delta_timestamp', 'img_timestamp', 'ego2global', 'lidar2ego'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file='pandaset_long_infos_train.pkl',
        data_prefix=dict(
            pts='lidar',
            front_camera='camera/front_camera',
            front_left_camera='camera/front_left_camera',
            front_right_camera='camera/front_right_camera',
            back_camera='camera/back_camera',
            left_camera='camera/left_camera',
            right_camera='camera/right_camera'),
        pipeline=train_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=False,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file='pandaset_long_infos_val.pkl',
        data_prefix=dict(
            pts='lidar',
            front_camera='camera/front_camera',
            front_left_camera='camera/front_left_camera',
            front_right_camera='camera/front_right_camera',
            back_camera='camera/back_camera',
            left_camera='camera/left_camera',
            right_camera='camera/right_camera'),
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file='pandaset_long_infos_val.pkl',
        data_prefix=dict(
            pts='lidar',
            front_camera='camera/front_camera',
            front_left_camera='camera/front_left_camera',
            front_right_camera='camera/front_right_camera',
            back_camera='camera/back_camera',
            left_camera='camera/left_camera',
            right_camera='camera/right_camera'),
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))


val_evaluator = dict(
    type='SortedPandaSetMetric',
    data_root=data_root,
    ann_file=data_root + 'pandaset_long_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# Optimizer
# lr=0.0002 for num_gpus*batch_size = 4
# Note: lr=0.0002 with num_gpus*batch_size = 8 (e.g. num_gpus=4, batch_size=2)
#       may improve accuracy 
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2))

num_epochs = 24

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        T_max=num_epochs,
        by_epoch=True,
    )
]

train_cfg = dict(max_epochs=num_epochs, val_interval=num_epochs)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=2, save_last=True))

find_unused_parameters = False

load_from = './checkpoints/petrv2/petrv2_keyframe_epoch_24_20250828.pth'
resume = False
