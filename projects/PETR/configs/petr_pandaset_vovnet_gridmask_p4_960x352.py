_base_ = [
    '../../../configs/_base_/datasets/pandaset-3d-3classes.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/cyclic-20e.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
custom_imports = dict(imports=['projects.PETR.petr'])

randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False)

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

input_modality = dict(use_camera=True)
model = dict(
    type='PETR',
    save_onnx_model=False,
    imgfeat_size=[22, 60],
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        bgr_to_rgb=False,
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
        type='PETRHead',
        num_classes=3,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
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

dataset_type = 'PandaSetDataset'
data_root = 'data/pandaset/'
backend_args = None

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
        backend_args=backend_args),
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
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ])
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        backend_args=backend_args),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf,
        training=False),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            pts='',
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
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            pts='',
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
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            pts='',
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

# Different from original PETR:
# We don't use special lr for image_backbone
# This seems won't affect model performance
optim_wrapper = dict(
    # TODO Add Amp
    # type='AmpOptimWrapper',
    # loss_scale='dynamic',
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
        # TODO Figure out what T_max
        T_max=num_epochs,
        by_epoch=True,
    )
]

train_cfg = dict(max_epochs=num_epochs, val_interval=num_epochs)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=4, save_last=True))

find_unused_parameters = False

# pretrain_path can be found here:
# https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view
load_from = './pretrained/fcos3d_vovnet_imgbackbone-remapped.pth'
resume = False

