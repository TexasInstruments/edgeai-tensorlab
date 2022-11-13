_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size = [0.16, 0.16, 4]
model = dict(
    voxel_layer=dict(
        max_num_points=32,  # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000, 10000)  # (training, testing, onnx) max count of voxels
    ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        replace_mat_mul = True,
        feat_scale_fact = 32.0),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432], use_scatter_op=True),

    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],
        upsample_cfg=dict(type='nearest'))
        )
# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
# PointPillars adopted a different sampling strategies among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15))

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,

    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

save_onnx_model = False
quantize = True

if quantize == False:
    lr = 0.001
    optimizer = dict(lr=lr)
    # max_norm=35 is slightly better than 10 for PointPillars in the earlier
    # development of the codebase thus we keep the setting. But we does not
    # specifically tune this parameter.
    optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    # PointPillars usually need longer schedule than second, we simply double
    # the training schedule. Do remind that since we use RepeatDataset and
    # repeat factor is 2, so we actually train 160 epochs.
    runner = dict(max_epochs=100)

    # Use evaluation interval=2 reduce the number of evaluation timese
    evaluation = dict(interval=2)
else:
    lr = 0.0005
    optimizer = dict(lr=lr)

    # max_norm=35 is slightly better than 10 for PointPillars in the earlier
    # development of the codebase thus we keep the setting. But we does not
    # specifically tune this parameter.
    optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

    # PointPillars usually need longer schedule than second, we simply double
    # the training schedule. Do remind that since we use RepeatDataset and
    # repeat factor is 2, so we actually train 160 epochs.
    runner = dict(max_epochs=100)

    # Use evaluation interval=2 reduce the number of evaluation timese
    evaluation = dict(interval=1)

    load_from = './work_dirs/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_nearest/latest.pth'
    work_dir = './work_dirs/quant_train_dir_nearest/'
    custom_hooks = dict(type='FreezeRangeHook')

