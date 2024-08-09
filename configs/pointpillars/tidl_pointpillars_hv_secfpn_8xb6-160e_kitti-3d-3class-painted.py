_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.PointPillars.pointpillars'])

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size = [0.16, 0.16, 4]
model = dict(
    type='PPVoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000, 10000)   # (training, testing, onnx) max count of voxels
    )),
    voxel_encoder=dict(
        type='CustomPillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        point_color_dim=4,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        replace_mat_mul = True,
        feat_scale_fact = 32.0),
    middle_encoder=dict(
        type='CustomPointPillarsScatter', in_channels=64, output_shape=[496, 432], use_scatter_op=True),

    neck=dict(
        type='PPSECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],
        upsample_cfg=dict(type='nearest'))
        )
# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

# PointPillars adopted a different sampling strategies among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_point_painting_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='PPLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=[0, 1, 2, 3, 4, 5, 6, 7],
        color_dim = 4,
        use_color=True,
        backend_args=backend_args),
    backend_args=backend_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='PPLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=8,
        color_dim=4,
        use_color=True,
        backend_args=backend_args),
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
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]

eval_pipeline = [
    dict(
        type='PPLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=8,
        color_dim=4,
        use_color=True,
        backend_args=backend_args),
    dict(
        type='Pack3DDetInputs',
        class_names=class_names,
        with_label=False,
        keys=['points'])
]

test_pipeline = [
    dict(
        type='PPLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=8,
        color_dim=4,
        use_color=True,
        backend_args=backend_args),
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    dataset=dict(type='RepeatDataset',
                 times=2,
                 dataset=dict(pipeline=train_pipeline,  metainfo=metainfo, data_prefix=dict(pts='training/velodyne_painted_reduced'))))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline,  metainfo=metainfo, data_prefix=dict(pts='training/velodyne_painted_reduced')))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline,  metainfo=metainfo, data_prefix=dict(pts='training/velodyne_painted_reduced')))


save_onnx_model =True
quantize = False

runner_type = 'EdgeAIRunner'

if quantize == False:
    #momentum_config = dict(_delete_=True)
    lr = 1e-3
    optim_wrapper = dict(
        optimizer=dict(_delete_=True, type='SGD', lr=lr, momentum=0.9, weight_decay=1e-03))


    train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=10)
    val_cfg = dict()
    test_cfg = dict()

    evaluation = dict(interval=10,save_best='KITTI/Overall_3D_AP11_moderate',rule='greater')
    checkpoint_config = dict(interval=10)
    load_from = './work_dirs/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/epoch_80.pth'
else:
    lr = 1e-4

    optimizer = dict(_delete_=True, type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3)
    optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))
    warmup_cfg = dict(_delete_=True,warmup='linear', warmup_iters=2000, warmup_ratio=0.001)
    lr_config = dict(_delete_=True,
      policy='CosineAnnealing',
      min_lr_ratio=0.0001,
      warmup='linear',
      warmup_iters=2000,
      warmup_ratio=0.001)

    epoch_num = 20
    train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
    val_cfg = dict()
    test_cfg = dict()

    evaluation = dict(interval=1,save_best='KITTI/Overall_3D_AP11_moderate',rule='greater')
    checkpoint_config = dict(interval=1)

    #load_from = './work_dirs/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-painted/latest.pth'
    #work_dir = './work_dirs/3class_quant_train_dir_2/'
    #custom_hooks = dict(type='FreezeRangeHook')

val_evaluator = dict(    
    ann_file=data_root + 'kitti_point_painting_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator