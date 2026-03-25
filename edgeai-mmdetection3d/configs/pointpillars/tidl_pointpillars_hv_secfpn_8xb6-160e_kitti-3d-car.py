# model settings
_base_ = './pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car']
metainfo = dict(classes=class_names)
backend_args = None


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
        upsample_cfg=dict(type='nearest')),
    bbox_head=dict(
        type='PPAnchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),

    test_cfg=dict(
        use_rotate_nms=True, # set this to False to enable upright rectangle NMS
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15),
    points_loader=dict(
        type='PPLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='PPLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler,use_ground_plane=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
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
test_pipeline = [
    dict(
        type='PPLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
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
                 dataset=dict(pipeline=train_pipeline, metainfo=metainfo)))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))

save_onnx_model = True
quantize = False

runner_type = 'EdgeAIRunner'

if quantize == False:
    lr = 6e-3
    optim_wrapper = dict(
        optimizer=dict(_delete_=True, type='SGD', lr=lr, momentum=0.9, weight_decay=1e-03))

    #runner = dict(max_epochs=80)
    train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=10)

    evaluation = dict(interval=10,save_best='KITTI/Car_3D_AP11_moderate_strict',rule='greater')
    checkpoint_config = dict(interval=10)
else:
    lr = 6e-4

    optim_wrapper = dict(
        optimizer=dict(_delete_=True, type='SGD', lr=lr, momentum=0.9, weight_decay=1e-03),
        optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2)),
        warmup_cfg = dict(_delete_=True,warmup='linear', warmup_iters=2000, warmup_ratio=0.001),
        lr_config = dict(_delete_=True,
                          policy='CosineAnnealing',
                          min_lr_ratio=0.0001,
                          warmup='linear',
                          warmup_iters=2000,
                          warmup_ratio=0.001)
        )


    #optimizer = dict(_delete_=True, type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3)
    #optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))
    #warmup_cfg = dict(_delete_=True,warmup='linear', warmup_iters=2000, warmup_ratio=0.001)
    #lr_config = dict(_delete_=True,
    #  policy='CosineAnnealing',
    #  min_lr_ratio=0.0001,
    #  warmup='linear',
    #  warmup_iters=2000,
    #  warmup_ratio=0.001)

    train_cfg = dict(max_epochs=20)

    evaluation = dict(interval=1,save_best='KITTI/Car_3D_AP11_moderate_strict',rule='greater')
    checkpoint_config = dict(interval=1)

    #load_from = './work_dirs/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car/latests.pth'
    #work_dir  = './work_dirs/car_cls_quant_train_dir_2/'
    
    #custom_hooks = dict(type='FreezeRangeHook')
