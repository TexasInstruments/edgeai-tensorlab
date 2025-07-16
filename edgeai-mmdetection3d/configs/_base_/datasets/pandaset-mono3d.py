dataset_type = 'PandaSetDataset'
data_root = 'data/pandaset/'
class_names = [
            'Car', 'Semi-truck', 'Other Vehicle - Construction Vehicle', 'Pedestrian with Object', 
            'Train', 'Animals - Bird', 'Bicycle', 'Rolling Containers', 'Pylons', 'Signs', 
            'Emergency Vehicle', 'Towed Object', 'Personal Mobility Device', 'Motorcycle', 
            'Tram / Subway', 'Other Vehicle - Uncommon', 'Other Vehicle - Pedicab', 
            'Temporary Construction Barriers', 'Animals - Other', 'Bus', 'Motorized Scooter', 
            'Pickup Truck', 'Road Barriers', 'Pedestrian', 'Construction Signs', 'Cones', 'Medium-sized Truck'
        ]
metainfo = dict(classes=class_names) # full
# metainfo = dict(classes=class_names, version='v1.0-mini') # mini
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(use_lidar=False, use_camera=True)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/nuscenes/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize3D', scale=(1920, 1080), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='Resize3D', scale=(1920, 1080), keep_ratio=True),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            pts='lidar', 
            back_camera='camera/back_camera', 
            front_camera='camera/front_camera', 
            front_left_camera='camera/front_left_camera', 
            front_right_camera='camera/front_right_camera', 
            left_camera='camera/left_camera', 
            right_camera='camera/right_camera'
            ),
        ann_file='pandaset_infos_train.pkl',
        # ann_file='pandaset_mini_infos_val.pkl', # mini
        load_type='mv_image_based',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='Camera' in monocular 3d
        # detection task
        box_type_3d='Camera',
        use_valid_flag=True,
        max_dist_thr = 50,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix = dict(
            pts='lidar', 
            back_camera='camera/back_camera', 
            front_camera='camera/front_camera', 
            front_left_camera='camera/front_left_camera', 
            front_right_camera='camera/front_right_camera', 
            left_camera='camera/left_camera', 
            right_camera='camera/right_camera'
            ),
        ann_file='pandaset_infos_val.pkl', 
        # ann_file='pandaset_mini_infos_val.pkl', #mini
        load_type='mv_image_based',
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Camera',
        use_valid_flag=True,
        max_dist_thr = 50,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='PandaSetMetric',
    data_root=data_root,
    max_dists = 50,
    ann_file=data_root + 'pandaset_infos_val.pkl', 
    # ann_file=data_root + 'pandaset_mini_infos_val.pkl', # mini
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
