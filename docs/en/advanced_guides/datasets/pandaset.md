# PandaSet Dataset

This page provides specific tutorials about the usage of **EdgeAI-MMDetection3D** for PandaSet dataset.

Pandaset Dataset Support for MMDetection3D is based on nuScenes Dataset. Please, read through [nuscenes.md](nuscenes.md) for basic information.

## Before Preparation
You can download PandaSet 3D detection `Full dataset (v1.0)` [HERE](https://pandaset.org/) and unzip all zip files. 

Like the general way to prepare dataset, it is recommended to symlink the dataset root to `edgeai-mmdetection3d/data`. The folder structure in `edgeai-mmdetection3d/data/pandaset/data` should be organized as follows before our processing.

```
├── LICENSE.txt
├── annotations
│   ├── cuboids
│   │   ├── 00.pkl.gz
│   │   .
│   │   .
│   │   .
│   │   └── 79.pkl.gz
│   └── semseg  // Semantic Segmentation is available for specific scenes
│       ├── 00.pkl.gz
│       .
│       .
│       .
│       ├── 79.pkl.gz
│       └── classes.json
├── camera
│   ├── back_camera
│   │   ├── 00.jpg
│   │   .
│   │   .
│   │   .
│   │   ├── 79.jpg
│   │   ├── intrinsics.json
│   │   ├── poses.json
│   │   └── timestamps.json
│   ├── front_camera
│   │   └── ...
│   ├── front_left_camera
│   │   └── ...
│   ├── front_right_camera
│   │   └── ...
│   ├── left_camera
│   │   └── ...
│   └── right_camera
│       └── ...
├── lidar
│   ├── 00.pkl.gz
│   .
│   .
│   .
│   ├── 79.pkl.gz
│   ├── poses.json
│   └── timestamps.json
└── meta
    ├── gps.json
    └── timestamps.json
```
## Dataset Preparation

After downloading the dataset to unzip all `.pkl.gz` files to `.pkl` files, we have to run the following command:
```bash
tools/dataset_converters/pandaset_unzip.sh ./data/pandaset/data
```

We typically need to organize the useful data information with a `.pkl` file in a specific style.
To prepare these files for pandaset, run the following command:

```bash
python tools/create_data.py pandaset --root-path ./data/pandaset/data --out-dir ./data/pandaset/data --extra-tag pandaset
```

<!-- This command creates `.pkl` files for PETR, BEVFormer and FCOS3D. To include additional data fields for BEVDet and PETRv2, we should add `--bevdet` and `--petrv2`, respectively, to the command. For example,

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --canbus ./data --bevdet --petrv2
```

FastBEV uses multiple temporal frames and therefore need to organize neighboring frames's information as well in a `.pkl` file for training. For this purpose, we should run the following script, which will create `nuscenes_infos_train_fastbev.pkl` from `nuscenes_infos_train.pkl`.

```bash
python tools/dataset_converters/generate_fastbev_sweep_pkl.py n --root-path ./data/pandaset --version 'v1.0-trainval'
``` -->

The folder structure in `edgeai-mmdetection3d/data/pandaset/data` after processing should be as below .
```
├── LICENSE.txt
├── annotations
│   ├── cuboids
│   │   ├── 00.pkl
│   │   ├── 00.pkl.gz
│   │   .
│   │   .
│   │   .
│   │   ├── 79.pkl
│   │   └── 79.pkl.gz
│   └── semseg  // Semantic Segmentation is available for specific scenes
│   │   ├── 00.pkl
│   │   ├── 00.pkl.gz
│   │   .
│   │   .
│   │   .
│   │   ├── 79.pkl
│       ├── 79.pkl.gz
│       └── classes.json
├── camera
│   ├── back_camera
│   │   ├── 00.jpg
│   │   .
│   │   .
│   │   .
│   │   ├── 79.jpg
│   │   ├── intrinsics.json
│   │   ├── poses.json
│   │   └── timestamps.json
│   ├── front_camera
│   │   └── ...
│   ├── front_left_camera
│   │   └── ...
│   ├── front_right_camera
│   │   └── ...
│   ├── left_camera
│   │   └── ...
│   └── right_camera
│       └── ...
├── lidar
│   │   ├── 00.pkl
│   │   ├── 00.pkl.gz
│   │   .
│   │   .
│   │   .
│   │   ├── 79.pkl
│   ├── 79.pkl.gz
│   ├── poses.json
│   └── timestamps.json
├── meta
|   ├── gps.json
|   └── timestamps.json
├── pandaset_infos_train.pkl
└── pandaset_infos_val.pkl
```
<!-- ├── pandaset_infos_train_fastbev.pkl -->

- `pandaset_infos_train.pkl`: training dataset, a dict with two keys, `metainfo` and `data_list`   
`metainfo` and `data_list`.
  `metainfo` contains the basic information for the dataset itself, such as `categories`, `dataset` and `version`, 
- while `data_list` is a list of dict, each dict (hereinafter referred to as `info`) contains all the detailed information of single sample as follows:
  - info\[`token`\]: Sample data token.
  - info\[`timestamp`\]: Timestamp of the sample data.
  - info\[`ego2global`\]: The transformation matrix from the ego vehicle to global coordinates. (4x4 list)
  - info\[`images`\]: a dict containing information related to camera and images with camera_names being the keys and their corresponding values are dicts with following information:
    - `sample_data_token`: camera token
    - `img_path`: path to the image
    - `cam2img`: intrinsic matrix for camera to image
    - `timestamp`: timestamp of the image
    - `cam2ego`: transformation matrix from camera to ego vehicle
    - `lidar2cam`: transformation matrix from lidar to camera
    - `cam2global`: transformation matrix from camera to global
  - info\[`lidar_points`\]: a dict containing information for lidar data as following:
    - `num_pts_feats`: set 4 for pandaset
    - `lidar_path`: path to the lidar data
    - `lidar2ego`: transformation matrix from lidar to ego vehicle
    - `lidar2global`: transformation matrix from lidar to global
    - `timestamp`: timestamp of the lidar data
  - info\[`frame_idx`\]: The index of the frame in the whole dataset.
  - info\[`prev`\]: The token of the previous frame in the whole dataset.
  - info\[`next`\]: The token of the next frame in the whole dataset.
  - info\[`scene_token`\]: The token of the scene in the whole dataset.
  - info\[`gps`\]: The GPS data of the ego vehicle.
  - info\[`can_bus`\]: The CAN bus data (`zeros`).
  - info\[`instances`\]: a list of bbox data (dict) for each bbox in lidar coordinates for each frame
  - info\[`cam_instances`\]: a dict mapping camera_names to list of bbox data (dict) for each visible bbox in that camera coordinates for each frame
    - bbox data contains the following in different keys
      - `token`: token of the bbox
      - `bbox_label`: label of the bbox
      - `bbox_label_3d`: label of the bbox
      - `bbox_3d`: bbox in corresponding coordinates
      - `bbox_3d_isvalid`: whether the bbox is valid
      - `bbox_3d_isstationary`: whether the bbox is stationary
      - `num_lidar_pts`: number of lidar points included in the bbox
      - `velocity`: velocity of the bbox in the corresponding coordinateAs
      - `world_bbox3d`: bbox in global coordinates
      - `world_velocity`: velocity of the bbox in global coordinates
      - `attr_label`: attribute label of the bbox
      - `center2d`: projected center location on the image
      - `depth`: depth of projected center
      - `bbox`: 2D bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as \[x1, y1, x2, y2\]
  
  - camera_names are as followings:
    - `'front_camera'`
    - `'left_camera'`
    - `'front_right_camera'`
    - `'front_left_camera'`
    - `'right_camera'`
    - `'back_camera'`

Note:

1. The differences between `bbox_3d` in `instances` and that in `cam_instances`.
   Both `bbox_3d` have been converted to MMDet3D coordinate system, but `bboxes_3d` in `instances` is in LiDAR coordinate format and `bboxes_3d` in `cam_instances` is in Camera coordinate format. Mind the difference between them in 3D Box representation ('l, w, h' and 'l, h, w').

2. Here we only explain the data recorded in the training info files. The same applies to validation and testing set (the `.pkl` file of test set does not contains `instances` and `cam_instances`).

The function to generate `pandaset_info_xxx.pkl` file is [create_pandaset_infos](../../../../tools/dataset_converters/pandaset_converter.py#create_pandaset_infos) which is based on [\_fill_trainval_infos](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/nuscenes_converter.py#L146) which is for NuScenes.

In this function [pandaset-devkit](https://github.com/scaleapi/pandaset-devkit) is used to get the details from the files to fill the information in `pandaset_infos_xxx.pkl`.

Note:
- Originally, in pandaset-devkit's cuboid annotations, direction of front to back of the vehicles are along y axis, so we have to change the dimension with x axis and adjust the yaw accordingly to match the MMDet3D coordinate system. For more details, please refer to [PandaSet](https://arxiv.org/abs/2112.12610) paper

From the pandaset-devkit, we can get the following information for each frame of a scene:
    - `cam2global`: transformation matrix from camera to global
    - `lidar2global`: transformation matrix from lidar to global
    - `timestamp`: timestamp of the lidar data
    - `image paths`: paths to the images
    - `lidar paths`: paths to the lidar data files
    - `cam2img`: transformation matrix from camera to image
    - `cuboids`: annotations of the cuboids, etc.

The above information can be used for deriving information in `pandaset_infos_xxx.pkl`.

Unlike NuScenes which stores lidar data and annotations in `.bin` files, instead of that PandaSet stores the lidar data and annotations in `.pkl` files. So in dataloader pipeline, we should take data from `.pkl` files.

## Training pipeline

### LiDAR-Based Methods

A typical training pipeline of LiDAR-based 3D detection (including multi-modality methods) on  is as below. No LiDAR-Based network hasn't been tested in **EdgeAI-MMDetection3D** though.

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

<!-- Compared to general cases, nuScenes has a specific `'LoadPointsFromMultiSweeps'` pipeline to load point clouds from consecutive frames. This is a common practice used in this setting. -->
<!-- Please refer to the nuScenes [original paper](https://arxiv.org/abs/1903.11027) for more details.
The default `use_dim` in `'LoadPointsFromMultiSweeps'` is `[0, 1, 2, 4]`, where the first 3 dimensions refer to point coordinates and the last refers to timestamp differences. -->
Intensity is not used by default due to its yielded noise when concatenating the points from different frames.

### Vision-Based Methods

#### Monocular-based

In the PandaSet dataset, for multi-view images, this paradigm usually involves detecting and outputting 3D object detection results separately for each image, and then obtaining the final detection results through post-processing (such as NMS). Essentially, it directly extends monocular 3D detection to multi-view settings. A typical training pipeline of image-based monocular 3D detection on PandaSet is as below.

```python
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='mmdet.Resize', scale=(1920, 1080), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
```

It follows the general pipeline of 2D detection while differs in some details:

- It uses monocular pipelines to load images, which includes additional required information like camera intrinsics.
- It needs to load 3D annotations.
- Some data augmentation techniques need to be adjusted, such as `RandomFlip3D`.
  Currently we do not support more augmentation methods, because how to transfer and apply other techniques is still under explored.

#### BEV-based

BEV, Bird's-Eye-View, is another popular 3D detection paradigm. It directly takes multi-view images to perform 3D detection, for PandaSet, they are `front_camera`, `front_left_camera`, `front_right_camera`, `back_camera`, `left_camera` and `right_camera`. A basic training pipeline of bev-based 3D detection on PandaSet is as below.

```python
class_names = class_names = [
    'Car','Pedestrian','Temporary Construction Barriers'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
train_transforms = [
    dict(type='PhotoMetricDistortion3D'),
    dict(
        type='RandomResize3D',
        scale=(1920, 1080),
        ratio_range=(1., 1.),
        keep_ratio=True)
]
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=True,
         num_views=6, ),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    # optional, data augmentation
    dict(type='MultiViewWrapper', transforms=train_transforms),
    # optional, filter object within specific point cloud range
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # optional, filter object of specific classes
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

Note:
- Originally, PandaSet has 27 classes in the raw data. As only `Car` and `Pedestrian` are used in this setting, we only use `Car` and `Pedestrian` in the training pipeline and merge rest of the classes into one using a class mapping in the config.
To load multiple view of images, a little modification should be made to the dataset.

```python
data_prefix=dict(
    pts='lidar',
    back_camera='camera/back_camera',
    front_camera='camera/front_camera',
    front_left_camera='camera/front_left_camera',
    front_right_camera='camera/front_right_camera',
    left_camera='camera/left_camera',
    right_camera='camera/right_camera'
    )
train_dataloader = dict(
    batch_size=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type="PandaSetDataset",
        data_root="./data/pandaset/data",
        ann_file="pandaset_infos_train.pkl",
        data_prefix=data_prefix,
        modality=dict(use_camera=True, use_lidar=False, ),
        pipeline=train_pipeline,
        test_mode=False, )
)
```


## Evaluation

An example to evaluate PointPillars with 8 GPUs with Pandaset metrics is as follows.

```shell
bash ./tools/dist_test.sh configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_ps-mono3d_tidl_3classes.py/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_ps-mono3d_finetune_20210717_095645-8d806dc2_adjusted_3classes.pth 8
```

## Metrics

This dataset uses `nuScenes` metrics as base for evaluation. It consists of mean Average Precision (mAP), Average Translation Error (ATE), Average Scale Error (ASE), Average Orientation Error (AOE), Average Velocity Error (AVE) and Average Attribute Error (AAE).

In nuScenes, nuScenes-devkit's [DetectionEval](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py#L36) is used for evaluation. Instead of that, for pandaset we use the base functions of the [DetectionEval](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py#L36) which are used to evaluate the metrics, called through [pandaset_evaluate_metrics](../../../../mmdet3d/evaluation/metrics/pandaset_metric.py#pandaset_evaluate_metrics).

mean AP: 0.1584
mATE: 0.8797
mASE: 0.2729
mAOE: 0.7013
mAVE: 0.5642
mAAE: 0.4875
NDS: 0.2886
Eval Time 77.4s

Per-class results:
Object Class                        AP              ATE     ASE     AOE     AVE     AAE   
Car                                 0.215908        0.760   0.143   0.197   1.000   0.098 
Pedestrian                          0.218808        0.879   0.274   1.000   0.183   0.364 
Temporary Construction Barriers     0.040541        1.000   0.402   0.907   0.509   1.000 