# FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection

> [FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection](https://arxiv.org/abs/2104.10956)

<!-- [ALGORITHM] -->

## Abstract

Monocular 3D object detection is an important task for autonomous driving considering its advantage of low cost. It is much more challenging than conventional 2D cases due to its inherent ill-posed property, which is mainly reflected in the lack of depth information. Recent progress on 2D detection offers opportunities to better solving this problem. However, it is non-trivial to make a general adapted 2D detector work in this 3D task. In this paper, we study this problem with a practice built on a fully convolutional single-stage detector and propose a general framework FCOS3D. Specifically, we first transform the commonly defined 7-DoF 3D targets to the image domain and decouple them as 2D and 3D attributes. Then the objects are distributed to different feature levels with consideration of their 2D scales and assigned only according to the projected 3D-center for the training procedure. Furthermore, the center-ness is redefined with a 2D Gaussian distribution based on the 3D-center to fit the 3D target formulation. All of these make this framework simple yet effective, getting rid of any 2D detection or 2D-3D correspondence priors. Our solution achieves 1st place out of all the vision-only methods in the nuScenes 3D detection challenge of NeurIPS 2020.

<div align=center>
<img src="https://user-images.githubusercontent.com/30491025/143856739-93b7c4ff-e116-4824-8cc3-8cf1a433a84c.png" width="800"/>
</div>

## Introduction


FCOS3D is a general anchor-free, one-stage monocular 3D object detector adapted from the original 2D version FCOS. We implement and provide the results and checkpoints on the NuScenes dataset. 

## Dataset Preperation

### NuScenes

Prepare the nuScenes dataset as per the MMDetection3D documentation [NuScenes Dataset Preperation](../../docs/en/advanced_guides/datasets/nuscenes.md). 

After downloading nuScenes 3D detection dataset and unzipping all zip files, we typically need to organize the useful data information with a `.pkl` file in a specific style.
To prepare these files for nuScenes, run the following command:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

This command creates `.pkl` files for PETR, BEVFormer and FCOS3D. To include additional data fields for BEVDet and PETRv2, we should add `--bevdet` and `--petrv2`, respectively, to the command. For example,

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --bevdet --petrv2
```

The directory structure after processing should be as below.

```
edgeai-mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── lidarseg (optional)
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_gt_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```

### PandaSet

Download `pandaset.zip` from [HERE](https://huggingface.co/datasets/georghess/pandaset/tree/main) and unzip the file in `./data/pandaset`. Then run the following command to prepare `.pkl` files:

```bash
python tools/create_data.py pandaset --root-path ./data/pandaset --out-dir ./data/pandaset --extra-tag pandaset
```

The directory structure after processing should look like:

```
edgeai-mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── pandaset
│   │   ├── 001
│   │   │   ├── annotations
│   │   │   ├── camera
│   │   │   ├── LICENSE.txt
│   │   │   ├── lidar
│   │   │   └── meta
│   │   ├── 002 
.   .   .
.   .   .
.   .   .
│   │   ├── 158
│   │   ├── pandaset_infos_train.pkl
│   │   ├── pandaset_infos_val.pkl
```

## Get Started

Refer the MMDetection3D documentation [Test and Train with Standard Datasets](../../docs/en/user_guides/train_test.md) for general floating point training/evaluation/testing steps for standard datasets. Use the below steps for training and evaluation of BEVFormer:

1. cd to installation directory <install_dir>/edgeai-mmdetection3d

2. Do floating-model training using the command 
    "./tools/dist_train.sh <config_file> <num_gpus>"

    For example, to use 2 GPUs use the command
    ```bash
    # NuScenes
    ./tools/dist_train.sh ./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_tidl.py 2

    # PandaSet
    ./tools/dist_train.sh ./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_ps-mono3d_tidl.py 2
    ```

3.  Do evalution using the command 

    "python ./tools/test.py  <config_file> <latest.pth file generated from previous step #2>" 

    For example,

    ```bash
    # NuScenes
    python ./tools/test.py ./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_tidl.py ./work_dirs/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_tidl/epoch_12.pth

    # PandaSet
    python ./tools/test.py ./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_ps-mono3d_tidl_3classes.py ./work_dirs/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_ps-mono3d_tidl_3classes/epoch_12.pth
    ```
    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.


## Results

The following results are for FCOS3D with ResNet101 with NuScenes and PandaSet, respectively.

|  Dataset  |                    Model                                           | Mem (GB) | Inf time (fps) | mAP    | NDS   |
|:---------:| :----------------------------------------------------------------- | :------: | :------------: | :---:  | :--:  |
| NuScenes  | fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_tidl          |   0.77   |       TBA      | 29.81  | 37.74 | 
| PandaSet  | fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_ps-mono3d_tidl_3classes  |   0.77   |       TBA      | 15.84  | 28.86 | 

<!-- 
## 3D Object Detection Model Zoo

Complexity and Accuracy report of several trained models is available at the [3D Detection Model Zoo](../../docs/det3d_modelzoo.md) 


## Quantization
This tutorial explains more about quantization and how to do [Quantization Aware Training (QAT)](../../docs/det3d_quantization.md) of detection models.
-->

## ONNX & Prototxt Export

Export of ONNX model (.onnx) is supported by setting the field `save_onnx_model`=True in `model` in a config file. For example,

```bash
model = dict(
    save_onnx_model=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    ...
```
## References

[1] FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection, T. Want, X. Zhu, J. Pang and D. Lin, https://arxiv.org/abs/2104.10956
