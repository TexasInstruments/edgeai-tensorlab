# Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline

> [Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline](https://arxiv.org/abs/2301.12511)

<!-- [ALGORITHM] -->

## Abstract

Recently, perception task based on Bird's-Eye View (BEV) representation has drawn more and more attention, and BEV representation is promising as the foundation for next-generation Autonomous Vehicle (AV) perception. However, most existing BEV solutions either require considerable resources to execute on-vehicle inference or suffer from modest performance. This paper proposes a simple yet effective framework, termed Fast-BEV , which is capable of performing faster BEV perception on the on-vehicle chips. Towards this goal, we first empirically find that the BEV representation can be sufficiently powerful without expensive transformer based transformation nor depth representation. Our Fast-BEV consists of five parts, We novelly propose (1) a lightweight deployment-friendly view transformation which fast transfers 2D image feature to 3D voxel space, (2) an multi-scale image encoder which leverages multi-scale information for better performance, (3) an efficient BEV encoder which is particularly designed to speed up on-vehicle inference. We further introduce (4) a strong data augmentation strategy for both image and BEV space to avoid over-fitting, (5) a multi-frame feature fusion mechanism to leverage the temporal information. Through experiments, on 2080Ti platform, our R50 model can run 52.6 FPS with 47.3% NDS on the nuScenes validation set, exceeding the 41.3 FPS and 47.5% NDS of the BEVDepth-R50 model and 30.2 FPS and 45.7% NDS of the BEVDet4D-R50 model. Our largest model (R101@900x1600) establishes a competitive 53.5% NDS on the nuScenes validation set. We further develop a benchmark with considerable accuracy and efficiency on current popular on-vehicle chips. The code is released at https://github.com/Sense-GVT/Fast-BEV.

<div align=center>
<img src="https://github.com/Sense-GVT/Fast-BEV/blob/main/fast-bev++.png" width="800"/>
</div>


## Introduction

FastBEV is a lightweight and deployment-friendly BEV object detector. We implement and provide the results and checkpoints on the NuScenes dataset. 

## Dataset Preperation

### NuScenes

Prepare the nuScenes dataset as per the MMDetection3D documentation [NuScenes Dataset Preperation](../../docs/en/advanced_guides/datasets/nuscenes.md). 

After downloading nuScenes 3D detection dataset and unzipping all zip files, we typically need to organize the useful data information with a `.pkl` file in a specific style.
To prepare these files for nuScenes, run the following command:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --canbus ./data
```

This command creates `.pkl` files for PETR, BEVFormer and FCOS3D. To include additional data fields for BEVDet and PETRv2, we should add `--bevdet` and `--petrv2`, respectively, to the command. For example,

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --canbus ./data --bevdet --petrv2
```

FastBEV uses multiple temporal frames and therefore need to organize neighboring frames's information as well in a `.pkl` file for training. For this purpose, we should run the following script, which will create `nuscenes_infos_train_fastbev.pkl` from `nuscenes_infos_train.pkl`.

```bash
python tools/dataset_converters/generate_fastbev_sweep_pkl.py nuscenes --root-path ./data/nuscenes --version 'v1.0-trainval'
```

The directory structure after processing should be as below.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── can_bus
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── lidarseg (optional)
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_gt_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_train_fastbev.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```

### PandaSet

Download `pandaset.zip` from [HERE](https://huggingface.co/datasets/georghess/pandaset/tree/main) and unzip the file in `./data/pandaset`. Then run the following command to prepare base `.pkl` files:

```bash
python tools/create_data.py pandaset --root-path ./data/pandaset --out-dir ./data/pandaset --extra-tag pandaset
```

FastBEV uses multiple temporal frames and therefore need to organize neighboring frames's information as well in a `.pkl` file for training. For this purpose, we should run the following script, which will create `pandaset_infos_train_fastbev.pkl` from `pandaset_infos_train.pkl`.

```bash
python tools/dataset_converters/generate_fastbev_sweep_pkl.py pandaset --root-path ./data/nuscenes'
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
│   │   ├── pandaset_infos_train_fastbev.pkl
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
    ./tools/dist_train.sh project/FastBEV/configs/fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4.py 2

    # PandaSet
    ./tools/dist_train.sh project/FastBEV/configs/fastbev_pandaset_m2_r34_s256x704_v200x200x4_c224_d4_f4.py 2
    ```

3.  Do evalution using the command 

    "python ./tools/test.py <config_file> <latest.pth file generated from previous step #2>" 

    For example,

    ```bash
    # NuScenes
    python ./tools/test.py project/FastBEV/configs/fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4.py ./work_dirs/fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4/epoch_20.pth

    # PandaSet
    python ./tools/test.py project/FastBEV/configs/fastbev_pandaset_m2_r34_s256x704_v200x200x4_c224_d4_f4.py ./work_dirs/fastbev_pandaset_m2_r34_s256x704_v200x200x4_c224_d4_f4/epoch_20.pth

    ```
    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.


## Results

fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f1 and fastbev_pandaset_m0_r18_s256x704_v200x200x4_c192_d2_f1 use a current frame only with ResNet18 as a backbone, while fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4 and fastbev_pandaset_m2_r34_s256x704_v200x200x4_c224_d4_f4 three previous frames as well as a current frame with ResNet34.

|  Dataset  |                          Model                         | Mem (GB) | Inf time (fps) |  mAP   |  NDS  |
|:---------:|:------------------------------------------------------ | :------: | :------------: | :---:  | :---: |
| NuScenes  | fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f1          |   0.33   |       TBA      | 24.91  | 34.26 | 
|           | fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4          |   0.83   |       TBA      | 34.71  | 47.65 | 
| PandaSet  | fastbev_pandaset_m0_r18_s256x704_v200x200x4_c192_d2_f1 |   0.33   |       TBA      | 17.47  | 26.46 | 
|           | fastbev_pandaset_m2_r34_s256x704_v200x200x4_c224_d4_f4 |   0.83   |       TBA      | 23.07  | 31.50 | 

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
    type='FastBEV',
    style="v1",
    save_onnx_model=True,
    num_temporal_feats=n_times-1,
    feats_size = [6, 64, 64, 176],
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        **img_norm_cfg,
        pad_size_divisor=32),
    ...
```
## References

[1] Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline, Y. Li, B. Huang, Z. Chen, Y. Cui, F. Liang, M. shen, F. Liu, E. Xie, L.Sheng, W. Ouyang and J. Shao, https://arxiv.org/abs/2301.12511
