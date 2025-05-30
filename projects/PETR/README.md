# PETR: Position Embedding Transformation for Multi-View 3D Object Detection

> [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we develop position embedding transformation (PETR) for multi-view 3D object detection. PETR encodes the position information of 3D coordinates into image features, producing the 3D position-aware features. Object query can perceive the 3D position-aware features and perform end-to-end object detection. PETR achieves state-of-the-art performance (50.4% NDS and 44.1% mAP) on standard nuScenes dataset and ranks 1st place on the benchmark. It can serve as a simple yet strong baseline for future research. Code is available at [this URL](https://github.com/megvii-research/PETR).

# PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images

> [PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images](https://arxiv.org/abs/2206.01256)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we propose PETRv2, a unified framework for 3D perception from multi-view images. Based on PETR, PETRv2 explores the effectiveness of temporal modeling, which utilizes the temporal information of previous frames to boost 3D object detection. More specifically, we extend the 3D position embedding (3D PE) in PETR for temporal modeling. The 3D PE achieves the temporal alignment on object position of different frames. A feature-guided position encoder is further introduced to improve the data adaptability of 3D PE. To support for multi-task learning (e.g., BEV segmentation and 3D lane detection), PETRv2 provides a simple yet effective solution by introducing task-specific queries, which are initialized under different spaces. PETRv2 achieves state-of-the-art performance on 3D object detection, BEV segmentation and 3D lane detection. Detailed robustness analysis is also conducted on PETR framework. We hope PETRv2 can serve as a strong baseline for 3D perception. Code is available at [this URL](https://github.com/megvii-research/PETR). 


## Introduction

We implement and provide the results and checkpoints for PETR and PETRv2 on the NuScenes dataset. <!--  The result can be found in [Object Detection Zoo](../../docs/det3d_modelzoo.md) -->

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

Refer the MMDetection3D documentation [Test and Train with Standard Datasets](../../docs/en/user_guides/train_test.md) for general floating point training/evaluation/testing steps for standard datasets. Use the below steps for training and evaluation of BEVDet:

1. cd to installation directory <install_dir>/edgeai-mmdetection3d

2. Do floating-model training using the command 
    "./tools/dist_train.sh <config_file> <num_gpus>"

    For example, to use 2 GPUs use the command
    ```bash
    # NuScenes
    ./tools/dist_train.sh projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py 2

    # PandaSet
    ./tools/dist_train.sh projects/PETR/configs/petr_pandaset_vovnet_gridmask_p4_960x352.py 2
    ```

3.  Do evalution using the command 

    "python <config_file> <latest.pth file generated from previous step #2>" 

    For example,

    ```bash
    # NuScenes
    python ./tools/test.py projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py ./work_dirs/petr_vovnet_gridmask_p4_800x320/epoch_24.pth

    # PandaSet
    python ./tools/test.py projects/PETR/configs/petr_pandaset_vovnet_gridmask_p4_960x352.py ./work_dirs/petr_pandaset_vovnet_gridmask_p4_960x352/epoch_24.pth
    ```
    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.

## Results

The following results are for PETR and PETRv2 with NuScenes and PandaSet, respectively. PETRv2 with PandaSet is not avaiable yet.


|  Dataset  |                    Model                      | Mem (GB) | Inf time (fps) | mAP    | NDS   |
|:---------:| :-------------------------------------------- | :------: | :------------: | :---:  | :--:  |
| NuScenes  | petr_vovnet_gridmask_p4_800x320               |   1.09   |       TBA      | 37.83  | 43.11 |
|           | petrv2_vovnet_gridmask_p4_800x320             |   1.87   |       TBA      | 39.36  | 47.89 |
| PandaSet  | petr_pandaset_vovnet_gridmask_p4_960x352      |   1.34   |       TBA      | 25.37  | 28.88 |
|           | petrv2_pandaset_vovnet_gridmask_p4_960x352    |          |                |        |       |


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
    type='PETR',
    save_onnx_model=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        ...
```

## References

[1] PETR: Position Embedding Transformation for Multi-View 3D Object Detection, Y. Liu, T. Wang, X. Zhang, J. Sun, https://arxiv.org/abs/2203.05625

[2] PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images, Y. Liu, J. Yan, F. Jia, S. Li, A. Gao, T. Wang, X. Zhang, J. Sun, https://arxiv.org/abs/2206.01256

