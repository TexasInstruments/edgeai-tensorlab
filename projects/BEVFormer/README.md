# BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers

> [BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/pdf/2203.17270)

<!-- [ALGORITHM] -->

## Abstract

3D visual perception tasks, including 3D detection and map segmentation based on multi-camera images, are essential for autonomous driving systems. In this work, we present a new framework termed BEVFormer, which learns unified BEV representations with spatiotemporal transformers to support multiple autonomous driving perception tasks. In a nutshell, BEVFormer exploits both spatial and temporal information by interacting with spatial and temporal space through predefined
grid-shaped BEV queries. To aggregate spatial information, we design spatial cross-attention that each BEV query extracts the spatial features from the regions of interest across camera views. For temporal information, we propose temporal self-attention to recurrently fuse the history BEV information. Our approach achieves the new state-of-the-art 56.9% in terms of NDS metric on the nuScenes test set, which is 9.0 points higher than previous best arts and on par with the performance of LiDAR-based baselines. We further show that BEVFormer remarkably improves
the accuracy of velocity estimation and recall of objects under low visibility conditions. The code is available at https://github.com/fundamentalvision/BEVFormer.

## Introduction

We implement and provide the results and checkpoints on the NuScenes dataset. <!-- The result can be found in [Object Detection Zoo](../../docs/det3d_modelzoo.md) -->

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
    ./tools/dist_train.sh projects/BEVFormer/configs/bevformer_tiny.py 2

    # PandaSet
    ./tools/dist_train.sh projects/BEVFormer/configs/bevformer_tiny_pandaset.py 2
    ```

3.  Do evalution using the command 

    "python ./tools/test.py <config_file> <latest.pth file generated from previous step #2>" 

    For example,

    ```bash
    # NuScenes
    python ./tools/test.py projects/BEVFormer/configs/bevformer_tiny.py ./work_dirs/bevformer_tiny/epoch_24.pth

    # PandaSet
    python ./tools/test.py projects/BEVFormer/configs/bevformer_tiny_pandaset.py ./work_dirs/bevformer_tiny_pandaset/epoch_24.pth
    ```
    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.


## Results

The following results are for the bevformer_tiny configs with NuScenes and PandaSet, respectively.

|  Dataset  |                      Model                    | Mem (GB) | Inf time (fps) |  mAP   |  NDS  |
|:---------:|:--------------------------------------------- | :------: | :------------: | :---:  | :--:  |
| NuScenes  | bevformer_tiny                                |   0.63   |       TBA      | 26.84  | 37.09 | 
| PandaSet  | bevformer_tiny_pandaset                       |   0.80   |       TBA      | 23.01  | 30.05 | 

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
    type='BEVFormer',
    use_grid_mask=True,
    save_onnx_model=True,
    data_preprocessor=dict(
        type='BEVFormer3DDataPreprocessor', **img_norm_cfg, pad_size_divisor=32),
        ...
```
## References

[1] BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers, Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, J. Dai, https://arxiv.org/pdf/2203.17270
