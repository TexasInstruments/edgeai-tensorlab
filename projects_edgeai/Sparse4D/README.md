# Sparse4D v3: Advancing End-to-End 3D Detection and Tracking

> [Sparse4D v3: Advancing End-to-End 3D Detection and Tracking](https://arxiv.org/pdf/2311.11722)

<!-- [ALGORITHM] -->

## Abstract

n autonomous driving perception systems, 3D detection and tracking are the two fundamental tasks. This paper delves deeper into this field, building upon the Sparse4D framework. We introduce two auxiliary training tasks (Temporal Instance Denoising and Quality Estimation) and propose decoupled attention to make structural improvements, leading to significant enhancements in detection performance. Additionally, we extend the detector into a tracker using a straight-forward approach that assigns instance ID during inference, further highlighting the advantages of query-based algorithms. Extensive experiments conducted on the nuScenes benchmark validate the effectiveness of the proposed improvements. With ResNet50 as the backbone, we witnessed enhancements of 3.0%, 2.2%, and 7.6% in mAP, NDS, and AMOTA, achieving 46.9%, 56.1%, and 49.0%, respec-
tively. Our best model achieved 71.9% NDS and 67.7% AMOTA on the nuScenes test set. Code will be released at https://github.com/linxuewu/Sparse4D.

## Introduction

We implement and provide the results and checkpoints on the NuScenes dataset.

## Dataset Preperation

### NuScenes

Prepare the nuScenes dataset as per the MMDetection3D documentation [NuScenes Dataset Preperation](../../docs/en/advanced_guides/datasets/nuscenes.md). 

After downloading nuScenes 3D detection dataset and unzipping all zip files, we typically need to organize the useful data information with a `.pkl` file in a specific style. To prepare them with NuScenes dataset for Sparse4D, run the following command:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --sparse4d
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

PandaSet is not supported yet for Sparse4D.

## Get Started

Refer the MMDetection3D documentation [Test and Train with Standard Datasets](../../docs/en/user_guides/train_test.md) for general floating point training/evaluation/testing steps for standard datasets. Use the below steps for training and evaluation of BEVFormer:

1. cd to installation directory <install_dir>/edgeai-mmdetection3d

2. Do floating-model training using the command 
    "./tools/dist_train.sh <config_file> <num_gpus>"

    For example, to use 2 GPUs use the command
    ```bash
    # NuScenes
    ./tools/dist_train.sh projects_edgeai/Sparse4D/configs/nuscenes/sparse4dv3_temporal_r50_1x8_bs6_256x704.py 2
    ```

3.  Do evalution using the command 

    "python ./tools/test.py <config_file> <latest.pth file generated from previous step #2>" 

    For example,

    ```bash
    # NuScenes
    python ./tools/test.py projects_edgeai/Sparse4D/configs/nuscenes/sparse4dv3_temporal_r50_1x8_bs6_256x704.py ./work_dirs/sparse4dv3_temporal_r50_1x8_bs6_256x704/epoch_100.pth
    ```
    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.


## Results

The following results are for the Sparse4D configs with NuScenes. The model has been trained through 100 epochs.

|  Dataset  |                      Model                    | Mem (GB) | Inf time (fps) |  mAP   |  NDS  |
|:---------:|:--------------------------------------------- | :------: | :------------: | :---:  | :--:  |
| NuScenes  | sparse4dv3_temporal_r50_1x8_bs6_256x704       |   0.91   |       TBA      | 44.66  | 55.94 | 
| PandaSet  |                                               |          |                |        |       | 

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
    type='Sparse4D',
    use_grid_mask=True,
    save_onnx_model=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor', **img_norm_cfg, pad_size_divisor=32),
        ...
```
## References

[1] Sparse4D v3: Advancing End-to-End 3D Detection and Tracking, X. Lin, Z.Pei, T. Lin, L. Huang, Z. Su, https://arxiv.org/pdf/2311.11722
