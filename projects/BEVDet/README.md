# BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View

> [BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View](https://arxiv.org/abs/2112.11790)

<!-- [ALGORITHM] -->

## Abstract

Autonomous driving perceives its surroundings for decision making, which is one of the most complex scenarios in visual perception. The success of paradigm innovation in solving the 2D object detection task inspires us to seek an elegant, feasible, and scalable paradigm for fundamentally pushing the performance boundary in this area. To this end, we contribute the BEVDet paradigm in this paper. BEVDet performs 3D object detection in Bird-Eye-View (BEV), where most target values are defined and route planning can be handily performed. We merely reuse existing modules to build its framework but substantially develop its performance by constructing an exclusive data augmentation strategy and upgrading the Non-Maximum Suppression strategy. In the experiment, BEVDet offers an excellent trade-off between accuracy and time-efficiency. As a fast version, BEVDet-Tiny scores 31.2% mAP and 39.2% NDS on the nuScenes val set. It is comparable with FCOS3D, but requires just 11% computational budget of 215.3 GFLOPs and runs 9.2 times faster at 15.6 FPS. Another high-precision version dubbed BEVDet-Base scores 39.3% mAP and 47.2% NDS, significantly exceeding all published results. With a comparable inference speed, it surpasses FCOS3D by a large margin of +9.8% mAP and +10.0% NDS. The source code is publicly available for further research at [this https URL](https://github.com/HuangJunJie2017/BEVDet).

## Introduction

We implement and provide the results and checkpoints on the NuScenes dataset.  <!-- The result can be found in [Object Detection Zoo](../../docs/det3d_modelzoo.md) -->

## Dataset Preperation

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

The folder structure after processing should be as below.

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

## Get Started

Refer the MMDetection3D documentation [Test and Train with Standard Datasets](../../docs/en/user_guides/train_test.md) for general floating point training/evaluation/testing steps for standard datasets. Use the below steps for training and evaluation of BEVDet:

1. cd to installation directory <install_dir>/edgeai-mmdetection3d

2. Do floating-model training using the command 
    "./tools/dist_train.sh projects/BEVDet/configs/bevdet-r50.py <num_gpus>"

    For example, to use 2 GPUs use the command
    ```bash
    ./tools/dist_train.sh projects/BEVDet/configs/bevdet-r50.py 2
    ```

3.  Do evalution using the command 

    "python ./tools/test.py projects/BEVDet/configs/bevdet-r50.py <latest.pth file generated from previous step #2>" 

    For example,

    ```bash
    python ./tools/test.py projects/BEVDet/configs/bevdet-r50.py ./work_dirs/bevdet-r50/epoch_24.pth
    ```
    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.

## Results

This Result is trained by bevdet-r50.py.

|                    Model                      | Mem (GB) | Inf time (fps) | mAP    | NDS   |
| :-------------------------------------------: | :------: | :------------: | :---:  | :--:  |
| bevdet-r50                                    |   0.58   |       TBA      | 28.45  | 35.01 | 

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
    type='BEVDet',
    save_onnx_model=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675,  116.280, 103.530],
        std=[58.395, 57.120, 57.375],
        ...
```
## References

[1] BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View, J. Huang, G. Huang, Z. Zhu, Y. ye, D. Du, https://arxiv.org/abs/2112.11790
