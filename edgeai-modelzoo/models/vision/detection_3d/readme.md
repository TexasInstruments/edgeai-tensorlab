# 3D Object Detection Benchmark

## Introduction
3D Object Detection (3DOD) is an important problem in Computer Vision for ADAS, Autonomous Driving and Robotics. This page describes some of the popular 3DOD models.

## Datasets

### Vision-based BEV Object Detection
**PandaSet Dataset**
- [PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving](https://arxiv.org/abs/2112.12610)
- [PandaSet Devkit](https://github.com/scaleapi/pandaset-devkit)
- [PandaSet Tutorial(https://docs.basic.ai/docs/pandaset-dataset-tutorial)
- [Download Link](https://huggingface.co/datasets/georghess/pandaset)

### Lidar 3D Object Detection
**KITTI Dataset**: http://www.cvlibs.net/datasets/kitti/
- **KITTI 3D Object Detection Benchmark**: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d


## Models

Note: This page lists only a subset of models - a bigger list can be found in [edgeai-mmdetection3d](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-mmdetection3d)

### Vision-based models 

- [Models Link - PandaSet 3DOD](./pandaset/mmdet3d/)
- [Training Code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-mmdetection3d)

|Dataset      |Model Name                     |Input Size |GigaMACs  |mAP      | NDS      |Available|Notes |
|-------------|------------------------------ |-----------|----------|---------|----------|---------|------|
|PandaSet     | FastBEV_pandaset_r18_f1       |256x704    |344.05    |17.47    | 26.46    |Y        |      | 
|PandaSet     | FastBEV_pandaset_r34_f4       |256x704    |344.05    |23.07    | 31.56    |         |      | 
|PandaSet     | Bevformer_tiny_pandaset       |544x960    |          |23.01    | 30.05    |Y        |      |

FastBEV models with f1 suffix indicates the models that use only the current frame, while the ones with f4 uses three previous frame features as well as a current frame. r18 indicates ResNet18 backbone and r34 indicates ReNet34 backbone.


<h1> 

### Lidar 3D Object Detection models (deprecated)

- [Models Link - KITTI 3D OD](./kitti/mmdet3d/)
- [Models Link - PandaSet BEV OD](./pandaset/mmdet3d/)
- [Training Code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-mmdetection3d)


|Dataset      |Model Name                     |Input Size |GigaMACs  |AP 3D Moderate% (Car) |Available|Notes |
|-------------|------------------------------ |-----------|----------|----------------------|---------|------|
|Kitti        |PointPillars                   |496x432    |33.44     |76.36                 |Y        |      | 


## References

[1] Y. Li, B. Huang, Z. Chen, Y. Cui, F. Liang, M. shen, F. Liu, E. Xie, L.Sheng, W. Ouyang and J. Shao, Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline, https://arxiv.org/abs/2301.12511

[2] BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers, https://arxiv.org/abs/2203.17270

[3] A. Geiger, P. Lenz and R. Urtasun, Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite, Conference on Computer Vision and Pattern Recognition (CVPR), 2012

[4] A. Geiger, P. Lenz, C. Stiller and R. Urtasun, Vision meets Robotics: The KITTI Dataset, International Journal of Robotics Research (IJRR), 2013

[5] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang and O. Beijbom, PointPillars: Fast Encoders for Object Detection from Point Clouds, https://arxiv.org/abs/1812.05784

