# 3D Object Detection Benchmark


## Introduction
Applications that requires very high accuracy object detection employs 3D Object Detection. Bird's-Eye-View (BEV) detection has gained increasing attention for its capability to detect and present objects from multiple sensors in a unified framework. While vision-based detection is becoming more popular for its low cost and scalability, higher accuracy can be achieved when LiDAR sensor is used. We provide vision-based BEV models and LiDAR-based 3D detection model.

## Datasets

### Vision-based BEV Object Detection
- **PandaSet Dataset**: https://huggingface.co/datasets/georghess/pandaset
- **PandaSet devkit**: https://github.com/scaleapi/pandaset-devkit

### Lidar 3D Object Detection
- **KITTI Dataset**: http://www.cvlibs.net/datasets/kitti/
- **KITTI 3D Object Detection Benchmark**: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d


## Models

- [Models Link - KITTI 3D OD](./kitti/mmdet3d/)
- [Models Link - PandaSet BEV OD](./pandaset/mmdet3d/)
- [Training Code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-mmdetection3d)

- Note: More models are being supported from [edgeai-mmdetection3d](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-mmdetection3d). We shall update this section once those models are verified in TIDL.


### Vision-based models 

|Dataset      |Model Name                     |Input Size |GigaMACs  |mAP      | NDS      |Available|Notes |
|-------------|------------------------------ |-----------|----------|---------|----------|---------|------|
|PandaSet     | FastBEV_plus_pandaset_r18_f1  |256x704    |344.05    |17.47    | 26.46    |Y        |      | 


### Lidar models

|Dataset      |Model Name                     |Input Size |GigaMACs  |AP 3D Moderate% (Car) |Available|Notes |
|-------------|------------------------------ |-----------|----------|----------------------|---------|------|
|Kitti        |PointPillars                   |496x432    |33.44     |76.36                 |Y        |      | 


## References

[1] Y. Li, B. Huang, Z. Chen, Y. Cui, F. Liang, M. shen, F. Liu, E. Xie, L.Sheng, W. Ouyang and J. Shao, Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline, https://arxiv.org/abs/2301.12511


[2] A. Geiger, P. Lenz and R. Urtasun, Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite, Conference on Computer Vision and Pattern Recognition (CVPR), 2012

[3] A. Geiger, P. Lenz, C. Stiller and R. Urtasun, Vision meets Robotics: The KITTI Dataset, International Journal of Robotics Research (IJRR), 2013

[4] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang and O. Beijbom, PointPillars: Fast Encoders for Object Detection from Point Clouds, https://arxiv.org/abs/1812.05784

