# 3D Object Detection Benchmark


## Introduction
Applications that requires very high accuracy object detection employs 3D Object Detection. When LIDAR sensor input is used, higher accuracies can be achieved compared to what is possible with image input.


## Datasets
- **KITTI Dataset**: http://www.cvlibs.net/datasets/kitti/
- **KITTI 3D Object Detection Benchmark**: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d


## Models

### PointPillars
- [Models Link - KITTI 3D OD](./kitti/mmdet3d/)
- [Additional information](https://arxiv.org/pdf/1812.05784.pdf)
- [Training Code](https://github.com/TexasInstruments/edgeai-mmdetection3d)
- Note: 3D Object Detection using Lidar input.


|Dataset      |Model Name                     |Input Size |GigaMACs  |AP 3D Moderate% (Car) |Available|Notes |
|-------------|------------------------------ |-----------|----------|----------------------|---------|------|
|Kitti        |PointPillars                   |496x432    |33.44     |76.36                 |Y        |      | 


## References

[1] Andreas Geiger and Philip Lenz and Raquel Urtasun, Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite, Conference on Computer Vision and Pattern Recognition (CVPR), 2012

[2] Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun, Vision meets Robotics: The KITTI Dataset, International Journal of Robotics Research (IJRR), 2013

[3] Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom, PointPillars: Fast Encoders for Object Detection from Point Clouds, https://arxiv.org/abs/1812.05784

