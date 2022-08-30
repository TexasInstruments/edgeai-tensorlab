# Object Detection Model Zoo

MMDetection3D (./../README_mmdet3d.md) has large Model Zoo, supporting a lot of models. Many of them are high complexity models that are not suitable for embedded scenario that require high throughput. However, here we primarily list speed/accuracy of optimized models that we have trained ourselves or is recommended for TIDL.


## Features

| Detector/Backbone           | SECFPN   | SECFPN(FP16)  | FPN      | Other    |
|-----------------------------|:--------:|:-------------:|:--------:|:--------:|
| pointPillars                | ✓        |               |          |          |

✓ Available, ☐ In progress or partially available, ✗ TBD


#### Results

|Dataset |Model Name                    |Max Num Voxels |Accuracy AP(@70% IOU)(Easy-Moderate-Hard)      |Config File |
|--------|------------------------------|---------------|---------------------------|--------------------------------------------------------|
|KITTI   |PointPillars                  |16000          |**87.25 - 77.43 - 74.43**  |configs/pointPillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py    |
|KITTI   |PointPillars Quantized        |16000          |**86.81 - 76.81 - 74.53**  |configs/pointPillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat.py|



## References

[1] MMDetection3d: https://github.com/open-mmlab/mmdetection3d

[2] PointPillars: Fast Encoders for Object Detection from Point Clouds, https://arxiv.org/abs/1812.05784, Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom