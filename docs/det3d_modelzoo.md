# Object Detection Model Zoo

[MMDetection3D](https://github.com/open-mmlab/mmdetection3d) has large Model Zoo, supporting a lot of models. Many of them are high complexity models that are not suitable for embedded scenario that require high throughput. However, here we primarily list speed/accuracy of optimized models that we have trained ourselves or is recommended for TIDL.


## Features

| Detector/Backbone           | SECFPN   | SECFPN(FP16)  | FPN      | Other    |
|-----------------------------|:--------:|:-------------:|:--------:|:--------:|
| pointPillars                | ✓        |               |          |          |
| pointPainting               | ✓        |               |          |          |

✓ Available, ☐ In progress or partially available, ✗ TBD


#### Results

|Dataset |Model Name                    |Max Num Voxels |Accuracy AP(Easy-Moderate-Hard)      |Config File |
|--------|------------------------------|---------------|---------------------------|--------------------------------------------------------|
|KITTI   |PointPillars (Car class)                  |10000          |**86.51 - 76.84 - 73.74**  |configs/pointPillars/tidl_pointpillars_hv_secfpn_8x6b_160e_kitti-3d-car.py    |
|KITTI   |PointPillars Quantized (Car class)        |10000          |**TBD**  |configs/pointPillars/tidl_pointpillars_hv_secfpn_8x6b_160e_kitti-3d-car_qat.py|
|KITTI   |PointPillars (3 class)                    |10000          |**73.84 - 62.66 - 57.88**  |configs/pointPillars/tidl_pointpillars_hv_secfpn_8x6b_160e_kitti-3d-3class.py|
|KITTI   |PointPillars Quantized (3 class)          |10000          |**TBD**  |configs/pointPillars/tidl_pointpillars_hv_secfpn_8x6b_160e_kitti-3d-3class_qat.py|
|KITTI   |PointPainting (3 class)                   |10000          |**74.62 - 63.49 - 59.95**  |configs/pointPillars/tidl_pointpillars_hv_secfpn_8xb6_160e_kitti-3d-3class-painted.py|
|KITTI   |PointPainting Quantized (3 class)         |10000          |**TBD**  |configs/pointPillars/tidl_pointpillars_hv_secfpn_8xb6_160e_kitti-3d-3class-painted_qat.py|

## References

[1] MMDetection3d: https://github.com/open-mmlab/mmdetection3d

[2] PointPillars: Fast Encoders for Object Detection from Point Clouds, https://arxiv.org/abs/1812.05784, Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom

[3] PointPainting: https://arxiv.org/abs/1911.10150