# Object Detection Model Zoo

MMDetection has a huge Model Zoo, supporting a lot of models. Many of them are high complexity models that are not suitable for embedded scenarios that require high throughput. (Please refer to the mmdetection documentation link above for details). However, here we primarily list speed/accuracy optimized models that we have trained ourselves or is recommending from another location.


## Features

| Detector/Backbone           | SECFPN   | SECFPN(FP16)  | FPN      | Other    |
|-----------------------------|:--------:|:-------------:|:--------:|:--------:|
| pointPillars                | ✓        |               |          |          |

✓ Available, ☐ In progress or partially available, ✗ TBD


#### Results

|Dataset |Model Name                    |Max Num Voxels |GigaMACS  |Accuracy AP(@70% IOU)      |Config File |
|--------|------------------------------|---------------|----------|---------------------------|--------------------------------------------------------|
|KITTI   |PointPillars                  |16000          |**TBD**   |**87.25 - 77.43 - 74.43**  |configs/pointPillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py    |
|KITTI   |PointPillars Quantized        |16000          |**TBD**   |**86.81 - 76.81 - 74.53**  |configs/pointPillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat.py|



## References

[1] MMDetection3d: https://github.com/open-mmlab/mmdetection3d

[2] PointPillars: Fast Encoders for Object Detection from Point Clouds, https://arxiv.org/abs/1812.05784, Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom