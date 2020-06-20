# Jacinto-AI-MMDetection Model Zoo

MMDetection has a huge Model Zoo, supporting a lot of models. Many of them are high complexity models that are not suitable for embedded scenarios that require high throughput. (Please refer to the mmdetection documentation link above for details). However, in this fork, we list only speed/accuracy optimized models that we have trained ourselves or is recommending from another location.

## Features

|                    | ResNet   | Reget    | MobileNet|
|--------------------|:--------:|:--------:|:--------:|
| SSD                | ✓        | ✓        | ✓        |
| RetinaNet          | ☐        | ☐        | ☐        |
| FCOS               | ✗        | ✗        | ✗        |
| Faster R-CNN       | ✗        | ✗        | ✗        |
| Mask R-CNN         | ✗        | ✗        | ✗        |

✓ Available, ☐ In progress or partially available, ✗ TBD

#### Pascal VOC2007 Dataset
- Train on Pascal VOC 2007+2012 TrainVal Set
- Test on Pascal VOC 2007 Test Set

|Dataset  |Model Arch   |Backbone Model|BB Stride|Resolution|MeanAP50 (mAP%)|Giga MACS|Model Config File                                 |Pre-training|Download |
|---------|----------   |-----------   |-------- |----------|--------       |-------  |----------                                        |----------- |---
|VOC2007  |SSD+FPN      |MobileNetV2   |32       |512x512   |**76.1**       |**2.21** |configs/ssd/ ssd_mobilenet_fpn.py                 |ImageNet    |[link](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-modelzoo/browse/pytorch/vision/object_detection/mmdetection/ssd/20200612-051942_ssd512_mobilenetv2_fpn) |
|VOC2007  |SSD+FPN      |RegNet800MF   |32       |512x512   |**79.7**       |**5.64** |configs/ssd/ ssd_regnet_fpn.py                    |ImageNet    |[link](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-modelzoo/browse/pytorch/vision/object_detection/mmdetection/ssd/20200611-200124_ssd512_regnet800mf_fpn_bgr) |
|VOC2007  |SSD+FPN      |ResNet50      |32       |512x512   |**80.5**       |**27.1** |configs/ssd/ ssd_resnet_fpn.py                    |ImageNet    |[link](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-modelzoo/browse/pytorch/vision/object_detection/mmdetection/ssd/20200614-234748_ssd512_resnet_fpn) |
|VOC2007  |SSD          |VGG16         |         |512x512   |               |**90.39**|mmdetection/configs/pascal_voc/ ssd512_voc0712.py|ImageNet    |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc)|
|.
|VOC2007  |RetinaNet+FPN|RegNet800MF   |32       |768x384    |              |**5.82** |configs/retinanet/ retinanet_resnet_fpn_bgr.py    |COCO        |   |
|VOC2007  |RetinaNet+FPN|ResNet50      |32       |768x384    |**81.6**      |**61.24**|configs/retinanet/ retinanet_resnet_fpn_bgr.py    |COCO        |   |

#### COCO 2017 Dataset
- Train on COCO 2017 Train Set
- Test on COCO 2017 Validation Set

|Dataset  |Model Arch   |Backbone Model|BB Stride|Resolution|AP [0.5:0.95]%|Giga MACS|Model Config File                                          |Pre-training|Download |
|---------|----------   |-----------   |-------- |----------|--------      |-------  |----------                                                 |----       |---      |
|COCO2017 |SSD          |VGG16         |         |512x512   |**29.34**     |**98.81**|mmdetection/configs/ ssd/ ssd512_coco.py                   |ImageNet   |         |
|.
|COCO2017 |RetinaNet+FPN|RegNet800MF   |32       |768x384   |              |**6.04** |configs/retinanet/ retinanet_regnet_fpn_bgr.py             |ImageNet   |         |
|COCO2017 |RetinaNet+FPN|ResNet50      |32       |768x384*  |**29.0**      |**68.88**|mmdetection/configs/ retinanet/ retinanet_r50_fpn_1x_coco.py|ImageNet   |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet)|
|COCO2017 |RetinaNet+FPN|ResNet50      |32       |1536x768* |**36.1**      |**275.5**|mmdetection/configs/retinanet/ retinanet_r50_fpn_1x_coco.py|ImageNet   |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/atss)      |


\* Inference is run at this resolution using the trained model given in the link.

## References

[1] MMDetection: Open MMLab Detection Toolbox and Benchmark, https://arxiv.org/abs/1906.07155, Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin

[2] SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg