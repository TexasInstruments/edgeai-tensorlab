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


#### COCO 2017 Dataset
- Train on COCO 2017 Train Set
- Test on COCO 2017 Validation Set

###### Single Shot Mult-Box Detector (SSD) 
Please see the reference [1] for algorithmic details of the detector.

|Model Arch       |Backbone Model|Resolution|AP [0.5:0.95]%|Giga MACS|Model Config File                |Download |
|----------       |--------------|----------|--------------|---------|---------------------------------|---------|
|SSDLite+FPN      |RegNetX800MF  |512x512   |              |         |ssd-lite_regnet_fpn.py           |         |
|SSDLite+FPN      |RegNetX1.6GF  |768x768   |              |         |ssd-lite_regnet_fpn.py           |         |
|.
|SSD+FPN          |ResNet50      |512x512   |              |         |ssd_resnet_fpn.py                |[link](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-modelzoo/browse/pytorch/vision/object_detection/xmmdet/coco/ssd_resnet_fpn) |
|.
|SSD*             |VGG16         |512x512   |**29.34**     |**98.81**|                                 |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd) |


###### RetinaNet Detector
Please see the reference [2] for algorithmic details of the detector.

|Model Arch       |Backbone Model|Resolution |AP [0.5:0.95]%|Giga MACS |Model Config File                |Download |
|----------       |--------------|-----------|--------------|----------|---------------------------------|---------|
|RetinaNetLite+FPN|RegNet800MF   |512x512    |              |**6.04**  |retinanet-lite_regnet_fpn_bgr.py |         |
|RetinaNetLite+FPN|RegNetX1.6GF  |768x768    |              |          |retinanet-lite_regnet_fpn.py     |         |
|.
|RetinaNet+FPN*   |ResNet50      |512x512    |**29.7**      |**68.88** |                                 |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) |
|RetinaNet+FPN*   |ResNet50      |768x768    |**34.0**      |**137.75**|                                 |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) |
|RetinaNet+FPN*   |ResNet50      |(1536,768) |**37.0**      |**275.5** |                                 |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) |
<br>

- The suffix **Lite** indicates that the model uses either Depthwise Convolutions (like in MobileNet models) or grouped convolutions (like in RegNetX models). When the backbone is a MobileNet, we use Depthwise convolutions even in FPN and the detector heads. When the backbone is a RegNet model, we use Grouped convolutions with the same group size that the RegNet backbone uses. But for backbones that use regular convolutions (such as ResNet) we do not use Depthwise or Grouped convolutions.
- A square resolution such as 512x512 indicates that the inputs are resized to that resolution without respecting the aspect ration of the image (keep_ratio=False in config files)<br>
- A non-square resolution indicated with comma (1536,768) or dash (1536-768) indicates that images are resized to fit within this maximum and minimum size - but the aspect ratio of the image is preserved (keep_ratio=True in config files). This means that each image may have a different size after it is resized and hence is not suitable for embedded inference. But the interesting thing is that such a model can also be inferred or evaluated using a square aspect ratio.<br>
- The models with a \* were not trained by us, but rather taken from mmdetection model zoo and inference is run at teh said resolution.<br>


## References

[1] SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg

[2] Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002, Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár