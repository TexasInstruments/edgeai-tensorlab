# Jacinto-AI-MMDetection Model Zoo

MMDetection has a huge Model Zoo, supporting a lot of models. Many of them are high complexity models that are not suitable for embedded scenarios that require high throughput. (Please refer to the mmdetection documentation link above for details). However, in this fork, we list only speed/accuracy optimized models that we have trained ourselves or is recommending from another location.


## Features

|                             | ResNet   | RegNetX  | MobileNet|
|-----------------------------|:--------:|:--------:|:--------:|
| SSD                         | ✓        | ✓        | ☐        |
| RetinaNet                   | ☐        | ☐        | ☐        |
| CenterNet(Objects As Points)| ✗        | ✗        | ✗        |
| EfficientDet                | ✗        | ✗        | ✗        |
| YOLOv3                      | ✗        | ✗        | ✗        |
| Faster R-CNN                | ✗        | ✗        | ✗        |
| Mask R-CNN                  | ✗        | ✗        | ✗        |

✓ Available, ☐ In progress or partially available, ✗ TBD


We have config files for ResNet [5], RegNetX [6] and MobileNet [7] backbone architectures. However RegNetX family of backbone architectures strike a good balance between complexity, accuracy and easiness to quantize. Hence we strongly recommend to use that.

The suffix **Lite** or **lite** in our model names or config files indicate that the model uses either Depthwise Convolutions (like in MobileNet models) or grouped convolutions (like in RegNetX models). When the backbone is a MobileNet, Depthwise convolutions are used even in FPN and the detector heads. When the backbone is a RegNetX model, Grouped convolutions are used with the same group size that the RegNetX backbone uses. But for backbones that use regular convolutions (such as ResNet) Depthwise or Grouped convolutions are not used as expected, and they are mostly compatible with the original mmdetection configurations.<br>

We shall add support for additional low complexity models as mmdetection adds support for those architectures. Additional detectors that we are looking forward to see in mmdetection are CenterNet(Objects As Points), EfficientDet and YOLOv3. For more information, see the [roadmap of mmdetection](https://github.com/open-mmlab/mmdetection/issues/2931).


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
|SSD+FPN          |ResNet50      |512x512   |**31.2**      |**30.77**|ssd_resnet_fpn.py                |[link](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-modelzoo/browse/pytorch/vision/object_detection/xmmdet/coco/ssd_resnet_fpn) |
|.
|SSD*             |VGG16         |512x512   |**29.34**     |**98.81**|                                 |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd) |


###### RetinaNet Detector
Please see the reference [2] for algorithmic details of the detector.

|Model Arch       |Backbone Model|Resolution |AP [0.5:0.95]%|Giga MACS |Model Config File                |Download |
|----------       |--------------|-----------|--------------|----------|---------------------------------|---------|
|RetinaNetLite+FPN|RegNetX800MF  |512x512    |              |**6.04**  |retinanet-lite_regnet_fpn_bgr.py |         |
|RetinaNetLite+FPN|RegNetX1.6GF  |768x768    |              |          |retinanet-lite_regnet_fpn.py     |         |
|.
|RetinaNet+FPN*   |ResNet50      |512x512    |**29.7**      |**68.88** |                                 |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) |
|RetinaNet+FPN*   |ResNet50      |768x768    |**34.0**      |**137.75**|                                 |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) |
|RetinaNet+FPN*   |ResNet50      |(1536,768) |**37.0**      |**275.5** |                                 |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) |
<br>


- A resolution range in the tables is indicated with comma (1536,768) or dash (1536-768) - it means that images are resized to fit within this maximum and minimum size - and the aspect ratio of the original image is preserved (keep_ratio=True in config files). Due to this, each resized image may have a different size depending on the aspect ratio of the original image.

- A fixed resolution in the tables is indicated by *width x height* and it means that the inputs are to be resized to that resolution without preserving the aspect ratio of the image (keep_ratio=False in config files). The resolution after resize can be square (example: 512x512) or non-square (example: 768x384).  Detectors such as SSD [1] and EfficientDet (arXiv:1911.09070) [3] uses this kind of fixed resizing. **It is preferable to use this kind of fixed resolution resize for embedded inference** - at least from an accuracy evaluation point of view (where the actual dataset with variable size images may need to be evaluated upon). <br>

- *Mmdetection typically uses a resolution range to train models for most models except SSD. An interesting observation that we had is that such  models trained for non-square resolutions can also be inferred or evaluated using square aspect ratios (with a bit of accuracy drop, of course). This leads to the possibility of reusing the models provided in the [mmdetection model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) for fixed resolution inference as well.* <br>

- The models with a \* were not trained by us, but rather taken from mmdetection model zoo and inference is run at the said resolution.<br>


## References

[1] SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg

[2] Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002, Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár

[3] EfficientDet: Scalable and Efficient Object Detection, https://arxiv.org/abs/1911.09070, Mingxing Tan, Ruoming Pang, Quoc V. Le

[4] Objects as Points, https://arxiv.org/abs/1904.07850, Xingyi Zhou, Dequan Wang, Philipp Krähenbühl

[5] Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

[6] MobileNetV2: Inverted Residuals and Linear Bottlenecks, https://arxiv.org/abs/1801.04381, Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen<br>
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, https://arxiv.org/abs/1704.04861, Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

[7] Designing Network Design Spaces, https://arxiv.org/abs/2003.13678, Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár

[8] ONNX Runtime cross-platform inferencing software, https://github.com/microsoft/onnxruntime