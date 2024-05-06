# Object Detection Model Zoo

MMDetection has a huge Model Zoo, supporting a lot of models. Many of them are high complexity models that are not suitable for embedded scenarios that require high throughput. (Please refer to the mmdetection documentation link above for details). However, here we primarily list speed/accuracy optimized models that we have trained ourselves or is recommending from another location.


## Features

| Detector/Backbone           | ResNet   | RegNetX  | MobileNet| Other    |
|-----------------------------|:--------:|:--------:|:--------:|:--------:|
| SSD                         | ✓        | ✓        | ✓        |          |
| RetinaNet                   | ☐        | ✓        | ✗        |          |
| CenterNet(Objects As Points)| ✗        | ✗        | ✗        |          |
| EfficientDet                | ✗        | ✗        | ✗        |          |
| YOLOv3                      | ✗        | ✗        | ✗        |✓         |
| YOLOX                       | ✗        | ✗        | ✗        |✓         |
| Faster R-CNN                | ✗        | ✗        | ✗        |          |
| Mask R-CNN                  | ✗        | ✗        | ✗        |          |

✓ Available, ☐ In progress or partially available, ✗ TBD


We have config files for ResNet, RegNetX and MobileNet backbone architectures. The detector meta architectures that we list here include SSD, RetinaNet, YOLOV3 and YOLOX. Overall, the RegNetX family of backbones strike a good balance between complexity, accuracy and easiness of quantization. Among the detector meta architectures listed, YOLOX is one of the best.

We shall add support for additional low complexity models as mmdetection adds support for those architectures. Additional detectors that we are looking forward to see in mmdetection are CenterNet(Objects As Points) and EfficientDet. For more information, see the [roadmap of mmdetection](https://github.com/open-mmlab/mmdetection/issues/2931).


#### Results for COCO 2017 Dataset
- Train on COCO 2017 Train Set
- Evaluate on COCO 2017 Validation Set


#### Notes
- Accuracy for Object Detection on COCO dataset uses the metrics AP[0.5:0.95], AP50 (in percentages). AP[0.5:0.95] is the MeanAveragePrecision values computed at IOUs ranging from 0.5 to 0.95 and averaged. AP50 is the MeanAveragePrevision computed at 0.5 IoU. 
- If only one accuracy metric is mentioned in a table cell, then it is AP[0.5:0.95]. Be sure to compare using the same metric when comparing across various detectors or configurations.


#### Results

|Dataset |Model Name                    |Input Size |GigaMACS  |Accuracy%      |Config File |
|--------|------------------------------|-----------|----------|---------------|------------|
|COCO    |RegNetX400MF+FPN+SSDLite      |384x384    |**2.13**  |**27.2**, 45.0 |configs/edgeailite/ssd/ssd_regnet_fpn_bgr_lite.py  |
|COCO    |RegNetX800MF+FPN+SSDLite      |512x512    |**6.03**  |**32.8**, 52.8 |configs/edgeailite/ssd/ssd_regnet_fpn_bgr_lite.py  |
|COCO    |RegNetX1.6GF+FPN+SSDLite      |768x768    |**24.19** |**37.0**, 57.3 |configs/edgeailite/ssd/ssd_regnet_fpn_bgr_lite.py  |
|-
|COCO    |MobileNetV2+SSDLite           |512x512    |**1.74**  |**22.2**       |configs/edgeailite/ssd/ssd_mobilenet_lite.py  |
|COCO    |MobileNetV2+FPN+SSDLite       |512x512    |**2.29**  |**26.0**       |configs/edgeailite/ssd/ssd_mobilenet_fpn_lite.py  |
|-
|COCO    |ResNet50+FPN+SSD              |512x512    |**30.77** |**31.2**, 52.2 |configs/edgeailite/ssd/ssd_resnet_fpn.py |
|COCO    |VGG16+SSD                     |512x512    |**98.81** |**29.34**      |         |
|-
|COCO    |RegNetX800MF+FPN+RetinaNetLite|512x512    |**11.08** |**33.0**, 50.8 |configs/edgeailite/retinanet/retinanet_regnet_fpn_bgr_lite.py |
|COCO    |RegNetX1.6GF+FPN+RetinaNetLite|768x768    |          |               |configs/edgeailite/retinanet/retinanet_regnet_fpn_bgr_lite.py |
|-
|COCO    |ResNet50+FPN+RetinaNet        |512x512    |**68.88** |**29.0**       |configs/edgeailite/retinanet/retinanet_resnet_fpn.py |
|COCO    |ResNet50+FPN+RetinaNet        |768x768    |**137.75**|**34.0**       |configs/edgeailite/retinanet/retinanet_resnet_fpn.py |
|COCO    |ResNet50+FPN+RetinaNet        |(1536,768) |**275.5** |**37.0**       |configs/edgeailite/retinanet/retinanet_resnet_fpn.py |
|-
|COCO    |RegNet1.6GF+YOLO3+Lite        |416x416    |**10.48** |**30.8**, 52.00|configs/edgeailite/yolov3/yolov3_regnet_bgr_lite.py |
|COCO    |YOLO3+ReLU                    |416x416    |**33.0**  |**30.7**, 51.30|configs/edgeailite/yolov3/yolov3_d53_relu.py |
|COCO    |YOLOv3                        |416x416    |**33.0**  |**29.6**       |configs/edgeailite/yolov3/yolov3_d53.py |
|COCO    |YOLOv3                        |(416,416)  |**33.0**  |**30.9**       |configs/edgeailite/yolov3/yolov3_d53.py |
|COCO    |YOLOv3                        |(608,608)  |**70.59** |**33.4**       |configs/edgeailite/yolov3/yolov3_d53.py |
|-
|COCO    |YOLOX-s-Lite                  |640x640    |**13.43** |**38.2**, 57.0 |configs/edgeailite/yolox/yolox_s_lite.py |
|COCO    |YOLOX-tiny-Lite               |416x416    |**3.240** |**29.5**, 46.4 |configs/edgeailite/yolox/yolox_tiny_lite.py |
|COCO    |YOLOX-nano-Lite               |416x416    |**0.552** |**20.9**, 35.2 |configs/edgeailite/yolox/yolox_nano_lite.py |

- The suffix 'Lite' in the name of models indicates the use of Depthwise convolutions or Grouped convolutions. If the feature extractor (encoder) uses Depthwise Convolutions (eg. MobileNet), then Depthwise convolutions are used throughout such models - even in the neck and decoder. If the feature extractor (encoder) uses grouped convolutions as in the case of RegNetX, then grouped convolutions (with the same group size as that of the feature extractor) are used even in the neck and decoder.<br>
- GigaMACS: Complexity in Giga Multiply-Accumulations (lower is better). This is an important metric to watch out for when selecting models for embedded inference.<br>
- Accuracy%: Original Floating Point Validation Accuracy obtained after training.<br>
<br>
- A resolution range in the tables is indicated with comma (1536,768) or dash (1536-768) - it means that images are resized to fit within this maximum and minimum size - and the aspect ratio of the original image is preserved (keep_ratio=True in config files). Due to this, each resized image may have a different size depending on the aspect ratio of the original image.<br>
- A fixed resolution in the tables is indicated by *width x height* and it means that the inputs are to be resized to that resolution without preserving the aspect ratio of the image (keep_ratio=False in config files). The resolution after resize can be square (example: 512x512) or non-square (example: 768x384).  Detectors such as SSD [1] and EfficientDet (arXiv:1911.09070) [3] uses this kind of fixed resizing. **It is preferable to use this kind of fixed resolution resize for embedded inference** - at least from an accuracy evaluation point of view (where the actual dataset with variable size images may need to be evaluated upon).<br>
- *Mmdetection typically uses a resolution range to train models for most models except SSD. An interesting observation that we had is that such  models trained for non-square resolutions can also be inferred or evaluated using square aspect ratios (with a bit of accuracy drop, of course). This leads to the possibility of reusing the models provided in the [mmdetection model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) for fixed resolution inference as well.*<br>
- YOLOv3 gives 30.9% AP[0.5:0.95] when evaluated with mmdetection at (416,416). But mmdetection uses variable size resize preserving the aspect ratio which means different images have different input sizes to the models within (416,416) - this kind of inference is not suitable for embedded inference. When inferred at fixed 416x416 resolution, it gives 29.6% AP[0.5:0.95] with mmdetection.<br>
- All models in the above table are not trained for the same number of epochs. We recommend at least 120 to 240 epochs of training to get good results for MobileNet and RegNetX models (although 60 epochs of training can give reasonable results). 


## References

[1] MMDetection: Open MMLab Detection Toolbox and Benchmark, Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin, https://arxiv.org/abs/1906.07155

[2] The PASCAL Visual Object Classes (VOC) Challenge, Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
International Journal of Computer Vision, 88(2), 303-338, 2010, http://host.robots.ox.ac.uk/pascal/VOC/

[2] Microsoft COCO: Common Objects in Context, Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár, https://arxiv.org/abs/1405.0312

[3] SSD: Single Shot MultiBox Detector, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, https://arxiv.org/abs/1512.02325

[4] Focal Loss for Dense Object Detection, Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár, https://arxiv.org/abs/1708.02002

[5] Feature Pyramid Networks for Object Detection Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie, https://arxiv.org/abs/1612.03144

[6] YOLOv3: An Incremental Improvement, Joseph Redmon, Ali Farhadi, https://arxiv.org/abs/1804.02767

[7] Objects as Points, Xingyi Zhou, Dequan Wang, Philipp Krähenbühl, https://arxiv.org/abs/1904.07850

[8] EfficientDet: Scalable and Efficient Object Detection, Mingxing Tan, Ruoming Pang, Quoc V. Le, https://arxiv.org/abs/1911.09070