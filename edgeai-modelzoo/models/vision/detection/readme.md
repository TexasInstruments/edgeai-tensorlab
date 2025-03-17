# Object Detection Benchmark

## Introduction
Object Detection from Image (a.k.a. 2D Object Detection) is an important problem in Computer Vision. This page describes some of the popular Object Detection models.


## Datasets

**COCO** dataset: COCO is a large scale object detection, segmentation and captioning dataset. In this page we make use of the detection part of COCO dataset. COCO 2017 dataset is being used, unless otherwise mentioned. Accuracy evaluation is done on the validation split of COCO 2017, unless otherwise mentioned.


## Models 
The models are grouped in terms of repositories used to train them or the repositories through they are made available. 
Note: **Transformer models have been added using the edgeai-hf-transformers repository (see below)**


### edgeai-mmdetection
- [Models Link](./coco/edgeai-mmdet/)
- [**Training Source Code**](https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-mmdetection)

This is our fork of the popular mmdetection training framework for object detection. We provide several optimized, embedded friendly configurations that provide high throughput on our SoCs. One can train and export models to onnx format, and can then be used in our fork of onnxruntime. Most of the models listed here have been trained using configs in [edgeai-mmdetection/configs/edgeailite](https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-mmdetection/tree/master/configs/edgeailite). Please visit the link above for more information.

#### Newer models

| Dataset | Model             | Input Size  | AP[0.5:0.95]%, AP50% | config file | Notes |
|---------|-------------------|-------------|----------------------|-------------|-------|
|         | **YOLOX models** 
| COCO    | YOLOX-tiny-lite     | 416x416     | 24.8, 40.1        | configs_edgeailite/yolox/yolox_tiny_8xb8-300e_coco.py|       |
| COCO    | YOLOX-nano-lite     | 416x416     | 30.5, 47.4        | configs_edgeailite/yolox/yolox_nano_8xb8-300e_coco.py|       |
| COCO    | YOLOX-s-lite       | 640x640     | 38.3, 56.9         | configs_edgeailite/yolox/yolox_s_8xb8-300e_coco.py|       |
| COCO    | YOLOX-m-lite       | 640x640     | 44.4, 62.9         | configs_edgeailite/yolox/yolox_m_8xb8-300e_coco.py|       |
| COCO    | YOLOX-l-lite      | 640x640     | -                   | configs_edgeailite/yolox/yolox_l_8xb8-300e_coco.py|       |
| COCO    | YOLOX-x-lite       | 640x640     | -                  | configs_edgeailite/yolox/yolox_x_8xb8-300e_coco.py|       |
| COCO    | YOLOv7-l-lite       | 640x640     | 45.9, 65.1        | configs_edgeailite/yolov7/yolov7_l_coco_lite.py |       |
| COCO    | YOLOv7-l-orig       | 640x640     | 50.6, 69.3        | configs_edgeailite/yolov7/yolov7_l_coco_orig.py |       |
| COCO    | YOLOv9-s-lite       | 640x640     | 38.3, 54.0        | configs_edgeailite/yolov9/yolov9_s_coco_lite.py |       |
| COCO    | YOLOv9-s-plus       | 640x640     | 40.0, 55.9        | configs_edgeailite/yolov9/yolov9_s_coco_plus.py |       |
|         | **FCOS models** 
| COCO    | FCOS-r50-lite       | 512x512     | 36.6, 56.0          | configs_edgeailite/fcos/fcos_r50-caffe_fpn_bn-head_1x_coco.py|       |
|         | **Centernet models** 
| COCO    | Centernet-r18     | 512x512     | 25.9, 42.6          | configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py|       |
|         | **Efficientdet models** 
| COCO    | Efficientdet-b0-lite    | 512x512     | 28.0, 45.9           | configs_edgeailite/efficientdet/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py|      |
| COCO    | Efficientdet-b1-lite    | 640x640     | -                    | configs_edgeailite/efficientdet/efficientdet_effb1_bifpn_8xb16-crop512-300e_coco.py|      |


### Very small models (not recommended for general use)

| Dataset | Model             | Input Size  | AP[0.5:0.95]%, AP50% | config file | Notes |
|---------|-------------------|-------------|----------------------|-------------|-------|
|         | **YOLOX models** 
| COCO    | YOLOX-femto-lite    | 320x320     |	12.7, 21.9        | configs_edgeailite/yolox/yolox_femto_8xb8-300e_coco.py|       |
| COCO    | YOLOX-pico-lite       | 320x320     | 17.9, 29.4      | configs_edgeailite/yolox/yolox_pico_8xb8-300e_coco.py|       |


#### Other models
| Dataset | Model Name                         | Input Size | GigaMACS  | AP[0.5:0.95]%, AP50% |Available|Notes |
|---------|------------------------------------|------------|-----------|----------------------|---------|----- |
|         | **SSD models**                     
| COCO    | MobileNetV2+FPN+SSDLite            | 768x768    | **5.83**  | **29.0**, 47.8       |         |      |
| COCO    | MobileNetV2+FPN+SSDLite            | 512x512    | **2.29**  | **27.2**, 45.7       |Y        |      |
| COCO    | MobileNetV2+SSDLite                | 512x512    | **1.67**  | **25.1**, 43.1       |Y        |      |
| COCO    | MobileNetV2-0.5+SSDLite            | 320x320    | **0.197** | **11.8**, 23.1       |Y        |      |
| -       
| COCO    | RegNetX200MF+SSDLite               | 320x320    | **0.61**  | **20.7**, 36.7       |Y        |      |
| COCO    | RegNetX400MF+SSDLite               | 320x320    | **1.12**  | **24.1**, 41.0       |         |      |
| COCO    | RegNetX800MF+FPN+SSDLite           | 512x512    | **6.03**  | **32.8**, 52.8       |Y        |      |
| COCO    | RegNetX1.6GF+FPN+SSDLite           | 768x768    | **24.19** | **37.0**, 57.3       |Y        |      |
| COCO    | RegNetX1.6GF+BiFPN168x4+SSDLite    | 768x768    | **25.37** | **39.8**, 58.4       |         |      |
|         | **YOLOV3 models**                  
| COCO    | YOLOv3(LeakyReLU)                  | 416x416    | **33.0**  | **31.0**, 52.1       |         |Fine tuned to use fixed size |
| COCO    | YOLOv3(ReLU)                       | 416x416    | **33.0**  | **30.7**, 51.2       |Y        |Fine tuned to use fixed size and ReLU |
| COCO    | RegNetX800MF+YOLOV3Lite            | 416x416    | **3.6**   | **26.9**, 46.9       |         |      |
| COCO    | RegNetX1.6GF+YOLOV3Lite            | 512x512    | **10.48** | **30.8**, 51.9       |Y        |      |
| -       
| COCO    | ResNet50+FPN+SSD                   | 512x512    | **30.77** | **31.2**, 52.2       |Y        |      |
|         | **RetinaNet models**               
| COCO    | RegNetX800MF+FPN+RetinaNetLite     | 512x512    | **11.08** | **33.0**, 50.8       |Y        |      |
| -       | **YOLOX models**                   
| COCO    | YOLOX-m-Lite                       | 640x640    | **36.9**  | **44.4**, 62.9       |Y        |      |
| COCO    | YOLOX-s-Lite                       | 640x640    | **13.43** | **38.3**, 56.9       |Y        |      |
| COCO    | YOLOX-tiny-Lite                    | 416x416    | **3.240** | **30.5**, 47.4       |Y        |      |
| COCO    | YOLOX-nano-Lite (depthwise=False)  | 416x416    | **1.476** | **24.8**, 40.1       |Y        |      |
| COCO    | YOLOX-pico-Lite (depthwise=False)  | 320x320    | **0.503** | **17.9**, 29.4       |Y        |      |
| COCO    | YOLOX-femto-Lite (depthwise=False) | 320x320    | **0.238** | **12.7**, 21.9       |Y        |      |
|         | **YOLOV3 models**                  
| COCO    | YOLOv3(LeakyReLU)                  |(416,416)  | **33.0**  |**30.9**            |         |      |
| COCO    | YOLOv3(LeakyReLU)                  |(608,608)  | **70.59** |**33.4**            |         |      |
|         | **SSD model**                      
| COCO    | VGG16+SSD                          |512x512    | **98.81** |**29.34**           |         |      |
|         | **RetinaNet models**               
| COCO    | ResNet50+FPN+RetinaNet             |(1536,768) | **275.5** |**37.0**            |         |      |
| COCO    | ResNet50+FPN+RetinaNet             |512x512    | **68.88** |**29.0**            |         |      |


### MMYOLO Models
- See this information thread: https://github.com/TexasInstruments/edgeai-tensorlab/issues/7
- This repository is not maintained anymore - we do not recommend this - use edgeai-mmdetection listed above. 


### edgeai-yolov5 Models
- [Models & Training Source Code](https://github.com/TexasInstruments/edgeai-yolov5)
- This repository is not maintained anymore - we do not recommend this - use edgeai-mmdetection listed above.  

### edgeai-hf-transformers
- [Models Link](./coco/hf-transformers/)
- **[Training Source Code](https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-hf-transformers)**
- Note: the above repository is forked and modified from: https://github.com/huggingface/transformers

|Dataset | Model Name                        | Input Size | GigaMACS | AP[0.5:0.95]%, AP50% |Available|Notes |
|--------|-----------------------------------|------------|----------|----------------------|---------|----- |
|COCO    | facebook/detr-resnet-50           | 768x768    | 86       |**42.0**, 62.4        |         |      |


### Tensorflow TPU Models
- [Models Link](./coco/tf-tpu/)
- [Training Source Code](https://github.com/tensorflow/tpu/tree/master/models/official/detection)


### TensorFlow Model Garden - Object Detection API Models Using Tensorflow 1.0 
- [Models Link](./coco/tf1-models/)
- [Training Source Code](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Training Documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md)
- [Object Detection API Model Zoo For Tensorflow 1.0](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

|Dataset |Model Name                    |Input Size |GigaMACS  |AP[0.5:0.95]%, AP50%|Available|Notes   |
|--------|------------------------------|-----------|----------|--------------------|---------|--------|
|COCO    |ssdlite_mobiledet_dsp_320x320_coco_20200519|320x320|2.818|28.9            |Y        |        |
|COCO    |ssdlite_mobiledet_edgetpu_320x320_coco_20200519|320x320|1.534|25.9        |Y        |        |
|COCO    |ssd_mobilenet_v1_coco_2018_01_28|300x300  |1.237      |**23.0**           |Y        |Also used in MLPerf benchmark |
|COCO    |ssd_mobilenet_v2_300          |300x300    |1.875     |**22.0**            |Y        |Also used in MLPerf benchmark |
|COCO    |ssdlite_mobilenet_v2_coco_2018_05_09|300x300|0.75    |**22.0**            |Y        |        |

### TensorFlow Model Garden - Object Detection API Models Using Tensorflow 2.0
- [Models Link](./coco/tf2-models/)
- [Training Source Code](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Training Documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)
- [Object Detection API Model Zoo For Tensorflow 2.0](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

|Dataset |Model Name                              |Input Size |GigaMACS  |AP[0.5:0.95]%, AP50%|Available|Notes |
|--------|----------------------------------------|-----------|----------|--------------------|---------|----- |
|COCO    |ssd_mobilenet_v1_fpn_640x640_coco17     |640x640    |61.593    |29.1                |Y        |      |
|COCO    |ssd_mobilenet_v2_320x320_coco17         |320x320    |0.773     |20.2                |Y        |      |
|COCO    |ssd_mobilenet_v2_fpnlite_320x320_coco17 |300x300    |0.987     |22.2                |Y        |      |
|COCO    |ssd_mobilenet_v2_fpnlite_640x640_coco17 |640x640    |3.945     |28.2                |Y        |      |
|COCO    |ssd_resnet50_v1_fpn_640x640_coco17      |640x640    |89.229    |34.3                |         |      |

### Models trained using github.com/google/automl
- [**Information about our training**](./coco/google-automl/README.md)
- [Models Link](./coco/google-automl/)
- [More Information - github.com/google/automl](https://github.com/google/automl)


|Dataset |Model Name                              |Input Size |GigaMACS  |AP[0.5:0.95]%, AP50%|Available|Notes |
|--------|----------------------------------------|-----------|----------|--------------------|---------|----- |
|        |**Our training - Lite models**
|COCO    |efficientdet-lite0_bifpn_maxpool2x2_relu_ti-lite|512x512    |**2.50**  |33.61, 52.27        |Y        |ti-lite flavour|
|        |**Model from original repository**
|COCO    |EfficientDet-D0                       |512x512    |          |34.3, 53.0          |         |      |

Note: EfficientDet-Lite (or efficientdet-lite) is the embedded friendly variant of EfficientDet. We have also done some additional modifications on top of the original EfficientDet-Lite to better suite our platform (which we referred to as ti-lite flavour in the table). Please refer to [**information about our training**](./coco/google-automl/README.md) for more details.


## Models From github.com/onnx/models
- [Models Link](./coco/onnx-models/)
- [More Information - github.com/onnx/models](https://github.com/onnx/models)


## Face Detection with EdgeAI-MMDetection
- [Models Link](./widerface/) Trained on WIDER FACE Dataset [9]
- [**Training Source Code**](https://github.com/TexasInstruments/edgeai-mmdetection) These models have been trained using configs in [edgeai-mmdetection/configs/edgeailite](https://github.com/TexasInstruments/edgeai-mmdetection/tree/master/configs/edgeailite).

|Dataset      |Model Name                       |Input Size |GigaMACS  |AP[0.5:0.95]%, AP50%|Available|Notes |
|-------------|---------------------------------|-----------|----------|--------------------|---------|----- |
|             |**SSD models**               
|WIDERFace    |RegNetX800MF+FPN+SSDLite         |512x512    |**5.44**  |**23.7**, 49.2      |         |      |
|WIDERFace    |RegNetX1.6GF+FPN+SSDLite         |768x768    |**10.10** |**27.3**, 54.7      |         |      |
|             |**YOLOX models**
|WIDERFace    |YOLOX-tiny-Lite                  |416x416    |**3.19**  |**23.5**, 49.1      |Y        |      |
|WIDERFace    |YOLOX-tiny-Lite                  |1024x1024  |**19.35** |**---**, 68.6       |         |      |
|WIDERFace    |YOLOX-s-Lite                     |640x640    |**13.29** |**31.62**, 64.4     |Y        |      |
|WIDERFace    |YOLOX-s-Lite                     |1024x1024  |**34.03** |**---**, 72.3       |Y        |      |
|WIDERFace    |YOLOX-m-Lite                     |640x640    |**36.67** |**33.6**, 67.5      |Y        |      |


## Notes
- GigaMACS: Complexity in Giga Multiply-Accumulations (lower is better). This is an important metric to watch out for when selecting models for embedded inference.<br>
- Accuracy for Object Detection on COCO dataset primarily uses two accuracy metrics AP[0.5:0.95] and AP50 (in percentages). AP[0.5:0.95] is the Mean of Average Precision values computed at IOUs ranging from 0.5 to 0.95 and averaged. AP50 is the Average Precision computed at 0.5 IoU. If only one accuracy metric is mentioned in a table cell, then it is AP[0.5:0.95]. Be sure to compare using the same metric when comparing across various detectors or configurations.
- The suffix 'Lite' in the name of models such as SSDLite, RetinaNetLite indicates the use of Depthwise convolutions or Grouped convolutions. If the feature extractor (encoder) uses Depthwise Convolutions, then usually Depthwise convolutions are used throughout such models - even in the neck and decoder. If the feature extractor (encoder) uses grouped convolutions as in the case of RegNetX, then grouped convolutions (with the same group size as that of the feature extractor) are used even in the neck and decoder.<br>
- Lite models also uses ReLU activation (instead of complicated activations such as Swish, HSwish etc.) and also does not use other modules that are unfriendly to embedded inference - such as Squeeze And Excitation (SE).
- A fixed resolution in the tables is indicated by *width x height* and it means that the inputs are to be resized to that resolution without preserving the aspect ratio of the image (keep_ratio=False in config files). The resolution after resize can be square (example: 512x512) or non-square (example: 768x384).  Detectors such as SSD [1] and EfficientDet (arXiv:1911.09070) [3] uses this kind of fixed resizing. **It is preferable to use this kind of fixed resolution resize for embedded inference** - at least from an accuracy evaluation point of view (where the actual dataset with variable size images may need to be evaluated upon).<br>
- A resolution range in the tables is indicated with comma (1536,768) - it means that images are resized to fit within this maximum and minimum size - and the aspect ratio of the original image is preserved (keep_ratio=True in config files). Due to this, each resized image may have a different size depending on the aspect ratio of the original image.<br>
- *Mmdetection typically uses a resolution range to train models for most models except SSD. An interesting observation that we had is that such  models trained for non-square resolutions can also be inferred or evaluated using square aspect ratios (with a bit of accuracy drop, of course). This leads to the possibility of reusing the models provided in the [mmdetection model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) for fixed resolution inference as well.*<br>
- YOLOv3 gives 30.9% AP[0.5:0.95] when evaluated with mmdetection at (416,416). But mmdetection uses variable size resize preserving the aspect ratio which means different images have different input sizes to the models within (416,416) - this kind of inference is not suitable for embedded inference. When inferred at fixed 416x416 resolution, it gives 29.6% AP[0.5:0.95] with mmdetection.<br>


## References

[1] MMDetection: Open MMLab Detection Toolbox and Benchmark, Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin, https://arxiv.org/abs/1906.07155

[2] The PASCAL Visual Object Classes (VOC) Challenge, Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
International Journal of Computer Vision, 88(2), 303-338, 2010, http://host.robots.ox.ac.uk/pascal/VOC/

[2] Microsoft COCO: Common Objects in Context, Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár, https://arxiv.org/abs/1405.0312, https://cocodataset.org/

[3] SSD: Single Shot MultiBox Detector, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, https://arxiv.org/abs/1512.02325

[4] Focal Loss for Dense Object Detection, Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár, https://arxiv.org/abs/1708.02002

[5] Feature Pyramid Networks for Object Detection Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie, https://arxiv.org/abs/1612.03144

[6] YOLOv3: An Incremental Improvement, Joseph Redmon, Ali Farhadi, https://arxiv.org/abs/1804.02767

[7] Objects as Points, Xingyi Zhou, Dequan Wang, Philipp Krähenbühl, https://arxiv.org/abs/1904.07850

[8] EfficientDet: Scalable and Efficient Object Detection, Mingxing Tan, Ruoming Pang, Quoc V. Le, https://arxiv.org/abs/1911.09070

[9] WIDER FACE: A Face Detection Benchmark, Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, http://shuoyang1213.me/WIDERFACE/

