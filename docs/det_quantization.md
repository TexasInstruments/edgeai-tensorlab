# Quantization Aware Training of Object Detection Models

Post Training Calibration for Quantization (PTQ) or Quantization Aware Training (QAT) are often required to achieve the best acuracy for inference in fixed point. This repository can do QAT and/or PTQ on object detection models trained here. PTQ can easily be performed on the inference engine itself, it need not be done using a training framework like this. While PTQ is fast, QAT provides the best accuracy. Due to these reasons, we shall focus on QAT in this repository. 

Although  repository does Quantization, the data is still kept as discrete floating point values. Activation range information is inserted into the model using Clip functions, wherever appropriate.  

The foundational components for Quantization are provided in [PyTorch-Jacinto-AI-DevKit](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/pytorch-jacinto-ai-devkit/browse/). This repository uses Quantization tools from there. Please consult the [documentation on Quantization](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/docs/Quantization.md) to understand the internals of our implementation of QAT / PTQ.


## Features

|                                                  | Float    | 16 bit   | 8bit     | 4bit     |
|--------------------                              |:--------:|:--------:|:--------:|:--------:|
| Float32 training and test                        |✓         |          |          |          |
| Float16 training and test                        |          |          |          |          |
| Post Training Calibration for Quantization (PTQ) |          | ☐        | ☐        |✗         |
| Quantization Aware Training (QAT)                |          | ✓        | ✓        |✗         |
| Test/Accuracy evaluation of QAT / PTQ models     |          | ✓        | ✓        |✗         |

✓ Available, ☐ In progress or partially available, ✗ TBD


## Training

#### Floating Point Training
- Please see [Usage](./docs/usage.md) for training and testing in floating point with this repository.


#### How to do Quantization

Everything required for quantization is already done in this repository and the only thing that user needs to be do is to set a **quantize** flag appropriately in the config file. If quantize flag is not set, the usual floating point training of evaluation will happen. These are the values of the quantize flag and their meanings:
- False: Conventional floating point training (default).
- True or 'training': Quantization Aware Training (QAT)
- 'calibration': Post Training Calibration for Quantization (PTQ).

Accuracy Evaluation with Quantization: If quantize flag is set in the config file when test script is invoked, accuracy evalatuon with quantization is being done.

#### What is happening behind the scenes   
- PyTorch-Jacinto-AI-DevKit provides several modules to aid Quantization: QuantTrainModule for QAT, QuantCalibrateModule for PTQ and QuantTestModule for accuracy evaluation with Quantization. 

- QuantTrainModule and QuantTestModule supports multiple gpus, whereas QuantCalibrateModule has the additional limitation that it doesn't support multiple gpus. But since PTQ is fast, this is not a real issue.

- After a model is created, it is wrapped in one of the Quantization modules depending on whether the current phase is QAT, PTQ or accuracy evaluation with Quantization.

- Loading of pretrained model or saving of trained model needs slight change when wrapped with the above modules as the original model is inside the wrapper (otherwise the symbols in pretrained will not match).

- Training with QuantTrainModule is just like any other training. However using QuantCalibrateModule is a bit different in that it doesn't need backpropagation - so backpropagation is disabled when using PTQ.

All this has been taken care already in the code and the description in this section is for information only. 

#### Results for COCO 2017 Dataset

###### Single Shot Mult-Box Detector (SSD) 
Please see the reference [2] for algorithmic details of the detector.

|Model Arch       |Backbone Model|Resolution |Giga MACS |Float AP [0.5:0.95]%|8-bit QAT AP [0.5:0.95]%|Download |
|----------       |--------------|-----------|----------|--------------------|------------------------|---------|
|SSDLite+FPN      |RegNetX800MF  |512x512    |**6.03**  |**29.9**            |**29.6**                |         |
|SSDLite+FPN      |RegNetX1.6GF  |768x768    |          |                    |                        |         |
|.
|SSD+FPN          |ResNet50      |512x512    |**30.77** |**31.2**            |                        |[link](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-modelzoo/browse/pytorch/vision/object_detection/xmmdet/coco/ssd_resnet_fpn) |


###### RetinaNet Detector
Please see the reference [3] for algorithmic details of the detector.

|Model Arch       |Backbone Model|Resolution |Giga MACS |Float AP [0.5:0.95]%|8-bit QAT AP [0.5:0.95]%|Download |
|----------       |--------------|-----------|----------|--------------------|------------------------|---------|
|RetinaNetLite+FPN|RegNetX800MF  |512x512    |**6.04**  |                    |                        |         |
|RetinaNetLite+FPN|RegNetX1.6GF  |768x768    |          |                    |                        |         |
|.
|RetinaNet+FPN*   |ResNet50      |512x512    |**68.88** |**29.7**            |                        |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) |
|RetinaNet+FPN*   |ResNet50      |768x768    |**137.75**|**34.0**            |                        |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) |
|RetinaNet+FPN*   |ResNet50      |(1536,768) |**275.5** |**37.0**            |                        |[link](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) |
<br>

- Float AP [0.5:0.95]% : COCO Mean Average Precision metric in percentage for IoU range [0.5:0.95], with the floating point model.
- 8-bit QAT AP [0.5:0.95]% : COCO Mean Average Precision metric in percentage for IoU range [0.5:0.95], with the the 8-bit Quantization Aware Trained Model.


## References
[1] [PyTorch-Jacinto-AI-DevKit](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/) and its [Quantization documentation](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/docs/Quantization.md). 

[2] SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg

[3] Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002, Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
