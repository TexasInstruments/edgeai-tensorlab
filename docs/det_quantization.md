# Jacinto-AI-MMDetection Quantization

Post Training Quantization (PTQ) or Quantization Aware Training (QAT) are often required to achieve the best acuracy for inference in fixed point. 

The foundational components for Quantization are provided in [PyTorch-Jacinto-AI-DevKit Main Page](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/pytorch-jacinto-ai-devkit/browse/). This repository uses those tools. Please consult the [documentation on Quantization](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/docs/Quantization.md) to understand the internals of our implementation of QAT.


## Features

|                                                  | Float    | 16 bit   | 8bit     | 4bit     |
|--------------------                              |:--------:|:--------:|:--------:|:--------:|
| Float32 training and test                        |✓         |          |          |          |
| Float16 training and test                        |          |          |          |          |
| Post Training Calibration for Quantization (PTQ) |          | ✓        | ✓        |✗         |
| Quantization Aware Training (QAT)                |          | ✓        | ✓        |✗         |
| Test/Accuracy evaluation of PTQ & QAT models     |          | ✓        | ✓        |✗         |

✓ Available, ☐ In progress or partially available, ✗ TBD


## Training

#### Floating Point Training
- Please see [Usage](./docs/usage.md) for training and testing in floating point with this repository.


#### How to do Quantization

We support two quantization modes: Quantization Aware Training (QAT) and Post Training Calibration for Quantization (PTQ/Calibration). Everything required for quantization is already done in this repository and the only thing that user needs to be do is to set a **quantize** flag appropriately in the config file. If quantize flag is not set, the usual floating point training of evaluation will happen. These are the values of the quantize flag and their meanings:
- False: Conventional floating point training (default).
- True: Quantization Aware Training
- calibration: Post Training Calibration for Quantization (PTQ/Calibration).


#### What is supported for Quantization

- Post Training Calibration for Quantization (PTQ/Calibration): This is a fast method of improving the quantization accuracy of a model. It typically needs only one epoch to produce reasonable accuracy. This uses the module QuantCalibrateModule from PyTorch-Jacinto-AI-DevKit<br>
- Quantization Aware Training (QAT): QAT is the recommended method for improving accuracy with quantization. This provides good accuracy with quantization, but may require several epochs (example: 10 or 25 or even more) of training. This uses the module QuantTrainModule<br>
- Accuracy Evaluation with Quantization: When quantize flag is set in teh config file and when accuracy evaluation script is invoked, accuracy evalatuon with quantization is being done. This uses the module QuantTestModule<br>


#### What is happening behind the scenes   
- After a model is created, it is wrapped in one of the above modules depending on whether the current phase is QAT, PTQ or Evaluation with Quantization.

- Loading of pretrained model or saving of trained model needs slight change when wrapped with the above modules as the original model is inside the wrapper (otherwise the symbols in pretrained will not match).

- QuantCalibrateModule is fast, but QuantTrainModule typically gives better accuracy. QuantTrainModule and QuantTestModule supports multiple gpus, whereas QuantCalibrateModule has the additional limitation that it doesn't support multiple gpus. 

- Training with QuantTrainModule is just like any other training. However using QuantCalibrateModule is a bit different in that it doesn't need backpropagation - so backpropagation is disabled when using QuantCalibrateModule.


## Results

#### Pascal VOC2007 Dataset
- Train on Pascal VOC 2007+2012
- Test on Pascal VOC 2007

|Dataset    |Mode Arch        |Backbone Model |Backbone Stride|Resolution |Acc Float|Acc 8bit Calib|Acc 8bit QAT|Model Config File                      |
|---------  |----------       |-----------    |-------------- |-----------|-------- |-------       |----------  |----------                             |
|VOC2007    |SSD with FPN     |MobileNetV2    |32             |512x512    |76.1     |75.4          |75.4        |configs/ssd/ ssd_mobilenet_fpn.py|
|VOC2007    |SSD with FPN     |RegNet800MF    |32             |512x512    |79.7     |79.0          |79.5        |configs/ssd/ ssd_regnet_fpn.py   |
|VOC2007    |SSD with FPN     |ResNet50       |32             |512x512    |80.5     |77.0          |79.5        |configs/ssd/ ssd_resnet_fpn.py   |
|.
|VOC2007    |SSD              |VGG16          |32             |512x512    |79.8     |              |            |mmdetection/configs/pascal_voc/ ssd512_voc0712.py   |

- Acc Float: MeanAP50(mAP) Accuracy in percentage in this case.
- Acc 8bit Calib: Same metric with 8bit quantization using PTQ/Calibration 
- Acc Float: Same metric with QAT


## References
Please Refer to the [pytorch-jacinto-ai-devkit](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/) and its [Quantization documentation](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/docs/Quantization.md) for further details on Quantization. 