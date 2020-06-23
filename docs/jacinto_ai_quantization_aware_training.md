# Jacinto-AI-MMDetection Quantization

Quantization Aware Training (QAT) is often required to achieve the best acuracy for inference in fixed point. 

We have developed several tools to aid QAT and is provided in [PyTorch-Jacinto-AI-DevKit Main Page](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/pytorch-jacinto-ai-devkit/browse/). Please consult the documentation on Quantization provided there to understand the internals of our implementation of QAT.

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
- Floating point training and testing can be done using the scripts provided in the [scripts](../../scripts) folder. Please consult [Usage/Instructions](jacinto_ai/jacinto_ai_mmdetection_usage.md) for more information.

#### Quantization Aware Training
- The following are the main tools provided for Quantization are:<br>
    - Quantization Aware Training (QAT): QuantTrainModule<br>
    - Post Training Calibration for Quantization (PTQ/Calibration): QuantCalibrateModule<br>
    - Accuracy Test with Quantization: QuantTestModule<br>
        
- After a model is created in pytorch-mmdetection, it is wrapped in one of the above modules depending on whether the current phase is QAT, PTQ or Test with Quantization.

- Loading of pretrained model or saving of trained model needs slight change when wrapped with the above modules as the original model is inside the wrapper (otherwise the symbols in pretrained will not match).

- QuantCalibrateModule is fast, but QuantTrainModule typically gives better accuracy. QuantTrainModule and QuantTestModule supports multiple gpus, whereas QuantCalibrateModule has the additional limitation that it doesn't support multiple gpus. 

- Training with QuantTrainModule is just like any other training. However using QuantCalibrateModule is a bit different in that it doesn't need backpropagation - so backpropagation is disabled when using QuantCalibrateModule.

- We have derived additional classes from these modules called MMDetQuantTrainModules, MMDetQuantCalibrateModules and MMDetQuantTestModules because the forward call of models in mmdetection is a bit different. For example for tracing through the model, a forward_dummy method is used in mmdetection. Also the way arguments are passed to forward call are also a bit different.  

- Training can be done by using the scripts ./scripts/train_main.py or ./scripts/train_dist.py. 

- To enable quantization during training, the quantize flag in the config file being used must be a "truth value in Python" - i.e. a string or True or something like that. If quantize is commented out or if it is False, None etc, quantization will not be performed.

## Testing
- Test can be done by using the scripts ./scripts/test_main.py or ./scripts/test_dist.py

- To enable quantization during test, the quantize flag in the config file being used must be a "truth value in Python" - i.e. a string or True or something like that. If quantize is commented out or if it is False, None etc, quantization will not be performed.

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
Please Refer to the [pytorch-jacinto-ai-devkit](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/) and its [Quantization documentation](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/docs/Quantization.md) for further details on the internals of these Quant Modules. 