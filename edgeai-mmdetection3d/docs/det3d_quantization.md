# Quantization Aware Training of 3D Object Detection Models

Quantization (especially 8-bit Quantization) is important to get best throughput for inference. Quantization can be done using either **Post Training Quantization (PTQ)** or **Quantization Aware Training (QAT)**. TIDL natively supports PTQ to quantize floating point models. 

The guidelines provided in the **[quantization documentation](https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Quantization.md)** are important to get the best accuracy with quantization. 

However, in case there is significant accuracy drop native PTQ in TIDL, then it is possible to improve the accuracy using QAT. Please read more about QAT at [edgeai-torchvision](https://github.com/TexasInstruments/edgeai-torchvision) and its [quantization documentation](https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Quantization.md). 

Although this repository does QAT with the help of [edgeai-torchvision](https://github.com/TexasInstruments/edgeai-torchvision), the quantized weights are still kept as discrete floating point values. Activation range information is inserted into the model using Clip functions, wherever appropriate. TIDL will convert these QAT models to fixed point using these ranges and the discrete weights.


## Features in this repository

|                                                              | Float    | 16 bit   | 8bit     |
|--------------------                                          |:--------:|:--------:|:--------:|
| Float32 training and test                                    |✓         |          |          |
| Quantization Aware Training (QAT)                            |          | ✓        | ✓        |
| Test/Accuracy evaluation of QAT                              |          | ✓        | ✓        |

✓ Available, ☐ In progress or partially available, ✗ TBD


## Training

#### Floating Point and QAT Training
- Please see [Usage](./det3d_usage.md) for training and testing in floating point and QAT with this repository.


#### How to do Quantization Aware Training

Everything required for quantization is already done in this repository and the only thing that user needs to be do is to set a **quantize** flag appropriately in the config file. If quantize flag is not set, the usual floating point training of evaluation will happen. These are the values of the quantize flag and their meanings:
- False: Conventional floating point training (default).
- True or 'training': Quantization Aware Training (QAT)
- 'calibration': Post Training Calibration for Quantization (denoted as Calibration/PTQ) - this is an intermediate method that is close to PTQ - fast, but not as accurate as QAT. **This method is not supported currently**

Accuracy Evaluation with Quantization: If quantize flag is set in the config file when test script is invoked, accuracy evalatuon with quantization is being done.

**To integrate QAT in another repository user can refer this repository. Codes under the flag "quantize" needs to be carefully looked and to be appropriately taken in another repository as required.**

#### What is happening behind the scenes   
- EdgeAI-Torchvision provides several modules to aid Quantization: QuantTrainModule for QAT, QuantCalibrateModule for Calibration/PTQ and QuantTestModule for accuracy evaluation with Quantization. 

- If the quantize flag is set in the config file being used, the model is wrapped in one of the Quantization modules depending on whether the current phase is QAT, Calibration/PTQ or accuracy evaluation with Quantization.

- Loading of pretrained model or saving of trained model needs slight change when wrapped with the above modules as the original model is inside the wrapper (otherwise the symbols in pretrained will not match).

- Training with QuantTrainModule is just like any other training. However using QuantCalibrateModule is a bit different in that it doesn't need backpropagation - so backpropagation is disabled when using Calibration/PTQ.

All this has been taken care already in the code and the description in this section is for information only. 


## References
[1] [PyTorch-Jacinto-AI-DevKit](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/) and its [Quantization documentation](https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/docs/Quantization.md). 

[2] PointPillars: Fast Encoders for Object Detection from Point Clouds, https://arxiv.org/abs/1812.05784, Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom

[3] PointPainting: https://arxiv.org/abs/1911.10150