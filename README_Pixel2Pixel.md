# EdgeAI-TorchVision
Training & Quantization Tools For Embedded AI Development - in PyTorch.


<br><hr><br>


## Introduction
This code provides a set of low complexity deep learning examples and models for low power embedded systems. Low power embedded systems often requires balancing of complexity and accuracy. This is a tough task and requires significant amount of expertise and experimentation. We call this process **complexity optimization**. In addition we would like to bridge the gap between Deep Learning training frameworks and real-time embedded inference by providing ready to use examples and enable **ease of use**. Scripts for training, validation, complexity analysis are also provided. 

This code also includes tools for **Quantization Aware Training** that can output an 8-bit Quantization friendly model - these tools can be used to improve the quantized accuracy and bring it near floating point accuracy. For more details, please refer to the section on [Quantization](docs/pixel2pixel/Quantization.md).


<br><hr><br>


## Examples
Below are some of the examples are currently available. Click on each of the links above to go into the full description of the example.

[Image Classification](docs/pixel2pixel/Image_Classification.md)<br>

[Semantic Segmentation](docs/pixel2pixel/Semantic_Segmentation.md)<br>

[Depth Estimation](docs/pixel2pixel/Depth_Estimation.md)<br>

[Motion Segmentation](docs/pixel2pixel/Motion_Segmentation.md)<br>

[Multi Task Estimation](docs/pixel2pixel/Multi_Task_Learning.md)<br>

[Object Detection](https://github.com/TexasInstruments/edgeai-mmdetection) - this link will take you to another repository, where we have our object detection training scripts.

[Object Keypoint detection / Human Pose Estimation](docs/pixel2pixel/Keypoint_Estimation.md)

[**Quantization**](docs/pixel2pixel/Quantization.md)<br>

<br><hr><br>


## Model Quantization
Quantization (especially 8-bit Quantization) is important to get best throughput for inference. Quantization can be done using either **Post Training Quantization (PTQ)** or **Quantization Aware Training (QAT)**.

[TI Deep Learning Library (TIDL)](https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/latest/exports/docs/psdk_rtos_auto/docs/user_guide/sdk_components.html#ti-deep-learning-library-tidl) that is part of the [Processor SDK RTOS for Jacinto7](https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/latest/exports/docs/psdk_rtos_auto/docs/user_guide/index.html) natively supports **PTQ** - TIDL can take floating point models and can quantize them using advanced calibration methods. 

We have  guidelines on how to choose models and how train them to get best accuracy with Quantization. It is unlikely that there will be significant accuracy drop with **PTQ** if these guidelines are followed. In spite of this, if there are models that have significant accuracy drop with quantization, it is possible to improve the accuracy using **QAT**. Please read more details in the documentation on **[Quantization](docs/pixel2pixel/Quantization.md)**.

<br><hr><br>


## Additional Information
- Some of the common training and validation commands are provided in shell scripts (.sh files) in the root folder. <br>
- Landing Page for our models, training and quantization scripts: [https://github.com/TexasInstruments/edgeai](https://github.com/TexasInstruments/edgeai) <br>

<br><hr><br>


## Acknowledgements
Apart from [torchvision](https://github.com/pytorch/vision), our source code uses parts of the following open source projects. We would like to thank their authors for making their code bases publicly available.

|Module/Functionality              |Parts of the code borrowed/modified from                                             |
|----------------------------------|-------------------------------------------------------------------------------------|
|Datasets, Models                  |https://github.com/pytorch/vision, https://github.com/ansleliu/LightNet              |
|Training, Validation Engine/Loops |https://github.com/pytorch/examples, https://github.com/ClementPinard/FlowNetPytorch |

<br><hr><br>


## License

Please see the [LICENSE](./LICENSE) file for more information about the license under which this code is made available.