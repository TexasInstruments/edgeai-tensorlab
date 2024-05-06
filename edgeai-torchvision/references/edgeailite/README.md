# EdgeAI-XVision - Examples Models & Training Scripts
Model Training Examples For Embedded AI Development - in PyTorch.

<br><hr><br>

## Introduction
This code provides a set of low complexity deep learning examples and models for low power embedded systems. Low power embedded systems often requires balancing of complexity and accuracy. This is a tough task and requires significant amount of expertise and experimentation. We call this process **complexity optimization**. In addition we would like to bridge the gap between Deep Learning training frameworks and real-time embedded inference by providing ready to use examples and enable **ease of use**. Scripts for training, validation, complexity analysis are also provided. 

<br><hr><br>

## Models and Training
[Image Classification](./docs/image_classification.md)<br>

[Semantic Segmentation](./docs/semantic_segmentation.md)<br>

[Depth Estimation](./docs/depth_estimation.md)<br>

[Motion Segmentation](./docs/motion_segmentation.md)<br>

[Multi Task Estimation](./docs/multi_task_learning.md)<br>

[Object Detection](./docs/object_detection.md)<br>

[Object Keypoint detection / Human Pose Estimation](./docs/keypoint_detection.md)<br>


<br><hr><br>

## Guidelines for Model training & quantization
Quantization (especially 8-bit Quantization) is important to get best throughput for inference. Quantization can be done using either **Post Training Quantization (PTQ)** or **Quantization Aware Training (QAT)**. Guidelines for Model training and tools for QAT are given the **[documentation on Quantization](./edgeai_xvision/xnn/quantization/README.md)**.

- Post Training Quantization (PTQ): TIDL natively supports PTQ - it can take floating point models and can quantize them using advanced calibration methods. In the above link, we have provided guidelines on how to choose models and how to train them for best accuracy with quantization - these guidelines are important to reduce accuracy drop during quantization with **PTQ**. 

- Quantization Aware Training (QAT): In spite of following these guidelines, if there are models that have significant accuracy drop with PTQ, it is possible to improve the accuracy using **QAT**. See the above link for more details.


<br><hr><br>


## Acknowledgements
Apart from [torchvision](https://github.com/pytorch/vision), our source code uses parts of the following open source projects. We would like to thank their authors for making their code bases publicly available.

|Module/Functionality              |Parts of the code borrowed/modified from                                             |
|----------------------------------|-------------------------------------------------------------------------------------|
|Datasets, Models                  |https://github.com/pytorch/vision, https://github.com/ansleliu/LightNet              |
|Training, Validation Engine/Loops |https://github.com/pytorch/examples, https://github.com/ClementPinard/FlowNetPytorch |

<br><hr><br>

