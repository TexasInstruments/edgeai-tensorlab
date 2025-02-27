# edgeai-tensorvision
* Examples Deep Neural Network Models for Computer Vision Tasks
* Model Training Scripts & Examples For Embedded AI Models - in PyTorch.

<hr>

## Introduction
This code provides a set of low complexity deep learning examples and models for low power embedded systems. Low power embedded systems often requires balancing of complexity and accuracy; this is a tough task that requires significant amount of expertise and experimentation. In addition, we would like to bridge the gap between Deep Learning training frameworks and real-time embedded inference by providing ready to use examples and scripts for training, validation and complexity analysis. 

<hr>

## Models and Training
[Image Classification](./docs/image_classification.md)<br>

[Semantic Segmentation](./docs/semantic_segmentation.md)<br>

[Depth Estimation](./docs/depth_estimation.md)<br>

[Motion Segmentation](./docs/motion_segmentation.md)<br>

[Multi Task Estimation](./docs/multi_task_learning.md)<br>

[Object Detection](./docs/object_detection.md)<br>

[Keypoint detection / Human Pose Estimation](./docs/keypoint_detection.md)<br>


<hr>


## Acknowledgements
This also uses parts of the following open source projects. We would like to thank their authors for making their code bases publicly available.

|Module/Functionality              |Parts of the code borrowed/modified from                                             |
|----------------------------------|-------------------------------------------------------------------------------------|
|Datasets, Models                  |https://github.com/pytorch/vision, https://github.com/ansleliu/LightNet              |
|Training, Validation Engine/Loops |https://github.com/pytorch/examples, https://github.com/ClementPinard/FlowNetPytorch |

<hr>

