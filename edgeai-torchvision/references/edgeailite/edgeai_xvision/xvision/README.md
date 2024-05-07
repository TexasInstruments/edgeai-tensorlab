# EdgeAI-TorchVision / EdgeAILite Models & Training
Model Training Examples For Embedded AI Development - in PyTorch.


<br><hr><br>


## Introduction
This code provides a set of low complexity deep learning examples and models for low power embedded systems. Low power embedded systems often requires balancing of complexity and accuracy. This is a tough task and requires significant amount of expertise and experimentation. We call this process **complexity optimization**. In addition we would like to bridge the gap between Deep Learning training frameworks and real-time embedded inference by providing ready to use examples and enable **ease of use**. Scripts for training, validation, complexity analysis are also provided. 


<br><hr><br>


## Models and Training
[Image Classification](docs/pixel2pixel/Image_Classification.md)<br>

[Semantic Segmentation](docs/pixel2pixel/Semantic_Segmentation.md)<br>

[Depth Estimation](docs/pixel2pixel/Depth_Estimation.md)<br>

[Motion Segmentation](docs/pixel2pixel/Motion_Segmentation.md)<br>

[Multi Task Estimation](docs/pixel2pixel/Multi_Task_Learning.md)<br>

[Object Detection](https://github.com/TexasInstruments/edgeai-mmdetection) - this link will take you to another repository, where we have our object detection training scripts.

[Object Keypoint detection / Human Pose Estimation](docs/pixel2pixel/Keypoint_Estimation.md)


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
