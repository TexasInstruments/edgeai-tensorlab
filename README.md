# EdgeAI-ModelZoo

This repository provides a collection of example Deep Neural Network (DNN) Models for various Computer Vision tasks.

In order to run Deep Neural Networks (a.k.a. DNNs or Deep Learning Models or simply models) on embedded hardware, they need to be optimized and converted into embedded friendly formats. We have converted/exported several models from the original training frameworks in PyTorch, Tensorflow and MxNet into these embedded friendly formats and is being hosted in this repository. In this process we also make sure that these models provide optimized inference speed on our SoCs, so sometimes minor modifications are made to the models wherever necessary. These models provide a good starting point for our customers to explore high performance Deep Learning on our SoCs.


### Notice
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai 
- https://dev.ti.com/edgeai/
- https://github.com/TexasInstruments/edgeai
- Important Note: The models in this repository are being made available for experimentation and development  - they are not meant for deployment in production.

<hr>

## Release Notes

Please see [release notes](./docs/release_notes.md) file.

<hr>

## ModelZoo / pre-trained models collection & documentation

#### Image classification
- [Image Classification Models](./models/vision/classification/)

#### Object detection
- [Object Detection Models](./models/vision/detection/)
- [Face Detection Models](./models/vision/detection/) See the section on Face Detection Models at the bottom of the page in this link.

#### Semantic segmentation
- [Semantic Segmentation Models](./models/vision/segmentation/)

#### Depth estimation
- [Depth Estimation Models](./models/vision/depth_estimation/)

#### 3D object detection
- [3D Object Detection Models](./models/vision/detection_3d/)

#### 6D Pose Estimation
- [6D Pose Estimation Models](./models/vision/object_6d_pose/)

#### Public benchmarks
- [MLPerf Machine Learning Models](./models/docs/mlperf/)

<hr>

## Supported SoCs
List of supported SoCs are in [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) 

<hr>

## Compiling models
[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) provide information on compiling models for our SoCs. That is a good starting point to get familiarized with import/calibration and inference of models.

[edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) provides higher level scripts for model compilation, inference and accuracy benchmarking. You can find the compilation settings for these models there. The pre-compiled model artifacts here are compiled using that repository. The compiled artifacts from edgeai-benchmark can be used in EdgeAI SDKs of supported SOCs

This repository contains .link files which have the URLs of actual DNN models. These models are arranged according to task that they are used for and then according to the training repositories that were used to train them. If you are using edgeai-benchmark to run model compilation of run benchmark, you have to *git clone* this repository as well.

<hr>

## Pre-complied model artifacts 
See [pre-compiled model artifacts](./docs/precompiled_modelartifacts.md) that we provide with this repository.

<hr>

## Model inference
[edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) - as mentioned earlier edgeai-benchmark has been used to compile the models in the repository and compiled model artifacts are provided. edgeai-benchmark can also be used for inference & accuracy/performance benchmark of these models on PC or on EVM.

[PROCESSOR-SDK-LINUX-SK-TDA4VM](https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-SK-TDA4VM) - Processor SDK Linux for TDA4VM Starter Kit can seamlessly use model artifacts from this repository. The SDK provides software and tools to let you effectively balance deep learning performance with system power and cost on Texas Instrument’s processors for edge AI applications such as latest TDA4 class of SoCs. We offer a practical embedded inference solution for next-generation vehicles, smart cameras, edge AI boxes, and autonomous machines and robots. In addition to general compute core and deep learning cores accelerator, our processors for edge AI integrate imaging, vision, multimedia cores hardware accelerators and with security enablers and optional microcontrollers for applications that require SIL-3 and ASIL-D functional safety certifications. With a few simple steps you can run high performance computer vision and deep learning demos using a live camera and display.

<hr>

## LICENSE
Please see the License under which this repository is made available: [LICENSE](./LICENSE.md)

<hr>

## References
[1] **ImageNet ILSVRC Dataset**: Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision, 2015. http://www.image-net.org/ <br>

[2] **COCO Dataset**: Microsoft COCO: Common Objects in Context, Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár, https://arxiv.org/abs/1405.0312, https://cocodataset.org/ <br>

[3] **PascalVOC Dataset**: The PASCAL Visual Object Classes (VOC) Challenge, Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A., International Journal of Computer Vision, 88(2), 303-338, 2010, http://host.robots.ox.ac.uk/pascal/VOC/ <br>

[4] **ADE20K Scene Parsing Dataset** Scene Parsing through ADE20K Dataset. Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso and Antonio Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. Semantic Understanding of Scenes through ADE20K Dataset. Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso and Antonio Torralba. International Journal on Computer Vision (IJCV). https://groups.csail.mit.edu/vision/datasets/ADE20K/, http://sceneparsing.csail.mit.edu/ <br>

[5] **Cityscapes Dataset**: M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. https://www.cityscapes-dataset.com/ <br>

[6] **MMDetection: Open MMLab Detection Toolbox and Benchmark**, Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua. arXiv:1906.07155, 2019 <br>

[7] **SSD: Single Shot MultiBox Detector**, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016. <br>

[8] **MLPerf Inference Benchmark**, Vijay Janapa Reddi and Christine Cheng and David Kanter and Peter Mattson and Guenther Schmuelling and Carole-Jean Wu and Brian Anderson and Maximilien Breughe and Mark Charlebois and William Chou and Ramesh Chukka and Cody Coleman and Sam Davis and Pan Deng and Greg Diamos and Jared Duke and Dave Fick and J. Scott Gardner and Itay Hubara and Sachin Idgunji and Thomas B. Jablin and Jeff Jiao and Tom St. John and Pankaj Kanwar and David Lee and Jeffery Liao and Anton Lokhmotov and Francisco Massa and Peng Meng and Paulius Micikevicius and Colin Osborne and Gennady Pekhimenko and Arun Tejusve Raghunath Rajan and Dilip Sequeira and Ashish Sirasao and Fei Sun and Hanlin Tang and Michael Thomson and Frank Wei and Ephrem Wu and Lingjie Xu and Koichi Yamada and Bing Yu and George Yuan and Aaron Zhong and Peizhao Zhang and Yuchen Zhou, arXiv:1911.02549, 2019 <br>

[9] **Pytorch/Torchvision**: Torchvision the machine-vision package of torch, Sébastien Marcel, Yann  Rodriguez, MM '10: Proceedings of the 18th ACM international conference on Multimedia October 2010 Pages 14851488 https://doi.org/10.1145/1873951.1874254, https://pytorch.org/vision/stable/index.html

[10] **TensorFlow Model Garden**: The TensorFlow Model Garden is a repository with a number of different implementations of state-of-the-art (SOTA) models and modeling solutions for TensorFlow users. https://github.com/tensorflow/models <br>

[11] **TensorFlow Object Detection API**: Speed/accuracy trade-offs for modern convolutional object detectors. Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z, Song Y, Guadarrama S, Murphy K, CVPR 2017, https://github.com/tensorflow/models/tree/master/research/object_detection <br>

[12] **Tensorflow DeepLab**: DeepLab: Deep Labelling for Semantic Image Segmentation https://github.com/tensorflow/models/tree/master/research/deeplab

[13] **TensorFlow Official Model Garden**, Chen Chen and Xianzhi Du and Le Hou and Jaeyoun Kim and Pengchong, Jin and Jing Li and Yeqing Li and Abdullah Rashwan and Hongkun Yu, 2020, https://github.com/tensorflow/models/tree/master/official <br>

[14] **GluonCV**: GluonCV and GluonNLP: Deep Learning in Computer Vision and Natural Language Processing
Jian Guo, He He, Tong He, Leonard Lausen, Mu Li, Haibin Lin, Xingjian Shi, Chenguang Wang, Junyuan Xie, Sheng Zha, Aston Zhang, Hang Zhang, Zhi Zhang, Zhongyue Zhang, Shuai Zheng, Yi Zhu, https://arxiv.org/abs/1907.04433

[15] **WIDER FACE**: A Face Detection Benchmark, Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, http://shuoyang1213.me/WIDERFACE/

