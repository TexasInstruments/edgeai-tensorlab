# Jacinto-AI-ModelZoo

This repository provides a collection of example Deep Learning Models for various Computer Vision tasks. These tasks include image classification, segmentation and detection. The models in this repository can be run either in PC simulation mode or directly in [Jacinto 7](https://training.ti.com/jacinto7) family of SoCs, for example [TDA4VM](https://www.ti.com/product/TDA4VM). 


#### Notice
This repository is part of Jacinto-AI-DevKit, which is a collection of repositories providing Training & Quantization scripts, Model Zoo and Accuracy Benchmarks. If you have not visited the landing page of [**Jacinto-AI-Devkit**](https://github.com/TexasInstruments/jacinto-ai-devkit) please do so before attempting to use this repository.

This repository can be obtained by git clone.
```
git clone https://git.ti.com/git/jacinto-ai/jacinto-ai-modelzoo.git
```

Online Documentation [link](https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/about/)

Tree [view](https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/tree/)


## Introduction
In order to run Deep Neural Networks (a.k.a. DNNs or Deep Learning Models or simply models) on embedded hardware, they need to be converted into embedded friendly formats and optimized. We have converted/exported several models from the original training frameworks in PyTorch, Tensorflow and MxNet into these embedded friendly formats and is being hosted in this repository. In this process we also make sure that these models provide optimized inference speed on our SoCs, so sometimes minor modifications are made to the models wherever necessary. These models provide a good starting point for our customers to explore high performance Deep Learning on our SoCs.

DNNs can be run on our SoCs using RTOS SDK for Jacinto 7 (PROCESSOR-SDK-RTOS-J721E). It can be downloaded from the page for Processor SDK for Jacinto 7 TDA4x a.k.a. **[PROCESSOR-SDK-J721E](https://www.ti.com/tool/PROCESSOR-SDK-J721E)**. 

RTOS SDK for Jacinto 7 TDA4x provides TI Deep Learning Library (TIDL), which is an optimized library that can run Neural Networks on our SoCs. TIDL provides several familiar interfaces for model inference - such as onnxruntime, tflite_runtime, tvm/dlr - apart from its own native interface. All these runtimes that are provided as part of TIDL have extensions on top of public domain runtimes that allow us to offload model execution into our high performance C7x+MMA DSP. For more information how to obtain and use these runtimes, please go through the TIDL documentation in the RTOS SDK. The documentation of TIDL can be seen if you click on the "SDK Documentation" link in the download page for [PROCESSOR-SDK-RTOS-J721E)](https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E)

### Components of this repository
- **Exported DNN models** (eg. tflite onnx and mxnet/json/params formats) - ready to be imported with and used in TIDL.
- Corresponding scripts and config files (for those models) to import/calibrate those models and run benchmarks - provided in a separate repository - [Jacinto-AI-Benchmark](https://git.ti.com/cgit/jacinto-ai/jacinto-ai-benchmark/about/) Please go through the documentation of that repository to understand the usage. 
- Example **pre-imported/calibrated artifacts** for our platform - these artifacts can be directly used in our platform.


**Important Note**: This repository is being made available for experimentation, analysis and research - this is not meant for deployment in production. We do not own the datasets that are used to train or evaluate these models and some of these datasets have restrictions on how they can be used.


## Usage
TIDL documentation (see information above) and test scripts provide information on running Deep Learning models in our SoCs. That is a good starting point to get familiarized with import/calibration and inference of such models.

However, we also provide higher level scripts that help to do inference and accuracy benchmarking on our platform easily, with just a few lines of Python code. These example scripts for Model Import/Calibration, Inference and Accuracy benchmarking are in the repository Jacinto-AI-Benchmark described above.


## Model Zoo Documentation

#### Image Classification
[Image Classification Model Zoo](./models/vision/classification/classification.md)

#### Object Detection
[Object Detection Model Zoo](./models/vision/detection/detection.md)

#### Semantic Segmentation
[Semantic Segmentation Model Zoo](./models/vision/segmentation/segmentation.md)

#### Public Benchmarks
[MLPerf Machine Learning Model Zoo](./models/docs/mlperf/mlperf.md)


## LICENSE
Please see the License under which this repository is made available: [LICENSE](./LICENSE.md)


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

