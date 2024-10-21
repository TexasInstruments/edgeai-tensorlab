# EdgeAI-Benchmark

### Notice
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai
- https://github.com/TexasInstruments/edgeai
- https://dev.ti.com/edgeai/

The Python version requirement has changed to 3.10 for SDK/TIDL 9.0 onwards. Please see [setup instructions](./docs/setup_instructions.md) for more information.

<hr>

This repository provides a collection of scripts for various image recognition tasks such as classification, segmentation, detection and keypoint detection. 
- These scripts can be used for Model Compilation, Inference, Accuracy & Performance benchmarking of Deep Neural Networks (DNN). 
- Aspects such dataset loading, pre-processing and post-processing as taken care for the models in our model zoo.
- These benchmarks in this repository can be run either in PC simulation mode or on device. 

Getting the correct functionality and accuracy with DNN Models is not easy. Several aspects such as dataset loading, pre-processing and post-processing operations have to be matched to that of the original training framework to get meaningful functionality and accuracy. There is much difference in these operations across various popular models and much effort has gone into matching that functionality.

<hr>

## Release Notes


<hr>

## Important features:
- Runs on both PC Simulation (model compilation and inference) and on EVM (model inference only).
- This package can be used for accuracy and performance (inferene time) estimates.
- Most of the models in TI ModelZoo [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modelzoo) is supported off-the-shelf in this package. Custom model benchmark can also be easily done (please refer to the documentation and example).
- Uses [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) for model compilation and inference. edgeai-tidl-tools can take a float model and compile it using PTQ (with an iterative calibration procedure) to an INT model for use on target. It can also accept a pre-quantized model to avoid the iterative calibration, so that the compilation is instantaneous. 
- Read more about quantization in general and specifically about pre-quantized models at [edgeai-modeloptimization/torchmodelopt](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modeloptimization/torchmodelopt)

<hr>

## Supported SOCs
At the moment, this repository supports compilation and inference for the following SoCs: **TDA4VM**, **AM68A**, **AM62A**, *AM67A**, **AM69A**, **AM62**

A reference to <SOC> in this repository as commandline argument to the scripts refer to one of these SoCs.

This <SOC> argument is used for multiple purposes:
- To set TIDL_TOOLS_PATH and LD_LIBRARY_PATH used to point to the correct tidl_tools for a device
- To choose the correct preset (of compilation flags) from the dictionary TARGET_DEVICE_SETTINGS_PRESETS in [constants.py](./edgeai_benchmark/constants.py)

More details regarding SoCs and devices can be seen at the [EdgeAI landing repository](https://github.com/TexasInstruments/edgeai/blob/main/readme_sdk.md).

<hr>

## Version information
By default, the tidl_tools that are installed are for the latest EdgeAI-SDK/TIDL release. however, if you would like to use an older version of tidl_tools, checkout the corresponding git branch and use that.

## Setup on PC
See the [setup instructions](./docs/setup_instructions.md)


## Usage on PC
See the [usage instructions](./docs/usage.md)

<hr>

## Compiling Custom Models on PC
See the **[instructions to compile custom models](./docs/custom_models.md)**


## Pre-Complied Model Artifacts 
See [pre-compiled model artifacts](./docs/precompiled_modelartifacts.md)


## Setup and Usage on development board/EVM
The compiled models can be used for inference on development board/EVM. See **[setup and usage instruction for EVM](./docs/usage_evm.md)**

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

[8] **Pytorch/Torchvision**: Torchvision the machine-vision package of torch, Sébastien Marcel, Yann  Rodriguez, MM '10: Proceedings of the 18th ACM international conference on Multimedia October 2010 Pages 14851488 https://doi.org/10.1145/1873951.1874254, https://pytorch.org/vision/stable/index.html

[8] **TensorFlow Model Garden**: The TensorFlow Model Garden is a repository with a number of different implementations of state-of-the-art (SOTA) models and modeling solutions for TensorFlow users. https://github.com/tensorflow/models <br>

[9] **TensorFlow Object Detection API**: Speed/accuracy trade-offs for modern convolutional object detectors. Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z, Song Y, Guadarrama S, Murphy K, CVPR 2017, https://github.com/tensorflow/models/tree/master/research/object_detection <br>

[10] **Tensorflow DeepLab**: DeepLab: Deep Labelling for Semantic Image Segmentation https://github.com/tensorflow/models/tree/master/research/deeplab

[11] **TensorFlow Official Model Garden**, Chen Chen and Xianzhi Du and Le Hou and Jaeyoun Kim and Pengchong, Jin and Jing Li and Yeqing Li and Abdullah Rashwan and Hongkun Yu, 2020, https://github.com/tensorflow/models/tree/master/official <br>

[12] **GluonCV**: GluonCV and GluonNLP: Deep Learning in Computer Vision and Natural Language Processing
Jian Guo, He He, Tong He, Leonard Lausen, Mu Li, Haibin Lin, Xingjian Shi, Chenguang Wang, Junyuan Xie, Sheng Zha, Aston Zhang, Hang Zhang, Zhi Zhang, Zhongyue Zhang, Shuai Zheng, Yi Zhu, https://arxiv.org/abs/1907.04433

[13] **MMPose: Open-source toolbox for pose estimation**, Collection of different models and post processing techniques that can be useful for multi-person pose estimation https://github.com/open-mmlab/mmpose

