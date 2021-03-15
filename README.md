# Accuracy Benchmark for Jacinto 7


#### Notice
This repository is part of Jacinto-AI-DevKit, which is a collection of repositories providing Training & Quantization scripts, Model Zoo and Accuracy Benchmarks. If you have not visited the landing page of [**Jacinto-AI-Devkit**](https://github.com/TexasInstruments/jacinto-ai-devkit) please do so before attempting to use this repository.


## Introduction
This repository provides a collection of example Deep Learning Models for various image recognition tasks such as classification, segmentation and detection. Scripts are provided for Model Import/Calibration, Inference and Accuracy benchmarking. 

The models in this repository are expected to run on the Jacinto7 family of SoCs. RTOS SDK for Jacinto 7 is required to run these. Please visit **[Processor SDK for Jacinto 7 TDA4x](https://www.ti.com/tool/PROCESSOR-SDK-J721E)** and from there navigate to **[RTOS SDK for Jacinto 7 TDA4x](https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E)** to download and untar/extract the RTOS SDK on your Ubuntu desktop machine. RTOS SDK for Jacinto 7 TDA4x provides TI Deep Learning Library (TIDL) which is an optimized library that can run Neural Networks. TIDL can directly accept the models using its config files. TIDL can also accept these models through some of the popular open source runtimes such as TVM+DLR, TFLite and ONNXRuntime. For more information how to obtain and use these runtimes that offload to TIDL, please go through the TIDL documentation in the RTOS SDK.

This repository is generic and can be used with a variety of runtimes and models supported by TIDL. We also provide several Deep Learning Models ready to be used for this benchmark in the repository **[Jacinto-AI-ModelZoo](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-modelzoo/browse)**. Please clone that repository. That repository uses git-lfs, so please install git-lfs before cloning. After cloning, **jacinto-ai-benchmark** and **jacinto-ai-modelzoo** must be in the same parent folder. 

**Important Note**: This repository is being made available for experimentation, analysis and research - this is not meant for final production.


## Requirements 
- We have tested this on Ubuntu 18.04 PC with Anaconda Python 3.6. This is the recommended environment. Create a Python 3.6 environment if you don't have it and activate it.
- Models used for this benchmark are provided in the repository Jacinto-AI-ModelZoo as explained earlier. Please clone that repository.
- RTOS SDK for Jacinto 7 is required to run this package. Please visit the linsk given above to download and untar/extract the RTOS SDK on your Ubuntu desktop machine.
- After extracting, follow the instructions in the RTOS package to download the dependencies required for it. Especially the following 3 steps are required:
- (1) Install PSDK-RTOS dependencies - especially graphviz and gcc-arm: Change directory to **psdk_rtos/scripts** inside the extracted SDK and run **setup_psdk_rtos.sh**
- (2) In the extracted SDK, change directory to tidl folder (it has the form tidl_j7_xx_xx_xx_xx). Inside the tidl folder, change directory to **ti_dl/test/tvm-dlr** and run **prepare_model_compliation_env.sh** to install TVM Deep Learning compiler, DLR Deep Learning Runtime and their dependencies. In our SDK, we have support to use TVM+DLR to offload part of the graph into the underlying TIDL backend running on the C7x+MMA DSP, while keeping the unsupported layers running on the main ARM processor. 
- (3) Inside the tidl folder, change directory to **ti_dl/test/tflrt** and run **prepare_model_compliation_env.sh** to install TFLite Runtime and its dependencies. In our SDK, we have support to use TFLite Runtime to offload part of the graph into the underlying TIDL backend running on the C7x+MMA DSP, while keeping the unsupported layers running on the main ARM processor.


## Installation Instructions
After cloning this repository, install it as a Python package by running:
```
./setup.sh
```

Open the shell scripts that starts the actual benchmarking [run_benchmarks.sh](./run_benchmarks.sh), [tutorials](./tutorials) and see the environment variables **PSDK_BASE_PATH** and **TIDL_BASE_PATH** being defined. Change these paths appropriately to reflect what is in your PC.

Once installed, the **jacinto_ai_benchmark** will be a available as a package in your Python environment. It can be imported just like any other Python package in a Python script:<br>
```
import jacinto_ai_benchmark
```
or
```
from jacinto_ai_benchmark import *
```

## Datasets
This benchmark uses several datasets. They should be available at the following locations - if you have the datasets stored somewhere else, create the symbolic links as necessary:
- ImageNet dataset validation split should be available in the path *dependencies/datasets/imagenet/val* and a text file describing the list of images and the corresponding ground truth classes must be available in *dependencies/datasets/imagenet/val.txt*
- Cityscapes dataset should be available in the path *dependencies/datasets/cityscpaes* - we use only the validation split - which should be in the folders *dependencies/datasets/cityscpaes/cityscapes/leftImg8bit/val* and *dependencies/datasets/cityscpaes/cityscapes/gtFine/val*
- COCO dataset should be available in the path *dependencies/datasets/coco* especially the folders *dependencies/datasets/coco/val2017* and *dependencies/datasets/coco/annotations*
- ADE20K dataset should be available in the path *dependencies/datasets/ADEChallengeData2016*
- PascalVOC2012 dataset should be available in the path *dependencies/datasets/VOCdevkit/VOC2012*

We have support to support to download several of these datasets automatically - but the download may take several hours even with a good internet connection - that's why it's important to make the datasets available at the above locations to avoid that download - if you have them already.

It you start the download and interrupt it in between, the datasets may be partially downloaded and it can lead to unexpected failures. If the download of a dataset is interrupted in between, delete that dataset folder manually to start over.


## Usage

#### Tutorial
- Samples Jupyter Notebook tutorials are in [tutorials](./tutorials)
- Start the Jupyter Notebook by running [./run_tutorials.sh](./run_tutorials.sh) and then select a notebook in the tutorials folder.
- Notice how we use settings to limit the benchmark only to a couple of networks and only the 'imagenet' dataset by setting the parameter 'dataset_loading'. 
- This kind of limiting helps you to focus on the networks and datasets that you are interested in. Such parameters can be set either in the yaml file being used (see the .sh file) or passed as arguments while creating ConfigSettings() in th code.
- Run that example benchmarking by running the the cells in the Notebook. 

#### Accuracy Benchmarking
- Accuracy benchmark can be done by running [run_benchmarks.sh](./run_benchmarks.sh)
- [run_benchmarks.sh](../run_benchmarks.sh) sets up some environment variables and then runs the benchmark code provided in [scripts/benchmark_accuracy.py](./scripts/benchmark_accuracy.py) using the settings defined in [accuracy_minimal_pc.yaml](accuracy_minimal_pc.yaml) - this is to run a sample benchmark on PC. 
- For full fledged benchmarking, you can use the yaml file [accuracy_full_pc.yaml](./accuracy_full_pc.yaml)
- Change the yaml settings file appropriately to run on J7 EVM. [accuracy_import_for_j7.yaml](./accuracy_import_for_j7.yaml) can be used to run the import/calibration of the models on PC, but targeted for the J7 platform. This will create the imported artifacts corresponding to the models in the folder specified as work_dir in the benchmark script. 
- Finally [accuracy_infer_on_j7.yaml](./accuracy_infer_on_j7.yaml) can be used when running the benchmark on the J7 EVM. This step will need the folder containing imported artifacts - so copy them over to the EVM or mount that folder via NFS.
- By default, the accuracy benchmark script uses our pre-defined models, defined in [jacinto_ai_benchmark/configs](./jacinto_ai_benchmark/configs).
- If you would like to do accuracy benchmark for your own custom model, then please look at the example given in [scripts/custom_example.py](./scripts/custom_example.py).


## References
- **ImageNet ILSVRC Dataset**: Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision, 2015. http://www.image-net.org/ <br>

- **COCO Dataset**: Microsoft COCO: Common Objects in Context, Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár, https://arxiv.org/abs/1405.0312, https://cocodataset.org/ <br>

- **PascalVOC Dataset**: The PASCAL Visual Object Classes (VOC) Challenge, Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A., International Journal of Computer Vision, 88(2), 303-338, 2010, http://host.robots.ox.ac.uk/pascal/VOC/ <br>

- **ADE20K Scene Parsing Dataset** Scene Parsing through ADE20K Dataset. Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso and Antonio Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. Semantic Understanding of Scenes through ADE20K Dataset. Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso and Antonio Torralba. International Journal on Computer Vision (IJCV). https://groups.csail.mit.edu/vision/datasets/ADE20K/, http://sceneparsing.csail.mit.edu/ <br>

- **Cityscapes Dataset**: M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. https://www.cityscapes-dataset.com/ <br>

- **MMDetection: Open MMLab Detection Toolbox and Benchmark**, Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua. arXiv:1906.07155, 2019 <br>

- **SSD: Single Shot MultiBox Detector**, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016. <br>

- **MLPerf Inference Benchmark**, Vijay Janapa Reddi and Christine Cheng and David Kanter and Peter Mattson and Guenther Schmuelling and Carole-Jean Wu and Brian Anderson and Maximilien Breughe and Mark Charlebois and William Chou and Ramesh Chukka and Cody Coleman and Sam Davis and Pan Deng and Greg Diamos and Jared Duke and Dave Fick and J. Scott Gardner and Itay Hubara and Sachin Idgunji and Thomas B. Jablin and Jeff Jiao and Tom St. John and Pankaj Kanwar and David Lee and Jeffery Liao and Anton Lokhmotov and Francisco Massa and Peng Meng and Paulius Micikevicius and Colin Osborne and Gennady Pekhimenko and Arun Tejusve Raghunath Rajan and Dilip Sequeira and Ashish Sirasao and Fei Sun and Hanlin Tang and Michael Thomson and Frank Wei and Ephrem Wu and Lingjie Xu and Koichi Yamada and Bing Yu and George Yuan and Aaron Zhong and Peizhao Zhang and Yuchen Zhou, arXiv:1911.02549, 2019 <br>

- **Pytorch/Torchvision**: Torchvision the machine-vision package of torch, Sébastien Marcel, Yann  Rodriguez, MM '10: Proceedings of the 18th ACM international conference on Multimedia October 2010 Pages 14851488 https://doi.org/10.1145/1873951.1874254, https://pytorch.org/vision/stable/index.html

- **TensorFlow Model Garden**: The TensorFlow Model Garden is a repository with a number of different implementations of state-of-the-art (SOTA) models and modeling solutions for TensorFlow users. https://github.com/tensorflow/models <br>

- **TensorFlow Object Detection API**: Speed/accuracy trade-offs for modern convolutional object detectors. Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z, Song Y, Guadarrama S, Murphy K, CVPR 2017, https://github.com/tensorflow/models/tree/master/research/object_detection <br>

- **Tensorflow DeepLab**: DeepLab: Deep Labelling for Semantic Image Segmentation https://github.com/tensorflow/models/tree/master/research/deeplab

- **TensorFlow Official Model Garden**, Chen Chen and Xianzhi Du and Le Hou and Jaeyoun Kim and Pengchong, Jin and Jing Li and Yeqing Li and Abdullah Rashwan and Hongkun Yu, 2020, https://github.com/tensorflow/models/tree/master/official <br>

- **GluonCV**: GluonCV and GluonNLP: Deep Learning in Computer Vision and Natural Language Processing
Jian Guo, He He, Tong He, Leonard Lausen, Mu Li, Haibin Lin, Xingjian Shi, Chenguang Wang, Junyuan Xie, Sheng Zha, Aston Zhang, Hang Zhang, Zhi Zhang, Zhongyue Zhang, Shuai Zheng, Yi Zhu, https://arxiv.org/abs/1907.04433


## License
Please see the [LICENSE](./LICENSE) file for the license details.
