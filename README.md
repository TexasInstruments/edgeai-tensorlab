# Accuracy Benchmark for Jacinto 7


## Introduction
This repository provides scripts for Accuracy benchmarking for TIDL. Open source runtime front ends of TIDL such as TVM-DLR and TFLRT (TFLite Runtime) are supported.


## Requirements 
- Models used for this benchmark are provided in the repository [Jacinto-AI-ModelZoo](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-modelzoo/browse). Please clone that repository into the same parent folder where this repository is cloned.
- We have tested this on Ubuntu 18.04 PC with Anaconda Python 3.6. This is the recommended environment.
- This package does accuracy benchmarking for Jacinto 7 family of devices using TIDL and its open source front ends. RTOS SDK for Jacinto 7 is required to run this package. Please visit [Processor SDK for Jacinto 7 TDA4x](https://www.ti.com/tool/PROCESSOR-SDK-J721E) and from there navigate to [RTOS SDK for Jacinto 7 TDA4x](https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E) to download and untar/extract the RTOS SDK on your Ubuntu desktop machine.
- After extracting, follow the instructions in the RTOS package to download the dependencies required for it. Especially the following 3 steps:
- (1) Install PSDK-RTOS dependencies - some of those dependencies are required in this benchmark code as well - especially graphviz and gcc-arm: Change directory to psdk_rtos/scripts inside the extracted SDK and run setup_psdk_rtos.sh
- (2) In the extracted SDK, change directory to tidl folder (it has the form tidl_j7_xx_xx_xx_xx). Inside the tidl folder, change directory to ti_dl/test/tvm-dlr and run prepare_model_compliation_env.sh to install TVM Deep Learning compiler, DLR Deep Learning Runtime and their dependencies. In our SDK, we have support to use TVM+DLR to offload part of the graph into the underlying TIDL backend running on the C7x+MMA DSP, while keeping the unsupported layers running on the main ARM processor. 
- (3) Inside the tidl folder, change directory to ti_dl/test/tflrt and run prepare_model_compliation_env.sh to install TFLite Runtime and its dependencies. In our SDK, we have support to use TFLite Runtime to offload part of the graph into the underlying TIDL backend running on the C7x+MMA DSP, while keeping the unsupported layers running on the main ARM processor. 


## Installation Instructions
This package can be installed by:
```
./setup.sh
```

Open the file run_benchmarks.sh and see the shell variables PSDK_BASE_PATH and TIDL_BASE_PATH being defined. Change these paths appropriately to reflect what is in your PC, if needed.

Once installed, the **jacinto-ai-benchmark** will be a available as a package in your Python environment. It can be imported just like any other Python package in a Python script:<br>
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

We have support to support to download several of these datasets automatically - but the download will take several hours even with a good internet connection - that's why it's important to make the datasets available at the above locations to avoid that download - if you have them already.

It you start the download and interrupt it in between, the datasets may be partially downloaded and it can lead to unexpected failures. If the download of a dataset is interrupted in between, delete that dataset folder manually to start over.

Important Note: If you wish to do benchmarking using only one or some of the datasets above, that dataset(s) can be specified in the yaml file in the parameter dataset_loading. See the help for dataset_loading in the yaml file for more information. 

## Usage

#### Tutorial
- Samples Jupyter Notebook tutorials are in [tutorials](./tutorials)
- Start the Jupyter Notebook by running [./run_tutorials.sh](./run_tutorials.sh) and then select a notebook to open the tutorial.
- Notice how we use settings to limit the benchmark only to a couple of networks and only the 'imagenet' dataset. This kind of limiting helps you to focus on the networks and datasets that you are interested in.
- Run that example benchmarking by running the the cells. 

#### Accuracy Benchmarking
- Accuracy benchmark can be done by running [run_benchmarks.sh](./run_benchmarks.sh)
- [run_benchmarks.sh](../run_benchmarks.sh) sets up some environment variables then runs the benchmark code provided in [scripts/benchmark_accuracy.py](./scripts/benchmark_accuracy.py) using the settings defined in [accuracy_minimal_pc.yaml](accuracy_minimal_pc.yaml) - this is to run a sample benchmark on PC. 
- Change the yaml settings file appropriately to run on J7 EVM. [accuracy_import_for_j7.yaml](./accuracy_import_for_j7.yaml) can be used to run the import/calibration of the models on PC, but targeted for the J7 platform. This will create the imported/calibrated artifacts corresponding to the models. Finally [accuracy_infer_on_j7.yaml](./accuracy_infer_on_j7.yaml) can be used when running the benchmark on the J7 EVM (this step will use the imported/calibrated artifacts).
- By default, the accuracy benchmark script uses our pre-defined models, defined in [jacinto_ai_benchmark/configs](./jacinto_ai_benchmark/configs).
- If you would like to do accuracy benchmark for your own custom model, then please look at the example given in [scripts/custom_example.py](./scripts/custom_example.py).


## References
- **MMDetection: Open MMLab Detection Toolbox and Benchmark**, Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua. arXiv:1906.07155, 2019 <br>

- **SSD: Single Shot MultiBox Detector**, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016. <br>

- **MLPerf Inference Benchmark**, Vijay Janapa Reddi and Christine Cheng and David Kanter and Peter Mattson and Guenther Schmuelling and Carole-Jean Wu and Brian Anderson and Maximilien Breughe and Mark Charlebois and William Chou and Ramesh Chukka and Cody Coleman and Sam Davis and Pan Deng and Greg Diamos and Jared Duke and Dave Fick and J. Scott Gardner and Itay Hubara and Sachin Idgunji and Thomas B. Jablin and Jeff Jiao and Tom St. John and Pankaj Kanwar and David Lee and Jeffery Liao and Anton Lokhmotov and Francisco Massa and Peng Meng and Paulius Micikevicius and Colin Osborne and Gennady Pekhimenko and Arun Tejusve Raghunath Rajan and Dilip Sequeira and Ashish Sirasao and Fei Sun and Hanlin Tang and Michael Thomson and Frank Wei and Ephrem Wu and Lingjie Xu and Koichi Yamada and Bing Yu and George Yuan and Aaron Zhong and Peizhao Zhang and Yuchen Zhou, arXiv:1911.02549, 2019 <br>

- **TensorFlow Official Model Garden**, Chen Chen and Xianzhi Du and Le Hou and Jaeyoun Kim and Pengchong, Jin and Jing Li and Yeqing Li and Abdullah Rashwan and Hongkun Yu, 2020, https://github.com/tensorflow/models/tree/master/official <br>

- **TensorFlow Object Detection API**: Speed/accuracy trade-offs for modern convolutional object detectors. Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z, Song Y, Guadarrama S, Murphy K, CVPR 2017, https://github.com/tensorflow/models/tree/master/research/object_detection <br>


## Notice
- Jacinto-AI-DevKit is a collection of repositories providing Training scripts, Model Zoo and Accuracy Benchmarks. If you have not visited the landing page of [**Jacinto-AI-Devkit**](https://github.com/TexasInstruments/jacinto-ai-devkit) please do so before attempting to use this repository.


## License
Please see the [LICENSE](./LICENSE) file for the license details.
