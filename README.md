# Accuracy Benchmark for Jacinto 7


## Introduction
This repository provides scripts for Accuracy benchmarking for TIDL. Open source runtime front ends of TIDL such as TVM-DLR and TFLRT (TFLite Runtime) are supported.


## Requirements 
- We have tested this on Ubuntu 18.04 PC with Anaconda Python 3.6. This is the recommended environment.
- This package does accuracy benchmarking for Jacinto 7 family of devices using TIDL and its open source front ends. RTOS SDK for Jacinto 7 is required to run this package. Please visit [Processor SDK for Jacinto 7 TDA4x](https://www.ti.com/tool/PROCESSOR-SDK-J721E) and from there navigate to [RTOS SDK for Jacinto 7 TDA4x](https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E) to download and install the RTOS SDK on your Ubuntu desktop machine.
- After installation, follow the instructions in the RTOS package to download the dependencies required for it to build the source. Some of those dependencies are required in this benchmark code as well - especially graphviz and gcc-arm.
- Models used for this benchmark are provided in the repository [Jacinto-AI-ModelZoo](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/jacinto-ai-modelzoo/browse). Please clone that repository into the same parent folder where this repository is cloned.


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
This benchmark uses several datasets. They should be available at the following locations - create the symbolic links as necessary:
- ImageNet dataset should be available in the path *dependencies/datasets/imagenet*
- Cityscapes dataset should be available in the path *dependencies/datasets/cityscpaes*
- COCO dataset should be available in the path *dependencies/datasets/coco*


## Usage

#### Accuracy Benchmarking
- Look at the example script [run_benchmarks.h](../run_benchmarks.h) to understand how to run these benchmarks. It sets up some environment variables then runs the scripts provided in [scripts](../scripts). 
- Parameters for the accuracy benchmark are specified in [config_settings.py](../config_settings.py). Change these parameters appropriately - for example to get int8 accuracy, int16 accuracy etc.
- *target_device = 'pc'* is to run accuracy benchmark simulation on PC. *target_device = 'j7'* is to run the accuracy benchmark on the target Jacinto 7 device.
- Parallel execution is supported on PC and can be enabled by providing a list of integers to the argument *parallel_devices* as seen in config_settings.py. 

#### Low Level Usage (Optional)
TODO

###### Intermediate Layer Outputs (Optional)
TODO

###### Examples
TODO

###### Tutorials
TODO


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
