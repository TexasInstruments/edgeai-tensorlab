# EdgeAI-MMDetection

This repository is an extension of the popular [mmdetection](https://github.com/open-mmlab/mmdetection) open source repository for object detection training. While mmdetection focuses on a wide variety of models, typically at high complexity, we focus on models that are optimized for speed and accuracy so that they run efficiently on embedded devices. 

If the accuracy degradation with Post Training Quantization (PTQ) is higher than expected, this repository provides instructions and functionality required to do Quantization Aware Training (QAT).

Kindly take time to read through the documentation of the original [mmdetection](https://github.com/open-mmlab/mmdetection) before attempting to use this repository.

### Notice
- If you have not visited the landing page of at https://github.com/TexasInstruments/edgeai, please do so before attempting to use this repository. We skip most of the introduction in this repository.
- This repository is located in Github at: https://github.com/TexasInstruments/edgeai-torchvision


## Installation
These installation instructions were tested using [Miniconda](https://docs.conda.io/en/latest/) Python 3.7 on a Linux Machine with Ubuntu 18.04 OS.

Make sure that your Python version is indeed 3.7 or higher by typing:<br>
```
python --version
```

Please install this by running [./setup.sh](./setup.sh)


### Advanced Installation help
setup.sh has the commonly used settings. If your CUDA version is different or your Python version si different or if you have some missing packages in your system, this script can fail. In those scenarios, the following explanations can help.

This repository requires [**mmdetection**](https://github.com/open-mmlab/mmdetection) and [**mmcv**](https://github.com/open-mmlab/mmcv) to be installed. 

Please refer to [installation instructions for mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md) for installation instructions. If you get any issues with the master branch of mmdetection, please try after checking out the latest release tag. After installation, a python package called "mmdet" will be listed if you do *pip list*

To install mmcv, browse to the github page of [mmcv](https://github.com/open-mmlab/mmcv), and see the section that says "**Install with pip**". Install the full version of mmcv using the instruction given there. Please check your CUDA version and PyTorch version and select the appropriate installation instructions.

After installing **mmdetection** and **mmcv**, please clone and install [**EdgeAI-Torchvision**](https://github.com/TexasInstruments/edgeai-torchvision) as this repository uses several components from there - especially to define low complexity models and to do Quantization Aware Training (QAT) or Calibration.

If you still face issues, see our minimal installation script [setup.sh](./setup.sh) and modify for your system.


## Get Started

Please see [Getting Started with MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md) for the basic usage of mmdetection. Note: Some of these may not apply to this repository.

Please see [Usage](./docs/det_usage.md) for training and testing with this repository.


## Object Detection Model Zoo

Complexity and Accuracy report of several trained models is available at the [Detection Model Zoo](./docs/det_modelzoo.md) 


## Quantization

This tutorial explains more about quantization and how to do [Quantization Aware Training (QAT)](./docs/det_quantization.md) of detection models.


## ONNX & Prototxt Export
**Export of ONNX model (.onnx) and additional meta information (.prototxt)** is supported. The .prototxt contains meta information specified in **[TIDL](https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/latest/exports/docs/psdk_rtos_auto/docs/user_guide/sdk_components.html#ti-deep-learning-library-tidl)** for object detectors. 

The export of meta information is now supported for **SSD** and **RetinaNet** detectors.

For more information please see [Usage](./docs/det_usage.md)


## Acknowledgement

This is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.

We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to train existing detectors and also to develop their own new detectors.


## Citation

This package/toolbox is an extension of mmdetection (https://github.com/open-mmlab/mmdetection). If you use this repository or benchmark in your research or work, please cite the following:

```
@article{PyTorch-Jacinto-AI-Detection,
  title   = {{PyTorch-Jacinto-AI-Detection}: An Extension To Open MMLab Detection Toolbox and Benchmark},
  author  = {Jacinto AI Team, jacinto-ai-devkit@list.ti.com},
  journal = {https://github.com/TexasInstruments/jacinto-ai-devkit},
  year={2020}
}
```
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```


## License

Please see [LICENSE](./LICENSE) file.


## Contact
This extension of MMDetection is part of Jacinto-AI-DevKit and is maintained by the Jacinto AI team (jacinto-ai-devkit@list.ti.com). For more details, please visit: https://github.com/TexasInstruments/jacinto-ai-devkit


## References
[1] MMDetection: Open MMLab Detection Toolbox and Benchmark, https://arxiv.org/abs/1906.07155, Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin