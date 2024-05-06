# EdgeAI-MMDetection

### Notice
- If you have not visited the landing page of at https://github.com/TexasInstruments/edgeai, please do so before attempting to use this repository. We skip most of the introduction in this repository.
- This repository is located in Github at: https://github.com/TexasInstruments/edgeai-torchvision

<hr>

This repository is an extension of the popular [mmdetection](https://github.com/open-mmlab/mmdetection) open source repository for object detection training. While mmdetection focuses on a wide variety of models, typically at high complexity, we focus on models that are optimized for speed and accuracy so that they run efficiently on embedded devices. For this purpose, we have added a set of embedded friendly model configurations and scripts - please see the [Usage](./docs/det_usage.md) for more information.

If the accuracy degradation with Post Training Quantization (PTQ) is higher than expected, this repository provides instructions and functionality required to do Quantization Aware Training (QAT).

<hr>


## Release Notes
See notes about recent changes/updates in this repository in [release notes](./docs/det_release_notes.md)


## Installation
These installation instructions were tested using [Miniconda](https://docs.conda.io/en/latest/) Python 3.7 on a Linux Machine with Ubuntu 18.04 OS.

Make sure that your Python version is indeed 3.7 or higher by typing:<br>
```
python --version
```

Please clone and install [**EdgeAI-Torchvision**](https://github.com/TexasInstruments/edgeai-torchvision) as this repository uses several components from there - especially to define low complexity models and to do Quantization Aware Training (QAT) or Calibration.

After that, install this repository by running [./setup.sh](./setup.sh)

After installation, a python package called "mmdet" will be listed if you do *pip list*

In order to use a local folder as a repository, your PYTHONPATH must start with a : or a .: Please add the following to your .bashrc startup file for bash (assuming you are using bash shell). 
```
export PYTHONPATH=:$PYTHONPATH
```
Make sure to close your current terminal or start a new one for the change in .bashrc to take effect or one can do *source ~/.bashrc* after the update. 


## Get Started
Please see [Usage](./docs/det_usage.md) for training and testing with this repository.


## Object Detection Model Zoo
Complexity and Accuracy report of several trained models is available at the [Detection Model Zoo](./docs/det_modelzoo.md) 


## Quantization
This tutorial explains more about quantization and how to do [Quantization Aware Training (QAT)](./docs/det_quantization.md) of detection models.


## ONNX & Prototxt Export
**Export of ONNX model (.onnx) and additional meta information (.prototxt)** is supported. The .prototxt contains meta information specified by **TIDL** for object detectors. 

The export of meta information is now supported for **SSD** and **RetinaNet** detectors.

For more information please see [Usage](./docs/det_usage.md)


## Advanced documentation
Kindly take time to read through the documentation of the original [mmdetection](README_mmdet.md) before attempting to use extensions added this repository.

The setup script [setup.sh](setup.sh) in this repository has the commonly used settings. If your CUDA version is different or your Python version is different or if you have some missing packages in your system, this script can fail. In those scenarios, please refer to [installation instructions for original mmdetection](./docs/get_started.md) for detailed installation instructions. 

Also see [documentation of MMDetection](./docs/index.rst) for the basic usage of original mmdetection. 

 
## Acknowledgement

This is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.

We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to train existing detectors and also to develop their own new detectors.


## License

Please see [LICENSE](./LICENSE) file of this repository.


## Citation

This package/toolbox is an extension of mmdetection (https://github.com/open-mmlab/mmdetection). If you use this repository or benchmark in your research or work, please cite the following:

```
@article{EdgeAI-MMDetection,
  title   = {{EdgeAI-MMDetection}: An Extension To Open MMLab Detection Toolbox and Benchmark},
  author  = {Texas Instruments EdgeAI Development Team, edgeai-devkit@list.ti.com},
  journal = {https://github.com/TexasInstruments/edgeai},
  year={2021}
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


## References
[1] MMDetection: Open MMLab Detection Toolbox and Benchmark, https://arxiv.org/abs/1906.07155, Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin