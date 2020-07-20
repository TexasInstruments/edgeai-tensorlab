# Jacinto-AI-Detection


This repository is an extension of the popular [mmdetection](https://github.com/open-mmlab/mmdetection) open source repository for object detection training. While mmdetection focuses on a wide variety of models, typically at high complexity, we focus on models that are optimized for speed and accuracy so that they run efficiently on embedded devices. 

Kindly take time to read through the documentation of the original [mmdetection](https://github.com/open-mmlab/mmdetection) before attempting to use this repository.


## License

Please see [LICENSE](./LICENSE) and [LICENSE.SPDX](./LICENSE.SPDX)


## Installation

This repository requires mmdet Python package from mmdetection to be installed. Please refer to [installation instructions for mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md) for installation and also for dataset preparation. If you get any issues with the master branch of mmdetection, please try after checking out the latest release tag. 

Note: mmdetection also requreis mmcv and several other dependencies to be installed as described in the above URL.

After installing mmdetection, please install [PyTorch-Jacinto-AI-DevKit](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/pytorch-jacinto-ai-devkit/browse/) as this repository uses several components from there - especially to define low complexity models and to do Quantization Aware Training (QAT) and Calibration.


## Get Started

Please see [Getting Started with MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md) for the basic usage of mmdetection. Note: Some of these may not apply to this repository.

Please see [Usage](./docs/det_usage.md) for training and testing with this repository.


## Benchmark and Model Zoo

Several trained models with accuracy report is available at [Jacinto-AI-Detection Model Zoo](./docs/det_modelzoo.md) 


## Quantization

This tutorial explaines how to do [Quantization Aware Training](./docs/det_quantization.md) of detection models. We also provide sample QAT models.


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


## Contact
This extension of MMDetection is part of Jacinto-AI-DevKit and is maintained by the Jacinto AI team (jacinto-ai-devkit@list.ti.com). For more details, please visit: https://github.com/TexasInstruments/jacinto-ai-devkit


## References
[1] MMDetection: Open MMLab Detection Toolbox and Benchmark, https://arxiv.org/abs/1906.07155, Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin