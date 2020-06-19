# Jacinto-AI-Detection


This repository is an extension of the popular [mmdetection](https://github.com/open-mmlab/mmdetection) open source repository for object detection training. While mmdetection focuses on a wide variety of models, typically at high complexity, we focus on models that are optimized for speed and accuracy so that they run efficiently on embedded devices. 

When we say MMDetection or mmdetection, we refer to the original repository. However, when we say Jacinto-AI-Detection or "this repository", we refer to this extension of mmdetection with speed/accuracy optimized models.

Kindly take time to read through the original documentation of the original [mmdetection](https://github.com/open-mmlab/mmdetection) before attempting to use this repository. This repository requires mmdetection to be installed.


## License

This repository is released under the following [LICENSE](./LICENSE).


## Installation

Please refer to [mmdetection install.md](https://github.com/open-mmlab/mmdetection/docs/install.md) for installation and dataset preparation. 

We used the the version **v2.1.0** of mmdetection to test our changes. If you get any issues with the master branch of mmdetection, try checking out that tag.

After installing mmdetection, please install [PyTorch-Jacinto-AI-DevKit](https://bitbucket.itg.ti.com/projects/JACINTO-AI/repos/pytorch-jacinto-ai-devkit/browse/) as our repository uses several components from there - especially to define low complexity models and to Quantization Aware Training (QAT).


## Get Started

Please see [getting_started.md](https://github.com/open-mmlab/mmdetection/docs/getting_started.md) for the basic usage of MMDetection. However, some of these may not apply to these repository.

Please see [usage/instructions](https://github.com/open-mmlab/mmdetection/docs/jacinto_ai_detection_usage.md) for training and testing with this repository.


## Benchmark and Model Zoo

Several trained models with accuracy report is available at [Jacinto-AI-Detection Model Zoo](docs/jacinto_ai/jacinto_ai_detection_model_zoo.md) 


## Quantization

Tutorial on how to do [Quantization Aware Training in Jacinto-AI-Detection](docs/jacinto_ai/jacinto_ai_quantization_aware_training.md) in Jacinto-AI-MMDetection. 


## Acknowledgement

This is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.


## Citation

This package/toolbox is an extension of mmdetection (https://github.com/open-mmlab/mmdetection). If you use this package/toolbox or benchmark in your research, please cite that project as well.

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
This extension of MMDetection is part of Jacinto-AI-DevKit and is maintained by the Jacinto AI team: jacinto-ai-devkit@list.ti.com
