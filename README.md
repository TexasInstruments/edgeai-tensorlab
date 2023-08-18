# EdgeAI-TorchVision 

### Notice 1
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai 
- https://github.com/TexasInstruments/edgeai


### Notice 2
- Model Optimization Tools including Quantization Aware Training (QAT) tools have been moved to [edgeai-modeltoolkit](https://github.com/TexasInstruments/edgeai-modeltoolkit). Please visit the new location.


<hr>

Develop Embedded Friendly Deep Neural Network Models in **PyTorch** ![PyTorch](./docs/source/_static/img/pytorch-logo-flame.png)

This is an extension of the popular github repository [pytorch/vision](https://github.com/pytorch/vision) that implements torchvision - PyTorch based datasets, model architectures, and common image transformations for computer vision.

The scripts in this repository requires torchvision to be installed using this repository - the standard torchvision will not support all the features in this repository. Please install our torchvision extension using the instructions below.

<hr>

## Setup 
[Setup Instructions](./references/edgeailite/docs/setup.md)


<hr>


## Categories of Models and Scripts

We have two categories of models and scripts in this repository:


<hr>

### Category 1: Original torchvision models and their "lite" variants

[See the original TorchVision documentation](README.rst)
![](https://static.pepy.tech/badge/torchvision) ![](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchvision%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)

While some models in this category work on our platform, several of them have unsupported layers and will either not work or will be slow. To make sure that unsupported layers are replaced with appropriate layers we use tools provided in [edgeai-modeltoolkit](https://github.com/TexasInstruments/edgeai-modeltoolkit)

These "lite" models (1) provide more variety to our Model Zoo making it richer (2) extensible as torchvision adds more models in the future (3) stay close to the official version of these models.

**Note**: these models are still in developmet and some of them may not be fully supported in TIDL yet.

The shell scripts **run_torchvision_....sh** can be used to train, evaluate or export these "lite" models.

<hr>

### Category 2: Our 'edgeailite' models

[See documentation of our edgeailite extensions to torchvision](./references/edgeailite/README.md)

torchvision originally had only classification models. So we went ahead added embedded friendly models and training scripts for tasks such as Semantic Segmentation, Depth Estimation, Multi-Task Estimation etc. 

Scripts are provided for training low complexity DNN models for a variety of tasks:

- Image Classification
- Semantic Segmentation
- Motion Segmentation
- Depth Estimation
- Multi-Task Estimation
- And more...

These models, transforms, and training scrpts are in [./references/edgeailite](./references/edgeailite). The models in our model zoo that were trained using these scripts carries a keyword "edgeailite". See the list of supported "edgeailite" models here: [./torchvision/edgeailite/xvision/models](./references/edgeailite/xvision/models)

The shell scripts **run_edgeailite_....sh** can be used to train, evaluate or export these "edgeailite" models. 

<hr>
