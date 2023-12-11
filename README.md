# EdgeAI-TorchVision 

### Notice 1
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai 
- https://github.com/TexasInstruments/edgeai


### Notice 2
- The scripts in this repository use **Model Optimization Tools** in **[edgeai-modeloptimization](https://github.com/TexasInstruments/edgeai-modeloptimization/tree/r9.1)**. It is installed during pip install using [requirements.txt](requirements.txt) file - so there is no need to clone it, unless you want to modify it.

<hr>

Develop Embedded Friendly Deep Neural Network Models in **PyTorch** ![PyTorch](./docs/source/_static/img/pytorch-logo-flame.png)

## Introduction
This is an extension of the popular GitHub repository [pytorch/vision](https://github.com/pytorch/vision) that implements torchvision - PyTorch based datasets, model architectures, and common image transformations for computer vision.

Apart from the features in underlying torchvision, we support the following features
- Models and model training scripts: We also have several example embedded friendly models and training scripts for various Vision tasks. These models and scripts are called when you run the .sh scripts containing the name edgeailite - for example [run_edgeailite_classification_v1.sh](./run_edgeailite_classification_v1.sh)
- Model optimization tools: We used added edgeai-modeloptimization to add support for Surgery, Purning and Quantization in torchvision's training scripts [./references/classification/train.py](./references/classification/train.py) and [./references/segmentation/train.py](./references/segmentation/train.py). These .sh scripts that utilize these have the keyword torchvision - for example [run_torchvision_classification_v2.sh](./run_torchvision_classification_v2.sh), [run_torchvision_classification_v2_qat.sh](./run_torchvision_classification_v2_qat.sh)

It is important to note that we do not modify the [torchvision](./torchvision) python package itself - so off-the-shelf, pip installed torchvision python package can be used with the scripts in this repository. See the setup documentation and the setup file for details. However, we do modify the training scripts in [references](./references) that uses the torchvision package. When we need to modify a model, we do that by modifying the model object/instance in the training script using our model surgery tool.

<hr>

## Setup 
[Setup Instructions](./references/edgeailite/docs/setup.md)

<hr>


## Models and Scripts

We have different categories of models and scripts in this repository:

### Category 1: "lite" variants of original torchvision models
These are models that created replacing unsupported layers of torchvision models with supported ones - we call them "lite" models. This replacing is done by using our Model Optimization Tools, just after the model is created, in the training script.

It is important to note that we do not modify the [torchvision](./torchvision) python package itself - so off-the-shelf, pip installed torchvision python package can be used with the scripts in this repository. However, we do modify the training scripts in [references](./references) that uses the torchvision package. When we need to modify a model, we do that by modifying the model object/instance in the training script using our Model Optimization Tools.

To see example usages of Model Optimization Tools, please refer to [references/classification/train.py](./references/classification/train.py) and [references/segmentation/train.py](./references/segmentation/train.py)

The shell scripts **run_torchvision_....sh** can be used to train, evaluate or export these "lite" models. Accuracy results of training these lite models are in the documentation of our Model Optimization Tools.


### Category 2: Our custom 'edgeailite' models

torchvision had only classification models before 2019. So we went ahead added embedded friendly models and training scripts for tasks such as Semantic Segmentation, Depth Estimation, Multi-Task Estimation etc. These are low complexity models that are suitable for embedded SoCs. 

[See documentation of our edgeailite extensions to torchvision](./references/edgeailite/README.md)

Scripts are provided for training these models for a variety of tasks:

- Image Classification
- Semantic Segmentation
- Motion Segmentation
- Depth Estimation
- Multi-Task Estimation
- And more...

These models, transforms, and training scrpts are in [./references/edgeailite](./references/edgeailite). The models in our model zoo that were trained using these scripts carries a keyword "edgeailite".

The shell scripts **run_edgeailite_....sh** can be used to train, evaluate or export these "edgeailite" models. 


### Original torchvision models and documentation
This repository is built on top of **0.15.x** release of torchvision. We do not modify torchvision python package itself, so the user can use the original models as well. See the original torchvision documentation:
- [online html version](https://pytorch.org/vision/0.15/)
- [the local git version](./README.rst)

<hr>
