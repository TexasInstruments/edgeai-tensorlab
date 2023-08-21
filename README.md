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

We have categories of models and scripts in this repository:


<hr>

### Category 1: Our 'edgeailite' models

[See documentation of our edgeailite extensions to torchvision](./references/edgeailite/README.md)

torchvision originally had only classification models. So we went ahead added embedded friendly models and training scripts for tasks such as Semantic Segmentation, Depth Estimation, Multi-Task Estimation etc. 

Scripts are provided for training low complexity DNN models for a variety of tasks:

- Image Classification
- Semantic Segmentation
- Motion Segmentation
- Depth Estimation
- Multi-Task Estimation
- And more...

These models, transforms, and training scrpts are in [./references/edgeailite](./references/edgeailite). The models in our model zoo that were trained using these scripts carries a keyword "edgeailite".

The shell scripts **run_edgeailite_....sh** can be used to train, evaluate or export these "edgeailite" models. 


<hr>

### Category 2: "lite" variants of original torchvision models
These are models that created replacing unsupported layers of torchvision models with supported ones - we call them "lite" models. These "lite" models (1) provide more variety to our Model Zoo making it richer (2) extensible as torchvision adds more models in the future (3) stay close to the official version of these models.

The shell scripts **run_torchvision_....sh** can be used to train, evaluate or export these "lite" models.


<hr>

## Category 3: Original torchvision models and documentation
This repository is built on top of **0.15.x** release of torchvision. We do not modify torchvision python package itself, so the user can use off-the-shelf as is done in our [setup](./setup.sh) script. See the original torchvision documentation:
- [online html version][https://pytorch.org/vision/0.15/]
- [the local git version](./README.rst)


<hr>

## Guidelines for Model training & quantization
Quantization (especially 8-bit Quantization) is important to get best throughput for inference. Quantization can be done using either **Post Training Quantization (PTQ)** or **Quantization Aware Training (QAT)**. Guidelines for Model training and tools for QAT are given the **[documentation on Quantization](./references/edgeailite/edgeai_xvision/xnn/quantization/README.md)**.

- Post Training Quantization (PTQ): TIDL natively supports PTQ - it can take floating point models and can quantize them using advanced calibration methods. In the above link, we have provided guidelines on how to choose models and how to train them for best accuracy with quantization - these guidelines are important to reduce accuracy drop during quantization with **PTQ**. 

- Quantization Aware Training (QAT): In spite of following these guidelines, if there are models that have significant accuracy drop with PTQ, it is possible to improve the accuracy using **QAT**. See the above link for more details.


