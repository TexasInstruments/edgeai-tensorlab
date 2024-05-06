# EdgeAI-TorchVision 

### Notice
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai 
- https://github.com/TexasInstruments/edgeai


<hr>

Develop Embedded Friendly Deep Neural Network Models in **PyTorch** ![PyTorch](./docs/source/_static/img/pytorch-logo-flame.png)

This is an extension of the popular github repository [pytorch/vision](https://github.com/pytorch/vision) that implements torchvision - PyTorch based datasets, model architectures, and common image transformations for computer vision.

The scripts in this repository requires torchvision to be installed using this repository - the standard torchvision will not support all the features in this repository. Please install our torchvision extension using the instructions below.


<hr>

## Requirements
These installation instructions were tested using [pyenv](https://github.com/pyenv/pyenv)  Python 3.6 on a Linux Machine with Ubuntu 18.04 OS. See [pyenv-installer](https://github.com/pyenv/pyenv-installer) for detailed pyenv installation instructions.

Here are the steps in brief:
```
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

echo '# pyenv settings ' >> ${HOME}/.bashrc
echo 'command -v pyenv >/dev/null || export PATH=":${HOME}/.pyenv/bin:$PATH"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME}/.bashrc
echo '' >> ${HOME}/.bashrc

exec ${SHELL}
```

Install Python 3.6 in pyenv and create an environment
```
pyenv install 3.6
pyenv virtualenv 3.6 py36
pyenv rehash
pyenv activate py36
pip install --upgrade pip
pip install --upgrade setuptools
```

Activation of Python environment - this activation step needs to be done everytime one starts a new terminal or shell. (Alternately, this also can be written to the .bashrc, so that this will be the default penv environment).
```
pyenv activate py36
```

Make sure that your Python version is indeed 3.6 or higher by typing:<br>
```
python --version
```

Make sure gcc and g++ versions are greater then or equal to 7 - if not install the required versions. The installed versions can be checked by:<br>
```
gcc --version
g++ --version
```

## Installation Instructions

Clone this repository into your local folder

Execute the following shell script to install the dependencies:<br>
```
./setup.sh
```

This installation script is derived from the instructions described in [pytorch/vision](https://github.com/pytorch/vision) to install/build from source.


<hr>


## Categories of Models and Scripts

We have three categories of models and scripts in this repository:


<hr>


### Category 1: Our 'edgeailite' models

**[See documentation of our edgeailite extensions to torchvision](README_edgeailite.md)**

torchvision originally had only classification models and also did not have an implementation of Quantization Aware Training (QAT). So we went ahead added embedded friendly models and training scripts for tasks such as Semantic Segmentation, Depth Estimation, Multi-Task Estimation etc and also QAT. 

**[Tools and scripts for Quantization Aware Training (QAT)]((https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Quantization.md))** that is best suited for our devices are provided.

Scripts are provided for training low complexity DNN models for a variety of tasks:

- Image Classification
- Semantic Segmentation
- Motion Segmentation
- Depth Estimation
- Multi-Task Estimation
- And more...

These models, transforms and utility functions are located in [./torchvision/edgeailite](./torchvision/edgeailite) and the corresponding training scrpts are in [./references/edgeailite](./references/edgeailite). The models trained using these scripts will carry a keyword "edgeailite". See the list of "edgeailite" models here: [./torchvision/edgeailite/xvision/models](./torchvision/edgeailite/xvision/models)

The shell scripts run_edgeailite_....sh can be used to train, evaluate or export these "edgeailite" models. 


<hr>


### Category 2: "lite" models created using original torchvision models

Recently torchvision has started adding support for several tasks such as Object Detection, Semantic Segmentation etc. Some of these do not run on our platform due to the presence of unsupported layers. But we have a model surgery function that creates embedded friendly versions of these models. The models thus created carry a keyworkd "lite". See the list of "lite" models here: [./torchvision/models/model_lite.py](./torchvision/models/model_lite.py), [./torchvision/models/detection/model_lite.py](./torchvision/models/detection/model_lite.py), [./torchvision/models/segmentation/model_lite.py](./torchvision/models/segmentation/model_lite.py)

These "lite" models (1) provide more variety to our Model Zoo making it richer (2) extensible as torchvision adds more models in the future (3) stay close to the official version of these models.

The shell scripts run_torchvision_....sh can be used to train, evaluate or export these "lite" models.

**Note**: these models are still in developmet and some of them may not be fully supported in TIDL yet.

<hr>


### Category 3: Original torchvision models

**[See the Original TorchVision documentation](README.rst)**
![](https://static.pepy.tech/badge/torchvision) ![](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchvision%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)

While some models in this category work on our platform, several of them have unsupported layers and will either not work or will be slow. We recommend to use models from one of the above two categories for inference on our embedded SoC platforms.

<hr>

## Guidelines for Model training & quantization
Quantization (especially 8-bit Quantization) is important to get best throughput for inference. Quantization can be done using either **Post Training Quantization (PTQ)** or **Quantization Aware Training (QAT)**. Guidelines for Model training and tools for QAT are given the **[documentation on Quantization](./docs/pixel2pixel/Quantization.md)**.

- Post Training Quantization (PTQ): TIDL natively supports PTQ - it can take floating point models and can quantize them using advanced calibration methods. In the above link, we have provided guidelines on how to choose models and how to train them for best accuracy with quantization - these guidelines are important to reduce accuracy drop during quantization with **PTQ**. 

- Quantization Aware Training (QAT): In spite of following these guidelines, if there are models that have significant accuracy drop with PTQ, it is possible to improve the accuracy using **QAT**. See the above link for more details.
