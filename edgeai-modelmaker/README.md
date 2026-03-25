# EdgeAI-ModelMaker

#### Notice
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai 
- https://github.com/TexasInstruments/edgeai

<hr>

#### Release Notes
- See the [release notes document](./docs/release_notes.md)

<hr>

## Introduction

EdgeAI-ModelMaker is an end-to-end model development tool that contains dataset handling, model training and compilation. Currently, it doesn't have an integrated feature to annotate data, but can accept annotated Dataset from a tool such as [Label Studio](https://labelstud.io/)

We have published several repositories for model training, model compilation and modelzoo as explained in our [edgeai gihub page](https://github.com/TexasInstruments/edgeai). This repository is an attempt to stitch several of them together to make [release_notes.md](docs%2Frelease_notes.md)a simple and consistent interface for model development. This does not support all the models that can be trained and compiled using our tools, but only a subset. This is a commandline tool and requires a Linux PC.

#### The following are the key operations supported by this tool:
- Dataset handling: This dataset formats supported by this tool is described in a section below. This can convert dataset formats and can automatically split the given dataset into train and validation sets (if it is not already split).
- Model training: Model training repositories such as [edgeai-torchvision](https://github.com/TexasInstruments/edgeai-torchvision) and [edgeai-mmdetection](https://github.com/TexasInstruments/edgeai-mmdetection) are integrated. Several models with pretrained checkpoints are incorporated for each of these repositories. 
- Model compilation: Model compilation tools [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) and [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) for TI's EdgeAI SoCs have been integrated.

These functionalities that are supported are fully integrated and the user can control it by setting  parameters in the config file. 

#### Task Types
- Image Classification
- Object Detection
- Semantic Segmentation
- Keypoint Detection (Note: Keypoint Detection support is broken as of now, but we plan to bring it back with a more flexible backend training repository)

#### Model Types
For Object Detection, we use YOLOX models. For Image Classification we have support for MobileNetV2 and RegNetX. For Semantic Segmentation we have support DeepLabV3Plus, FPN and UNet models. For Keypoint Detection we use the [YOLO-pose](https://arxiv.org/abs/2204.06806) method.

#### SoCs supported
These are devices with Analytics Accelerators (DSP and Matrix Multiplier Acceletator) along with ARM cores.
- AM62A
- AM68A / TDA4AL, TDA4VE, TDA4VL
- AM69A / TDA4VH, TDA4AH, TDA4VP, TDA4AP
- TDA4VM (AM68PA)
- AM67A / TDA4AEN

These are non-accelerated devices (model inference runs on ARM cores) supported.
- AM62 / AM62X, AM62P

The details of these SoCs are [here](https://github.com/TexasInstruments/edgeai/blob/master/readme_sdk.md) 


## Setup Instructions

### Step 1: OS & Environment 

This repository can be used from native Ubuntu bash terminal directly or from within a docker environment.

#### Step 1, Option 1: With native Ubuntu environment and pyenv (recommended)
We have tested this tool in Ubuntu 22.04 and with Python 3.10 (Note: From 9.0 release onwards edgeai-tidl-tools supports only Python 3.10).

In this option, we describe using this repository with the pyenv environment manager. 

Step 1.1a: Make sure that you are using bash shell. If it is not bash shell, change it to bash. Verify it by typing:
```
echo ${SHELL}
```

Step 1.2a: Install system dependencies
```
sudo apt update
sudo apt install build-essential curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev xz-utils wget curl
sudo apt install -y libffi-dev libjpeg-dev zlib1g-dev graphviz graphviz-dev protobuf-compiler
```

Step 1.3a: Install pyenv using the following commands
```
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

echo '# pyenv settings ' >> ${HOME}/.bashrc
echo 'command -v pyenv >/dev/null || export PATH=":${HOME}/.pyenv/bin:$PATH"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME}/.bashrc
echo '' >> ${HOME}/.bashrc

exec ${SHELL}
```

Further details on pyenv installation are given here https://github.com/pyenv/pyenv and https://github.com/pyenv/pyenv-installer


Step 1.4a: Install Python 3.10 in pyenv and create an environment
```
pyenv install 3.10
pyenv virtualenv 3.10 py310
pyenv rehash
pyenv activate py310
pip3 install --no-input --upgrade pip==24.2 setuptools==73.0.0
```

Step 1.5a: **Activate the Python environment.** This activation step needs to be done everytime one starts a new terminal or shell. (Alternately, this also can be written to the .bashrc, so that this will be the default penv environment).
```
pyenv activate py310
```


#### Step 1, Option 2: With docker environment

Step 1.1b: Install docker if you don't have it already. The following steps are for installation on Ubuntu 18.04
```
./docker/docker_setup.sh
```

Step 1.2b: Build docker image:
```
./docker/docker_build.sh
```

Step 1.3b: Run docker container to bring up the container terminal on docker:
```
./docker/docker_run.sh
```

Source .bashrc to update the PATH
```
source /opt/.bashrc
```

Step 1.4b: During docker run, we map the parent directory of this folder to /opt/code This is to easily share code and data between the host and the docker container. Inside the docker terminal, change directory to where this folder is mapped to:
```
cd /opt/code/edgeai-modelmaker
```


### Step 2: Setup the model training and compilation repositories

This tool depends on several repositories that we have published at https://github.com/TexasInstruments

The following setup script can take care of cloning the required repositories and running their setup scripts.

Install without GPU support:
```
./setup_cpu.sh
```

If you have NVIDIA GPU(s), install with GPU support:
```
./setup_gpu.sh
```
Note: To actually use GPU for training, in the config yaml file that you are using in this repository to run modelmaker, set the num_gpus: 1 under the section training.

If the script runs sucessfully, you should have this directory structure: 
<pre>
parent_directory
    |
    |--edgeai-modelzoo
    |--edgeai-torchvision
    |--edgeai-mmdetection
    |--edgeai-mmpose
    |--edgeai-tensorvision
    |--edgeai-benchmark
    |--edgeai-modelmaker
</pre>

Your python environment will have several model compilation python packages from [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) installed. See it by running:
<pre>
pip list | grep 'onnxruntime\|tflite\|tvm\|dlr\|osrt'
</pre>

Also, PyTorch and its related packages will be installed (This torchvision package is installed from our fork called edgeai-torchvision). See it by running:
<pre>
pip list | grep 'torch\|torchvision'
</pre>


### Step 3: Run the ready-made examples

```
./run_modelmaker.sh <target_device> <config_file>
```

#### Examples: 

Object detection example
```
./run_modelmaker.sh TDA4VM config_detection.yaml
```

Image classification example
```
./run_modelmaker.sh TDA4VM config_classification.yaml
```

Where TDA4VM above is an example of target_device supported. 

#### Target devices supported
The list of target devices supported depends on the tidl-tools installed by [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark). Currently **TDA4VM, AM68A, AM69A, AM62A, AM67A & AM62** are supported.


### Step 4: Prepare your own dataset with your own images and annotations
#### Step 4.1 Using TI Edge AI SUDIO Model Composer
- We recommend [TI Edge AI SUDIO Model Composer](https://dev.ti.com/edgeaistudio/) for annotating the images to create the dataset in supported format. 
- Download an example dataset from there are to understand the exact format. 
- In our example config yaml files also, we have given urls of example datasets that can be downloaded and used as reference.

#### Step 4.2: Dataset format (Optional)
- If you already have a dataset, you can convert to the format that is supported by TI Edge AI SUDIO Model Composer.
- The dataset format is similar to that of the [COCO](https://cocodataset.org/) dataset, but there are some changes as explained below.
- The annotated json file and images must be under a suitable folder with the dataset name. 
- Under the folder with dataset name, the following folders must exist: 
- (1) there must be an "images" folder containing the images
- (2) there must be an annotations folder containing the annotation json file with the name given below.

##### Object Detection dataset format
An object detection dataset should have the following structure. 

<pre>
data/datasets/dataset_name
                             |
                             |--images
                             |     |-- the image files should be here
                             |
                             |--annotations
                                   |--instances.json
</pre>

- Use a suitable dataset name instead of dataset_name
- The default annotation file name for object detection is instances.json
- The format of the annotation file is similar to that of the [COCO dataset 2017 Train/Val annotations](https://cocodataset.org/#download) - a json file containing 'info', 'images', 'categories' and 'annotations'. 'bbox' field in the annotation is required and is used as the ground truth.
- Look at the example dataset [animal_classification](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_classification.zip) to understand further.
- In the config file, provide the name of the dataset (dataset_name in this example) in the field dataset_name and provide the path or URL in the field input_data_path.
- Then the ModelMaker tool can be invoked with the config file.

##### Image Classification dataset format
An image classification dataset should have the following structure. (Use a suitable dataset name instead of dataset_name).

<pre>
data/datasets/dataset_name
                             |
                             |--images
                             |     |-- the image files should be here
                             |
                             |--annotations
                                   |--instances.json
</pre>

- Use a suitable dataset name instead of dataset_name
- The default annotation file name for image classification is instances.json
- The format of the annotation file is similar to that of the COCO dataset - a json file containing 'info', 'images', 'categories' and 'annotations'. However, one difference is that the 'bbox' or 'segmentation' information is not used for classification task and need not be present. The category information in each annotation (called the 'id' field) is needed.
- Look at the example dataset [animal_detection](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_detection.zip) to understand further.
- In the config file, provide the name of the dataset (dataset_name in this example) in the field dataset_name and provide the path or URL in the field input_data_path.
- Then the ModelMaker tool can be invoked with the config file.


##### Semantic Segmentation dataset format
An object detection dataset should have the following structure. 

<pre>
data/datasets/dataset_name
                             |
                             |--images
                             |     |-- the image files should be here
                             |
                             |--annotations
                                   |--instances.json
</pre>

- Use a suitable dataset name instead of dataset_name
- The default annotation file name for object detection is instances.json
- The format of the annotation file is similar to that of the [COCO dataset 2017 Train/Val annotations](https://cocodataset.org/#download) - a json file containing 'info', 'images', 'categories' and 'annotations'. 'segmentation' field in the annotation is required and is used as the ground truth. 
- Look at the example dataset [animal_classification](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_classification.zip) to understand further.
- In the config file, provide the name of the dataset (dataset_name in this example) in the field dataset_name and provide the path or URL in the field input_data_path.
- Then the ModelMaker tool can be invoked with the config file.


#### Notes
If the dataset has already been split into train and validation set already, it is possible to provide those paths separately as a tuple in input_data_path.

After the model compilation, the compiled models will be available in a folder inside [./data/projects](./data/projects)

If you have a dataset in another format, use the script provided to convert it into the COCO jSON format. See the examples given in [run_convert_dataset.sh](./run_convert_dataset.sh) for example conversions.

The config file can be in .yaml or in .json format


## Step 5: Accelerated Training using GPUs (Optional) 

Note: **This section is for advanced users only**. Familiarity with NVIDIA GPU and CUDA driver installation is assumed.

This tool can train models either on CPU or on GPUs. By default, CPU based training is used. 

It is possible to speedup model training significantly using GPUs (with CUDA support) - if you have those GPUs in the PC. The PyTorch version that we install by default is not capable of supporting CUDA GPUs. There are additional steps to be followed to enable GPU support in training. 
- In the file setup_all.sh, we are using setup_cpu.sh for several of the repositories that we are using. These will have to be changed to setup.sh before running setup_all.sh
- Install GPU driver and other tools as described in the sections below.
- In the config file, set a value for num_gpus to a value greater than 0 (should not exceed the number of GPUs in the system) to enable GPU based training.

#### Option 1: When using Native Ubuntu Environment

The user has to install an appropriate NVIDIA GPU driver that supports the GPU being used.

The user also has to install CUDA Toolkit. See the [CUDA download instructions](https://developer.nvidia.com/cuda-downloads). The CUDA version that is installed must match the CUDA version used in the PyTorch installer - see [our edgeai-torchvision setup script](https://github.com/TexasInstruments/edgeai-torchvision/blob/master/setup.sh) to understand the CUDA version used. 

#### Option 2: When using docker environment

Enabling CUDA GPU support inside a docker environment requires several additional steps. Please follow the instructions given in: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Once CUDA is installed, you will be able to model training much faster.

## Step 6: Model deployment
The compiled model has all the side information required to run the model on our Edge AI StarterKit EVM and SDK.
- Purchase the Edge AI StarterKit EVM and download the [Edge AI StarterKit SDK](https://github.com/TexasInstruments/edgeai/blob/master/readme_sdk.md) to use our model deployment tools.
- For more information, see this link: https://www.ti.com/edgeai 
