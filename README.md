# EdgeAI-ModelMaker

#### Notice
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai 
- https://github.com/TexasInstruments/edgeai


<hr>

EdgeAI-ModelMaker is an end-to-end model development tool that contains dataset handling, model training and compilation. Currently, it doesn't have an integrated feature to annotate data, but can accept annotated Dataset from a tool such as [Label Studio](https://labelstud.io/)

We have published several repositories for model training, model compilation and modelzoo as explained in our [edgeai gihub page](https://github.com/TexasInstruments/edgeai). This repository is an attempt to stitch several of them together to make a simple and consistent interface for model development. This does not support all the models that can be trained and compiled using our tools, but only a subset. This is a commandline tool and requires a Linux PC.

The following are the key functionality supported by this tool:
- Dataset handling: This dataset formats supported by this tool is described in a section below. This can convert dataset formats and can automatically split the given dataset into train and validation sets (if it is not already split).
- Model training: Model training repositories such as [edgeai-torchvision](https://github.com/TexasInstruments/edgeai-torchvision) and [edgeai-mmdetection](https://github.com/TexasInstruments/edgeai-mmdetection) are integrated. Several models with pretrained checkpoints are incorporated for each of these repositories. 
- Model compilation: Model compilation tools [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) and [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) for TI's EdgeAI SoCs have been integrated.

Tasks and Models
- Image classification and Object detection tasks are supported currently.
- The models supported and their parameters are collected in a [description file](./data/descriptions/description_vision.yaml) for convenience.

These functionalities that are supported are fully integrated and the user can control it by setting  parameters in the config file.  


## Step 1: OS & Environment 

This repository can be used from native Ubuntu bash terminal directly or from within a docker environment.

#### Step 1, Option 1: With native Ubuntu environment and pyenv (recommended)
We have tested this tool in Ubuntu 18.04 and with Python 3.6 (Note: Currently edgeai-tidl-tools supports only Python 3.6). We have not tested this on other Linux distributions.

In this option, we describe using this repository with the pyenv environment manager. 

Step 1.1a: Make sure that you are using bash shell. If it is not bash shell, change it to bash. Verify it by typing:
```
echo ${SHELL}
```

Step 1.2a: Install system dependencies
```
sudo apt install -y build-essential xz-utils libreadline-dev libncurses5-dev libssl-dev 
sudo apt install -y libbz2-dev libsqlite3-dev liblzma-dev python3.6-tk
sudo apt install -y curl libjpeg-dev zlib1g-dev graphviz graphviz-dev 
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


Step 1.4a: Install Python 3.6 in pyenv and create an environment
```
pyenv install 3.6
pyenv virtualenv 3.6 py36
pyenv rehash
pyenv activate py36
pip install --upgrade pip
pip install --upgrade setuptools
```

Step 1.5a: **Activate the Python environment.** This activation step needs to be done everytime one starts a new terminal or shell. (Alternately, this also can be written to the .bashrc, so that this will be the default penv environment).
```
pyenv activate py36
```


####  Step 1, Option 2: With native Ubuntu environment and Miniconda (deprecated)
We have tested this tool in Ubuntu 18.04 and with Python 3.6 (Note: Currently edgeai-tidl-tools supports only Python 3.6). We have not tested this on other Linux distributions.

We can also use Miniconda Python distribution from: https://docs.conda.io/en/latest/miniconda.html

Step 1.1b: Make sure that you are using bash shell. If it is not bash shell, change it to bash. Verify it by typing:
```
echo ${SHELL}
```

Step 1.2b: Install Miniconda Python distribution:
```
./conda_install.sh
```

Step 1.3b: At this point, conda installation is complete. Close your current terminal and start a new one. This is so that the change that the install script wrote to .bashrc takes effect. Create conda Python 3.6 environment with a suitable name (py36 below - but, it can be anything):
```
conda create -y -n py36 python=3.6
```

**These conda installation steps above need to be done only once for a user.** 


Step 1.4b: **Activate the Python environment.** This activation step needs to be done everytime one starts a new terminal or shell. (Alternately, this also can be written to the .bashrc, so that this will be the default conda environment).
```
conda activate py36
```


#### Step 1, Option 3: With docker environment

Step 1.1c: Install docker if you don't have it already. The following steps are for installation on Ubuntu 18.04
```
./docker/docker_setup.sh
```

Step 1.2c: Build docker image:
```
./docker/docker_build.sh
```

Step 1.3c: Run docker container to bring up the container terminal on docker:
```
./docker/docker_run.sh
```

Source .bashrc to update the PATH
```
source /opt/.bashrc
```

Step 1.4c: During docker run, we map the parent directory of this folder to /opt/code This is to easily share code and data between the host and the docker container. Inside the docker terminal, change directory to where this folder is mapped to:
```
cd /opt/code/edgeai-modelmaker
```


## Step 2: Setup the model training and compilation repositories

This tool depends on several repositories that we have published at https://github.com/TexasInstruments

The following setup script can take care of cloning the required repositories and running their setup scripts.
```
./setup_all.sh
```

If the script runs sucessfully, you should have this directory structure: 
<pre>
parent_directory
    |
    |--edgeai-modelzoo
    |--edgeai-torchvision
    |--edgeai-mmdetection
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


### Enabling optional components

In setup_all.sh, there are flags to enable additional models

PLUGINS_ENABLE_GPL: Setting this to 1 during setup adds support for YOLOv5 ti_lite models using the repository [edgeai-yolov5](https://github.com/TexasInstruments/edgeai-yolov5) .

PLUGINS_ENABLE_EXTRA: Setting this to 1 during setup enables additional models. 


## Step 3: Run the ready-made examples

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
The list of target devices supported depends on the tidl-tools installed by [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark). Currently **TDA4VM, AM68A, AM62A and AM69A** are supported.


## Step 4: Prepare your own dataset with your own images and object types (Data annotation)
- This section explains how to create and annotate your own dataset with your own object classes.
- Data Annotation can be done in any suitable tool as long as the format of the annotated is supported in this repository. The following description to use Label Studio is just an example.
- The annotation file must be in [COCO JSON](https://cocodataset.org/#format-data) format. LabelStudio supports exporting the dataset to COCO JSON format for Object Detection. For Image Classification, Label Studio can export in JSOM-Min format and this tool provides a converter script to convert to COCO JSON like format.

#### Step 4.1: Install LabelStudio
- Label Studio can be installed using the following command:
```bash
pip install -r requirements/labelstudio.txt
```

#### Step 4.2: Run LabelStudio
- Once installed, it can be launched by running
```bash
./run_labelstudio.sh
```

#### Step 4.3: How to use Label Studio for data annotation
- Create a new project in Label Studio and give it a name in the "Project Name" tab. 
- In the Data Import tab upload your images. (You can upload multiple times if your images are located in various folders in the source location).
- In the tab named "Labelling Setup" choose "Object Detction with Bounding Boxes" or "Image classification" depending on the task that you would like to annotate for.
- Remove the existing "Choices" and add your Label Choices (Object Types) that you would like to annotate. Clip on Save.
- Now the "project page" is shown with list of images and their previews. 
- Now click on an image listed to go to the "Labelling" page. Select the object category for the given image. For Object detection also draw boxes around the objects of interest. (Before drawing a box make sure the correct label choice below the mage is selected).
- Do not forget to click "Submit" before moving on to the next image. The annotations done for an image is saved only when "Submit" is clicked.
- After annotating the required images, go back to the "project page", by clicking ont he project name displayed on top. From this page we can export the annotation.
- Export the annotation in COCO-JSON (For Object Detection) of JSON-MIN (For Image Classification). Do not export using the JSON format. COCO-JSON format exported by Label Studio can be directly accepted by this ModelMaker tool. 
- However, the JSON-MIN format has to be converted to the COCO-JSON format by using an example given in [run_convert_dataset.sh](./run_convert_dataset.sh). For Image Classification task, use source_format as labelstudio_classification. Label Studio can also export into JSON-MIN for Object detection. In case you did that, use source_format as labelstudio_detection for the converter script.


## Step 5: Dataset format
- The dataset format is similar to that of the [COCO](https://cocodataset.org/) dataset, but there are some changes as explained below.
- The annotated json file and images must be under a suitable folder with the dataset name. 
- Under the folder with dataset name, the following folders must exist: 
- (1) there must be an "images" folder containing the images
- (2) there must be an annotations folder containing the annotation json file with the name given below.

#### Object Detection dataset format
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
- The format of the annotation file is similar to that of the [COCO dataset 2017 Train/Val annotations](https://cocodataset.org/#download) - a json file containing 'info', 'images', 'categories' and 'annotations'.
- Look at the example dataset [animal_classification](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_classification.zip) to understand further.
- In the config file, provide the name of the dataset (dataset_name in this example) in the field dataset_name and provide the path or URL in the field input_data_path.
- Then the ModelMaker tool can be invoked with the config file.

#### Image Classification dataset format
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
- The format of the annotation file is similar to that of the COCO dataset - a json file containing 'info', 'images', 'categories' and 'annotations'. However, one difference is that the bounding box information is not used for classification task and need not be present. The category information in each annotation (called the 'id' field) is needed.
- Look at the example dataset [animal_detection](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_detection.zip) to understand further.
- In the config file, provide the name of the dataset (dataset_name in this example) in the field dataset_name and provide the path or URL in the field input_data_path.
- Then the ModelMaker tool can be invoked with the config file.


#### Notes
If the dataset has already been split into train and validation set already, it is possible to provide those paths separately as a tuple in input_data_path.

After the model compilation, the compiled models will be available in a folder inside [./data/projects](./data/projects)

If you have a dataset in another format, use the script provided to convert it into the COCO jSON format. See the examples given in [run_convert_dataset.sh](./run_convert_dataset.sh) for example conversions.

The config file can be in .yaml or in .json format


## Step 6: Accelerated Training using GPUs (Optional) 

Note: **This section is for advanced users only**. Familiarity with NVIDIA GPU and CUDA driver installation is assumed.

This tool can train models either on CPU or on GPUs. By default, CPU based training is used. 

It is possible to speedup model training significantly using GPUs (with CUDA support) - if you have those GPUs in the PC. The PyTorch version that we install is capable of supporting CUDA GPUs. However, there are additional steps to be followed to enable GPU support in training.

Once the drivers are installed (as described in the appropriate section below), in the config file, set a value for num_gpus to a value greater than 0 (should not exceed the number of GPUs in the system) to enable GPU based training.

#### Option 1: When using Native Ubuntu Environment

The user has to install an appropriate NVIDIA GPU driver that supports the GPU being used.

The user also has to install CUDA Toolkit. See the [CUDA download instructions](https://developer.nvidia.com/cuda-downloads). The CUDA version that is installed must match the CUDA version used in the PyTorch installer - see [our edgeai-torchvision setup script](https://github.com/TexasInstruments/edgeai-torchvision/blob/master/setup.sh) to understand the CUDA version used. 

#### Option 2: When using docker environment

Enabling CUDA GPU support inside a docker environment requires several additional steps. Please follow the instructions given in: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Once CUDA is installed, you will be able to model training much faster.

## Step 7: Model deployment
The compiled model has all the side information required to run the model on our Edge AI StarterKit EVM and SDK.
- Purchase the Edge AI StarterKit EVM and download the [Edge AI StarterKit SDK](https://github.com/TexasInstruments/edgeai/blob/master/readme_sdk.md) to use our model deployment tools.
- For more information, see this link: https://www.ti.com/edgeai 
