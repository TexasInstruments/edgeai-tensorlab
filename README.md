# EdgeAI-ModelMaker

### Notice
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai 
- https://github.com/TexasInstruments/edgeai

<hr>

EdgeAI-ModelMaker is a model development tool that contains dataset handling, model training and compilation. 

Currently, it doesn't have an integrated feature to annotate data, but can accept annotated Dataset from a tool such as [Label Studio](https://labelstud.io/)

The following are the key functionality supported by this tool:
- Dataset handling: Accept dataset in various formats. Split the dataset into train and validation splits (if it is not already split).
- Model training: Model training repositories such as [edgeai-torchvision](https://github.com/TexasInstruments/edgeai-torchvision) and [edgeai-mmdetection](https://github.com/TexasInstruments/edgeai-mmdetection) are integrated. Several models with pretrained checkpoints are incorporated for each of these repositories.
- Model compilation: Model compilation tools [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) and [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) for TI's edgeai SoCs have been integrated.

These functionalities that are supported are fully integrated and the user doesn't need to do anything apart from setting some parameters in the beginning.  


## OS & Environment 

### With docker environment

Install docker if you don't have it already. The following steps are for installation on Ubuntu 18.04
```
sudo apt update
sudo apt install docker.io
sudo usermod -aG docker ${USER}
sudo systemctl start docker
sudo systemctl enable docker
# logout and log back in and docker should be ready to use.
```

Build docker image:
```
./docker_build.sh
```

Run docker container to bring up the container terminal on docker:
```
./docker_run.sh
```

During docker run, we map the parent directory of this folder to /home/edgeai/code. This is to easily share code and data between the host and the docker container. Inside the docker terminal, change directory to where this folder is mapped to:
```
cd /home/edgeai/code/edgeai-modelmaker
```

###  With native Ubuntu environment
We have tested this tool in Ubuntu 18.04 and with Python 3.6 (Note: Currently edgeai-tidl-tools supports only Python 3.6)

We recommend the Miniconda Python distribution from: https://docs.conda.io/en/latest/miniconda.html


## Setup the model training and compilation repositories

This tool depends on several repositories that we have published at https://github.com/TexasInstruments

The following setup script can take care of cloning the repositories and running their setup scripts.
```
./setup_all.sh
```

However, for some reason, if you wish to do it manually, the repositories mentioned below are to be cloned into the same parent directory containing this folder. i.e. the folder structure should be:

parent_directory<br>
    |<br>
    |--edgeai-benchmark<br>
    |--edgeai-torchvision<br>
    |--edgeai-mmdetection<br>
    |--edgeai-modelzoo<br>
    |--edgeai-modelmaker<br>


Note: edgeai-modelzoo doesn't have a setup script.<br>
Note: There is no need to clone edgeai-tidl-tools, as edgeai-benchmark takes care of the dependencies provided by edgeai-tidl-tools as well.


## Run examples

Object detection example
```
./run_modelmaker.sh config_detection.yaml
```

Image classification example
```
./run_modelmaker.sh config_classification.yaml
```

## Data annotation

Currently [COCO JSON format](https://cocodataset.org/#format-data) and LabelStudio JSOM-Min format are supported. LabelStudio supports exporting dataset to COCO JSON format and JSOM-Min format.

### Annotation Instructions:
- Data Annotation can be done in any suitable tool as long as the format of the annotated is supported in this repository. The following description to use Label Studio is just an example.
- Install [Label Studio](https://labelstud.io/) following the instructions given in that website.
- Once installed, it can be launched by running
```bash
./run_labelstudio.sh
```

- Create a project such as "Image classification" or "Object detection" in Label Studio. Import the images into the project and do the annotation.
- After the annotation, export the annotations into COCO JSON or JSON-Min formats.

## Make the dataset available to ModelMaker
- Copy the json file and images to a suitable folder. Have a look at the following example. It is not necessary to follow the same exact directory structure as below, but the key points to remember are (1) there must be an "images" folder containing (2) there must be an annotations folder containing the annotation json file with the name given below.

### Object Detection example
- An object detection dataset should look like this:
data / datasets / animal_detection<br>
                              |<br>
                              |-images<br>
                              |     |--copy the image files here<br>
                              |<br>
                              |-annotations<br>
                                    |<br>
                                    |--instances.json<br>
- In the config yaml file, provide the name of the dataset (animal_detection in this example) in the field dataset_name and provide the path (./data/datasets/animal_detection in this example) in the field input_data_path.

### Image Classification example
- An image classification dataset should look like this:
data / datasets / animal_classification<br>
                              |<br>
                              |-images<br>
                              |     |--copy the image files here<br>
                              |<br>
                              |-annotations<br>
                                    |<br>
                                    |--labels.json<br>

- In the config yaml file, provide the name of the dataset (animal_classification in this example) in the field dataset_name and provide the path (./data/datasets/animal_classification in this example) in the field input_data_path.


## Model training and compilation

Have a look at the parameters specified in [run_modelmaker.sh](./run_modelmaker.sh), [run_modelmaker.py](./scripts/run_modelmaker.py) and the config yaml files and make any changes necessary. Check if the dataset path provided in input_data_path is correct.

Note: If the dataset has already been split into train and validation set already, it is possible to provide those paths separately as a tuple in input_data_path.

This tool can be invoked for model training and compilation by running:
```bash
./run_modelmaker.sh
```

After the model compilation, the compiled models will be available in the folder [./data/projects](./data/projects)


## Mode deployment
The compiled model has all the side information required to easily run the model on our EdgeAI-StarterKit EVM and SDK. 
- Purchase the [EdgeAI-StarterKit EVM](https://www.ti.com/tool/SK-TDA4VM) if you do not have it.
- Download the [EdgeAI-StarterKit SDK](https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-SK-TDA4VM) and follow the instructions given there.
