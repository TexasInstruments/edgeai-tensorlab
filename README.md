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


## Setup instructions
We have tested this tool in Ubuntu 18.04 and with Python 3.6 (Note: Currently edgeai-tidl-tools supports only Python 3.6)
We recommend the Miniconda Python distribution from: https://docs.conda.io/en/latest/miniconda.html

The tools depends on several repositories that we have published at https://github.com/TexasInstruments
The repositories mentioned below must be cloned into the same parent folder containing this tool. i.e. the model structure should be:

parent_folder
    |
    |--edgeai-benchmark
    |--edgeai-torchvision
    |--edgeai-mmdetection
    |--edgeai-modelzoo
    |--edgeai-modelmaker


After cloning, go into the setup folders provided in each repository and execute them. 

Note: edgeai-modelzoo doesn't have a setup script.
Note: There is no need to clone edgeai-tidl-tools, as edgeai-benchmark takes care of the dependencies provided by edgeai-tidl-tools as well.

## Data annotation

Currently [COCO JSON format](https://cocodataset.org/#format-data) and LabelStudio JSOM-Min format are supported. LabelStudio supports exporting dataset to COCO JSON format and JSOM-Min format.

### Annotation Instructions:
- Install [Label Studio](https://labelstud.io/) following the instructions given in that website.
- Once installed, it can be launched by running
```bash
./run_label_studio.sh
```
- Create a task such as Image classification or Object detection. Upload annotation and export the annotations into COCO JSON or JSON-Min formats.
- Copy the json file and int images to a suitable folder. Have a look at the following example. It is not neccarry to follow the same exact directory structore as below, but he key points to remember are (1) there must be an "images" folder containing (2) there must be an annotations folder containing the annotation json file.

data / labelstudio / animal_detection
                              |
                              |-images
                              |     |--copy the image files here
                              |
                              |-annotations
                                    |
                                    |--instances.json


data / labelstudio / animal_classification
                              |
                              |-images
                              |     |--copy the image files here
                              |
                              |-annotations
                                    |
                                    |--labels.json

If the dataset has already been split into train and validation set already, ti is possible to provide those path separately. Have a look at the examples given in [run_ai_backend.py](./run_ai_backend.py)

## Model training and compilation

Have a look at the parameters specified in [run_ai_backend.py](./run_ai_backend.py) and make any changes necessary. Check of the dataset paths provided are correct.

This tool can be invoked for model training and compilation by running:
```bash
./run_ai_backend.sh
```

After the model compilation, the compiled models will be available in the folder [./data/projects](./data/projects)


## Mode deployment
The compiled model has all the side information required to easily run the model on our EdgeAI-StarterKit EVM and SDK. 
- Purchase the [EdgeAI-StarterKit EVM](https://www.ti.com/tool/SK-TDA4VM) if you do not have it: it costs only a maximum of $200
- Download the [EdgeAI-StarterKit SDK](https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-SK-TDA4VM) and follow the instructions given there.
