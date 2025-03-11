
## Overview
This is a tool for collecting data, training and compiling AI models for use on TI's embedded processors. The compiled models can be deployed on a local development board. A live preview/demo is also provided to inspect the quality of the developed model while it runs on the development board.

## Development flow
Bring your own data (BYOD): Retrain models from TI Model Zoo to fine-tune with your own data.

## Tasks supported
* Image Classification
* Object Detection

## Target device setup overview
In order to perform data capture from device, live preview or model deployment, a local area network connection (LAN) to the development board is required. To do this, please follow the steps below:
* Step 1: Make sure that you have a physical development board (of the specific device) with you. Refer the details below to understand how to procure it.
* Step 2: Download the SDK binary and flash an SD card as explained in the SDK.
* Step 3: Make sure that the development board is put in the same local area network (via ethernet or WiFI) as the computer where you are running the browser to use this service. Also connect the development board to the computer via USB serial connection - this is required to detect the IP address of the development board.
* Step 4: Connect a USB camera to the development board.
* Step 5: Power ON the development board.
* Step 6: On the top bar of the GUI of this service, click on Options | Serial port settings and follow the instructions to do TI Cloud Agent setup.
* Step 7: On the "Connect Device Camera" pop-up, click on the search icon to detect the IP address of the development board and connect to it.

## Supported target devices
These are the devices that are supported currently. As additional devices are supported, this section will be updated.

### TDA4VM
* Product information: https://www.ti.com/product/TDA4VM
* Development board: https://www.ti.com/tool/SK-TDA4VM
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-SK-TDA4VM
* SDK documentation & board setup: See Edge AI documentation at https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-SK-TDA4VM
* SDK release: 10_01_00

### AM62A
* Product information: https://www.ti.com/product/AM62A7
* Development board: https://www.ti.com/tool/SK-AM62A-LP
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM62A
* SDK documentation & board setup: See Edge AI documentation at https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM62A
* SDK release: 10_01_00

### AM68A
* Product information: https://www.ti.com/product/AM68A
* Development board: https://www.ti.com/tool/SK-AM68
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM68A
* SDK documentation & board setup: See Edge AI documentation at https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM68A
* SDK release: 10_01_00

### AM69A
* Product information: https://www.ti.com/product/AM69A
* Development board: https://www.ti.com/tool/SK-AM69
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM69A
* SDK documentation & board setup: See Edge AI documentation at https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM69A
* SDK release: 10_01_00

### AM62
* Product information: https://www.ti.com/product/AM625
* Development board: https://www.ti.com/tool/SK-AM62
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM62X
* SDK documentation & board setup: See analytics application and Edge AI documentation at https://www.ti.com/tool/SK-AM62#order-start-development
* SDK release: 10_01_00

## Additional information
* Edge AI introduction: https://ti.com/edgeai
* Edge AI model development information: https://github.com/TexasInstruments/edgeai
* Edge AI tools introduction: https://dev.ti.com/edgeai/


## Dataset format
- The dataset format is similar to that of the [COCO](https://cocodataset.org/) dataset, but there are some changes as explained below.
- The annotated json file and images must be under a suitable folder with the dataset name. 
- Under the folder with dataset name, the following folders must exist: 
- (1) there must be an "images" folder containing the images
- (2) there must be an "annotations" folder containing the annotation json file with the name given below.
- Notes on preparing the dataset zip file:
- (1) To prepare the dataset zip for your dataset in a windows PC, navigate inside that folder with the dataset name, select the folders images and annotations, right-click and then click on Sent to Compressed (zipped) folder.
- (2) To prepare the dataset zip file for your dataset in a Linux PC, navigate inside that folder with the dataset name, select the folders images and annotations, right-click and then select Compress.
- (3) Do not click on the folder with the dataset name to zip it - instead, go inside the folder and zip it, so that the images and annotations folders will be directly at the base of the zip file.

#### Object Detection dataset format
An object detection dataset should have the following structure. 
```
images/the image files should be here
annotations/instances.json
```

- The default annotation file name for object detection is instances.json
- The format of the annotation file is similar to that of the [COCO dataset 2017 Train/Val annotations](https://cocodataset.org/#download) - a json file containing 'info', 'images', 'categories' and 'annotations'.
- Look at the example dataset [animal_detection](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_detection.zip) to understand more.

#### Image Classification dataset format
An image classification dataset should have the following structure. (Use a suitable dataset name instead of dataset_name).
```
images/the image files should be here
annotations/instances.json
```

- The default annotation file name for image classification is instances.json
- The format of the annotation file is similar to that of the COCO dataset - a json file containing 'info', 'images', 'categories' and 'annotations'. However, one difference is that the bounding box information is not used for classification task and need not be present. The category information in each annotation (called the 'id' field) is needed.
- Look at the example dataset [animal_classification](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_classification.zip) to understand more.

## Model deployment
- The deploy page provides a button to download the compiled model artifacts to the development board. 
- The downloaded model artifacts are located in a folder inside /opt/projects. It can be used with edgeai-gst-apps included in the SDK to run inference. 
- Please see the section "Edge AI sample apps" in the SDK documentation for more information.

## Glossary of terms

### TRAINING
#### Epochs
Epoch is a term that is used to indicate a pass over the entire training dataset. It is a hyper parameter that can be tuned to get best accuracy. Eg. A model trained for 30 Epochs may give better accuracy than a model trained for 15 Epochs.
#### Learning rate
Learning Rate determines the step size used by the optimization algorithm at each iteration while moving towards the optimal solution. It is a hyper parameter that can be tuned to get best accuracy. Eg. A small Learning Rate typically gives good accuracy while fine tuning a model for a different task.
#### Batch size
Batch size specifies the number of inputs that are propagated through the neural network in one iteration. Several such iterations make up one Epoch.Higher batch size require higher memory and too low batch size can typically impact the accuracy.
#### Weight decay
Weight decay is a regularization technique that can improve stability and generalization of a machine learning algorithm. It is typically done using L2 regularization that penalizes parameters (weights, biases) according to their L2 norm.
### COMPILATION
#### Calibration frames
Calibration is a process of improving the accuracy during fixed point quantization. Typically, higher number of Calibration Frames give higher accuracy, but it can also be time consuming.
#### Calibration iterations
Calibration is a process of improving the accuracy during fixed point quantization. Calibration happens in iterations. Typically, higher number of Calibration Iterations give higher accuracy, but it can also be time consuming.
#### Tensor bits
Bitdepth used to quantize the weights and activations in the neural network. The neural network inference happens at this bit precision. 
#### Detection threshold
Also called Confidence Threshold. A threshold used to select good detection boxes. This is typically applied before a before the Non Max Suppression. Higher Detection Threshold means less false detections (False Positives), but may also result in misses (False Negatives). 
#### Detection topK
Number of detection boxes to be selected during the initial shortlisting before the Non Max Suppression.A higher number is typically used while measuring accuracy, but may impact the performance. 
### DEPLOY
#### Download trained model
Trained model can be downloaded to the PC for inspection.
#### Download compiled model artifacts to PC
Compiled model can be downloaded to the PC for inspection.
#### Download compiled model artifacts to EVM
Compiled model can be downloaded into the EVM for running model inference in SDK. Instructions are given in the help section.
