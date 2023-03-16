
## Overview
This is a tool for collecting data, training and compiling AI models for use in TI embedded processors. The compiled models can be deployed on a local development board. A live preview/demo is also provided to inspect the quality of the developed model while it runs on the development board.

## Development flow
Bring your own data (BYOD): Retrain TI models from TI Model Zoo to fine-tune performance for your unique application requirements.

## Tasks supported
* Image Classification
* Object Detection

## Target device setup overview
Data capture from a development board is supported over ethernet connection. The live stream appears in the browser window and user can capture frames as needed. Similarly live preview/inference/demo can also be streamed into the browser window. To establish a connection with a physical development board over ethernet, please follow the steps below. Also use the supported SDK version for that device - given in the details below.
* Step 1: 'Make sure that you have physical development board of specific device with you procured, refer below to find how to procure for each specific device.
* Step 2: 'Download the image to be flashed in SD card (refer steps 3)
* Step 3: 'Make sure that the development board is setup and also put in the same local area network as the computer where you are using this service. Also connect a USB camera to the dvelopment board.
* Step 4: 'Get the IP address of the development board using serial port connection
* Step 5: 'Connect to the development board using ssh and run device agent service as mentioned below. 
```
ssh root@<ip_address_of_dev_board> 
cd /opt/edgeai-studio-agent/src 
python3 ./device_agent.py
```
* Step 6: Now you can connect to development board from model composer by providing the IP address of development board.

## Supported target devices
These are the devices that are supported currently. As additional devices are supported, this section will be updated.

### TDA4VM
* Product information: https://www.ti.com/product/TDA4VM
* Development board: https://www.ti.com/tool/SK-TDA4VM
* Software development kit (SDK): https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-SK-TDA4VM
* Steps to setup board: https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-sk-tda4vm/08_06_00/exports/docs/getting_started.html
* SDK release: 08_06_00

### AM62A
* Product information: https://www.ti.com/product/AM62A7
* Development board: https://www.ti.com/tool/SK-AM62A-LP
* Software development kit (SDK): https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM62A
* Steps to setup board: https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-edgeai/AM62AX/08_06_00/exports/docs/devices/AM62AX/linux/getting_started.html
* SDK release: 08_06_00

### AM68A
* Product information: https://www.ti.com/product/AM68A
* Development board: https://www.ti.com/tool/SK-AM68
* Software development kit (SDK): https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM68A
* Steps to setup board: https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-edgeai/AM68A/08_06_00/exports/docs/devices/AM68A/linux/getting_started.html
* SDK release: 08_06_00

## Additional information
* Edge AI introduction: https://ti.com/edgeai
* Edge AI tools introduction: https://dev.ti.com/edgeai/
* Edge AI model development information: https://github.com/TexasInstruments/edgeai

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
