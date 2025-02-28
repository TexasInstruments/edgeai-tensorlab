# edgeai-benchmark

### Notice
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai
- https://github.com/TexasInstruments/edgeai
- https://dev.ti.com/edgeai/

<hr>

This repository provides a collection of scripts for various image recognition tasks such as classification, segmentation, detection and keypoint detection. Features supported are:
- Model Compilation, Inference, Accuracy & Performance benchmarking of Deep Neural Networks (DNN). 
- Aspects such dataset loading, pre-processing and post-processing and accuracy computation are taken care for the models in our model zoo.
- These benchmarks in this repository can be run either in PC simulation mode or on EVM/device. 

Getting the correct functionality and accuracy with DNN Models is not easy. Several aspects such as dataset loading, pre-processing and post-processing operations have to be matched to that of the original training framework to get meaningful functionality and accuracy. There is much difference in these operations across various popular models and much effort has gone into matching that functionality.

<hr>

## Release Notes

* Release name/tag: r10.1
* tidl_tools version: 10_01_04_01

### New models
We are in the process of adding support for several new models. Configs for models verified in this release are in this repository and the models are available in edgeai-modelzoo. The following new models have been verified:

| Model name                    | Model Type                            | Source repository      |
|-------------------------------|---------------------------------------|------------------------|
| swin_tiny_patch4_window7_224  | Image Clasifiation, Transformer       | edgeai-hf-transformers |
| deit_tiny_patch16_224         | Image Clasifiation, Transformer       | edgeai-hf-transformers |
| levit_128_224                 | Image Clasifiation, Transformer       | edgeai-hf-transformers |
| segformer_b0                  | Semantic Segmentation, Transformer    | edgeai-hf-transformers |
| YOLOv7 (mutliple flavours)    | Object Detection                      | edgeai-mmdetection     |
| YOLOv9 (multiple flavours)    | Object Detection                      | edgeai-mmdetection     |
| efficientdet_effb0_bifpn_lite | Object Detection                      | edgeai-mmdetection     |
| YOLOXPose                     | Keypoint detection / Human Pose       | edgeai-mmpose          |


### Note
The tidl_tools installed in this version is compatible with the publicly released version of 10.1 SDK. In order to take advantage of subsequent bugfixes, it is possible to set 'advanced_options:c7x_firmware_version' to a higher value than what is supported in the publicly released SDK. Two steps are required for this:
- Set 'advanced_options:c7x_firmware_version' to 10_01_04_00 in this repository while model compilation. For this, use  [run_benchmarks_pc_firmware_update.sh](./run_benchmarks_pc_firmware_update.sh) for model compilation, which sets the argument `--c7x_firmware_version 10_01_04_00`
- Do firmware update in SDK - for more info see [version compatibility table](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md)

<hr>

## Version information & firware update
The tidl_tools that are installed by the setup file in this repository need to correspond to the firmware used in Edge AI SDK / TIDL release. Sometimes the tidl_tools installed by the setup file in this repository may not be matching with the firmware in the publicly released SDK (tidl_tools in this repository can sometimes be ahead). In that case there are multiple options:

### Specify firmware version while model compilation

### Firmware update
New versions of fimrware often comes with improvements and fixes - hence it is recommended to sue that. If you you would like to use the new firmware in system, the EVM/device can be updated by using the script [update_target.sh in edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/update_target.sh)

### Targeting previous versions of SDK
If you are targeting an older version of SDK, checkout the corresponding git branch in this repository and use that. For example, branch r10 for SDK version 10.0

<hr>

## Important features:
- Runs on both PC Simulation (model compilation and inference) and on EVM (model inference only).
- This package can be used for accuracy and performance (inferene time) estimates.
- Most of the models in TI ModelZoo [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modelzoo) is supported off-the-shelf in this package. Custom model benchmark can also be easily done (please refer to the documentation and example).
- Uses [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) for model compilation and inference. edgeai-tidl-tools can take a float model and compile it using PTQ (with an iterative calibration procedure) to an INT model for use on target. It can also accept a pre-quantized model to avoid the iterative calibration, so that the compilation is instantaneous. 
- Read more about quantization in general and specifically about pre-quantized models at [edgeai-modeloptimization/torchmodelopt](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modeloptimization/torchmodelopt)

<hr>

## Supported SOCs
At the moment, this repository supports compilation and inference for the following SoCs: **TDA4VM**, **AM68A**, **AM62A**, *AM67A**, **AM69A**, **AM62**

A reference to <SOC> in this repository as commandline argument to the scripts refer to one of these SoCs.

This <SOC> argument is used for multiple purposes:
- To set TIDL_TOOLS_PATH and LD_LIBRARY_PATH used to point to the correct tidl_tools for a device
- To choose the correct preset (of compilation flags) from the dictionary TARGET_DEVICE_SETTINGS_PRESETS in [constants.py](./edgeai_benchmark/constants.py)

More details regarding SoCs and devices can be seen at the [Edge AI landing repository](https://github.com/TexasInstruments/edgeai/blob/main/readme_sdk.md).

<hr>

## Setup on PC
See the [setup instructions](./docs/setup_instructions.md)

<hr>

## Components of this repository
This repository is generic and can be used with a variety of runtimes and models supported by TIDL. This repository contains the following parts:

- **edgeai_benchmark**: Core scritps for core model compilation, inference and accuracy benchmark scripts provided as a python package (that can be imported using: *import edgeai_benchmark* or using: *from edgeai_benchmark import*)
- **scripts**: These are the top level python scripts - to compile models, to infer and do accuracy & performance benchmark, to collect accuracy report and to package the generated artifacts.
- **configs**: These are the actual configurations of to run the models in the model zoo. These configurations define the Dataset parameters, pre-processing, inference session parameters and post-processing.

The top level python **scripts** are launched via bash scripts in the root of this repo. 

<hr>

## Usage on PC
See the [usage instructions](./docs/usage.md)

<hr>

## Compiling Custom Models on PC
See the **[instructions to compile custom models](./docs/custom_models.md)**

<hr>

## Pre-Complied Model Artifacts 
See [pre-compiled model artifacts](./docs/precompiled_modelartifacts.md)

<hr>

## Setup and Usage on development board/EVM
The compiled models can be used for inference on development board/EVM. See **[setup and usage instruction for EVM](./docs/usage_evm.md)**

<hr>
<hr>

## References
[References](./docs/refernces.md)