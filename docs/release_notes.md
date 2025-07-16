<hr>


<hr>

## Release 11.0
* Release name: **11.0**
* Git branch: **r11.0**
* tidl_tools version: **11_00_08_00**
* [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) git tag: [11_00_08_00](https://github.com/TexasInstruments/edgeai-tidl-tools/releases/tag/11_00_08_00)
* Date: 2025 May - 2025 July

### New models in edgeai-modelzoo / edgeai-benchmark
We are in the process of adding support for several new models. Configs for models verified in this release are in this repository and the models are available in edgeai-modelzoo. The following new models have been verified:

| Model name                    | Model Type                            | Source repository |
|-------------------------------|---------------------------------------|------------------------|
| rtmdet lite version (multiple flavours) | Object Detection                      | edgeai-mmdetection     |
| fastbev (without temporal)    | Multi-view 3DOD for ADAS                      | edgeai-mmdetection3d     |
| bevformer_tiny    | Multi-view 3DOD for ADAS                      | edgeai-mmdetection3d     |

Note: The 3DOD models are trained with **pandaset dataset** (which is a Multi-view, Multi-modality ADAS / Automous Driving Dataset). edgeai-mmdetection3d and edgeai-benchmark now supports pandaset dataset. See more details of this dataset in edgeai-mmdetection3d.

### Update on 2025 July 15
* Accuracy fix for object detection models in edgeai-modelmaker and edgeai-mmdetection

<hr>

## Release 10.1
* Release name: **10.1**
* git branch: **r10.1**
* [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) git tag: [10_01_04_00](https://github.com/TexasInstruments/edgeai-tidl-tools/releases/tag/10_01_04_00)
* tidl_tools version: **10_01_04_01**
* Date: 2024 Dec ~ 2025 March

### edgeai-benchmark core runtime apis
[2025-03-29 update] edgai-benchmark/edgeai_benchmark/core contains simplified wrapper apis over the core runtimes that are easy to use and understand. See edgeai-benchmark/run_tutorials_script_pc.sh and edgeai-benchmark/tutorials to understand the usage of these core apis.

### New models in edgeai-modelzoo / edgeai-benchmark
We are in the process of adding support for several new models. Configs for models verified in this release are in this repository and the models are available in edgeai-modelzoo. The following new models have been verified:

| Model name                    | Model Type                            | Source repository |
|-------------------------------|---------------------------------------|------------------------|
| swin_tiny_patch4_window7_224  | Image Clasifiation, Transformer       | edgeai-hf-transformers |
| deit_tiny_patch16_224         | Image Clasifiation, Transformer       | edgeai-hf-transformers |
| levit_128_224                 | Image Clasifiation, Transformer       | edgeai-hf-transformers |
| segformer_b0                  | Semantic Segmentation, Transformer    | edgeai-hf-transformers |
| YOLOv7 (mutliple flavours)    | Object Detection                      | edgeai-mmdetection     |
| YOLOv9 (multiple flavours)    | Object Detection                      | edgeai-mmdetection     |
| efficientdet_effb0_bifpn_lite | Object Detection                      | edgeai-mmdetection     |
| YOLOXPose                     | Keypoint detection / Human Pose       | edgeai-mmpose          |

**Note**: some of these models don't work natively in 10.1, but require a **firmware update** in the SDK and model compilation with firmware version set corresponding to that in tidl-tools. A separate script called  run_benchmarks_firmware_update_pc.sh is provided in edgeai-benchmark to compile models with newer firmware.

### New models in edgeai-modelmaker
| Model name                    | Model Type                            | Source repository      |
|-------------------------------|---------------------------------------|------------------------|
| YOLOv7       | Object Detection                      | edgeai-mmdetection     |
| YOLOv9       | Object Detection                      | edgeai-mmdetection     |
| YOLOXPose                     | Keypoint detection / Human Pose       | edgeai-mmpose          |

**More details are in the [Release Notes](./docs/release_notes.md)**

### Tech reports
New Tech report on 3D Object Detection - see the section on "Tech Reports"

### Deprecations
edgeai-yolox repository is being deprecated - use [edgeai-mmpose](edgeai-mmpose) for Keypoint detection and [edgeai-mmdetection](edgeai-mmdetection) for detection YOLOX models. The previous version of edgeai-yolox is still available in the previous branches of this repository or [here](https://github.com/TexasInstruments/edgeai-yolox)

<hr>

## Release 10.0 (10_00)
- Git branch: **r10.0**
### Updates
#### Date: 18 October 2024
- git tag: v10.0.2
- tidl_tools update - bug fixes: edgeai-benchmark setup_pc.sh has been updated to install tidl_tools version **10_00_08_00**. See [release notes of edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools/releases) to understand the bug fixes that it has.
#### Date: September 2024
- git tag: v10.0.1
- Tech Reports added(**Object detection, PyTorch FX based Quantization**) - see this [link](./tech_reports/README.md)
- edgeai-hf-transformers (fork of https://github.com/huggingface/transformers) repository added, enabling training of Transformer based Image Classification, Segmentation and Object Detection Models. 
- Improvements for Model Surgery and Quantization in edgeai-modeloptimization
- Several additional models added in edgeai-modelzoo and edgeai-benchmark
- edgeai-mmpose(fork of open-mmlab/mmpose for efficient keypoint detection), edgeai-hf-transformers(fork of huggingface/transformers)
- Tools: [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) version used: **10_00_06_00**
- [Source code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r10.0)
- [Model Zoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r10.0/edgeai-modelzoo)
- [Model Zoo Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/10_00_00/)

<hr>

## Release 9.2 (09_02_00)
- Release branch: **r9.2**
- Date: April 2024
- [Source code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r9.2)
- [Model Zoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r9.2/edgeai-modelzoo)
- [Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/09_02_00/)

<hr>

## Release 9.1 (09_01_00)
- Release branch: **r9.1**
- Date: December 2023
- [Source code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r9.1)
- [Model Zoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r9.1/edgeai-modelzoo)
- [Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/09_01_00/)

<hr>

## Release 9.0 (09_00_00)
- Release branch: **r9.0**
- Date: August 2023
- [Source code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r9.1)
- [Model Zoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r9.1/edgeai-modelzoo)
- [Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/09_01_00/)

<hr>

## Release 8.6 (08_06_00)
- Release branch: **r8.6**
- Date: March 2023
- [Source code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r8.6)
- [Model Zoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r8.6/edgeai-modelzoo)
- [Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00/)

<hr>

## Release 8.2 (08_02_00)
- Release branch: **r8.2**
- Date: April 2022
- [Source code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r8.2)
- [Model Zoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r8.2/edgeai-modelzoo)
- [Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_02_00/)
#### Additional software & models
- [Models Code & Documentation edgeai-yolov5](https://github.com/TexasInstruments/edgeai-yolov5/tree/r8.2)
- [Compiled Model artifacts list for edgeai-yolov5](https://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/modelartifacts/8bits/artifacts.csv)
- [Release Manifest for edgeai-yolov5](https://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/manifest.html)

<hr>

## Release 8.1 (08_01_00)
- Date: December 2021
#### ModelZoo and compiled Model Artifacts
- [Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_01_00/)
#### Other changes
- This repository has been restructured. We now use the original mmdetection and added our changes on top of it as commits. This will help us to rebase from mmdetection as more and more features are added there. Please see more details in the usage.
- We now have a tool that can automatically replace some of embedded un-friendly layers.
#### Additional models
- Note: YOLOv5 models are not part of this repository but providing the link here: https://github.com/TexasInstruments/edgeai-yolov5/
- Note: YOLOv5 model artifacts are not part of this repository but providing the links here:  https://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_01_00_05/edgeai-yolov5/pretrained_models/modelartifacts/8bits/artifacts.csv

<hr>
