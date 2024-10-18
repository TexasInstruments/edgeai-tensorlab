
## Release 10.0 (10_00_00)
- Git branch: **r10.0**
### Important Notes
- edgeai-hf-transformers (fork of https://github.com/huggingface/transformers) repository added, enabling training of Transformer based Image Classification, Segmentation and Object Detection Models. 
- Improvements for Model Surgery and Quantization in edgeai-modeloptimization
- Several additional models added in edgeai-modelzoo and edgeai-benchmark
### Updates
#### Date: 18 October 2024
- tidl_tools update - bug fixes: edgeai-benchmark setup_pc.sh has been updated to install tidl_tools version 10_00_08_00. See [release notes of edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools/releases) to understand the bug fixes that it has.
#### Date: September 2024
- Tech Reports added(**Object detection, PyTorch FX based Quantization**) - see this [link](./tech_reports/README.md)
- edgeai-mmpose(fork of open-mmlab/mmpose for efficient keypoint detection), edgeai-hf-transformers(fork of huggingface/transformers)
- Tools: [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) version used: **10_00_06_00**
- [Source code](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r10.0)
- [Model Zoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/r10.0/edgeai-modelzoo)
- [Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/10_00_00/)

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
