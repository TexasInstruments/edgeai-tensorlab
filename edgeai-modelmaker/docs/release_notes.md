
<hr>

Note: We follow the EdgeAI-SDK (Processor SDK Linux with Edge AI) release numbering. For example, the models compiled with 9.2.0 version (r9.2 branch) works in 09.02.00 (09_02_00) SDK release.

## 9.2.0

### Git branch name: r9.2

### New Features
- Updated to support TIDL & SDK version 9.2

<hr>


## 9.1.0

### Git branch name: r9.1

### New Features
##### Task Types
- Support for Key point detection task has added. It is enabled by using the repository https://github.com/TexasInstruments/edgeai-yolox which implements the [YOLO-pose](https://arxiv.org/abs/2204.06806) method.
- Example dataset for Keypoint detection is yet to be added, but will be added in r9.1 branch itself in about a month or so.

##### Devices
- Support for [AM62](https://www.ti.com/product/AM625) SoC and it's starter kit [SK-AM62](https://www.ti.com/tool/SK-AM62) is enabled. Note 1: Unlike other Analytics SoCs, AM62 does not have a DSP or Matrix Multiplier Acceletator; so the model inference runs on the ARM CPU itself. Note 2: Currently we use floating point models in AM62, but we have a desire to use INT8 models to speedup inference in the future.

<hr>

## 9.0.0

### Git branch name: r9.0

### New Features
##### Task Types
- Support for Semantic Segmentation has been added.

<hr>

