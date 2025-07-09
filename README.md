## Notice
* Read the documentation landing page before using this repository: [https://github.com/TexasInstruments/edgeai](https://github.com/TexasInstruments/edgeai)
* Specifically the section: [edgeai-mpu](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)

<hr>

# edgeai-tensorlab 
Edge AI model training, quantization, compilation/benchmark & Model Zoo

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
<hr>

## Notice
Our documentation landing pages are the following:
- https://www.ti.com/edgeai : Technology page summarizing TI’s edge AI software/hardware products 
- **https://github.com/TexasInstruments/edgeai** : Landing page for developers to understand overall software and tools offering. Read it before navigating into this repository.

<hr>
<hr>

## How to get started

Want do use Edge AI on TI's MPU devices - but don't know where to start? We have multiple solutions to help develop and deploy models.

### Develop your model

#### [EDGE-AI-STUDIO](https://www.ti.com/tool/EDGE-AI-STUDIO) - easy to use GUI tools 
* Model Composer: Capture images, annotate them, train and compile models using GUI.
* Model Analyzer: Use our hosted Jupyter notebooks to try model compilation online.

#### [edgeai-modelmaker](edgeai-modelmaker) - a commandline tool that supports Bring Your Own Data (BYOD) development flow 
* Use EDGE-AI-STUDIO Model Composer (above GUI tool) to collect and annotate data to create a dataset
* Export the dataset on to your machine.
* Use edgeai-modelmaker to train a model using the dataset. edgeai-modelmaker allows you to tweak more parameters than what is supported in the GUI tool
* It is fully customizable, so you can look at how models and tasks are integrated and even add your own model or tasks.

#### [edgeai-modelzoo](edgeai-modelzoo) - for advanced users
* Navigagte to [edgeai-modelzoo](edgeai-modelzoo) to see see example models, their documentation and performance benchmarks.
* Browse to the respositories that were used to train those models and try to train your own model using one of those.


### Deploy your model
* Use [edgeai-benchmark](edgeai-benchmark) or [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) to compile models and create compiled artifacts.
* Run the compiled models using [Edge AI SDK](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md)

<hr>
<hr>


## Components
* The subcomponents have detailed documentation. In the browser, navigate into the sub-folders to see detailed documentation. Here is a high level overview.

| Category                                | ToolLink                                     | Purpose                                          | IS NOT    |
|-----------------------------------------|----------------------------------------------|--------------------------------------------------|-----------|
| Model Zoo / Models collection      | [edgeai-modelzoo](edgeai-modelzoo)           | provides collection of pretrained models, documentation & benchmark information         |           |
|Model compilation & benchmarking     | [edgeai-benchmark](edgeai-benchmark)         | Wrapper on top of [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) for easy model compilation and speed/accuracy benchmarking<br>- Bring your own model and compile, benchmark and generate artifacts for deployment on SDK with camera, inference and display (using edgeai-gst-apps)<br>- Comprehends inference pipeline including dataset loading, pre-processing and post-processing<br>- Benchmarking of accuracy and latency with large data sets<br>- Post training quantization<br>- Docker for easy development environment setup |  |
|Model training tools                 | [edgeai-modeloptimization](edgeai-modeloptimization)    | **Model optimization tools** for improved model training, tools to train TIDL friendly models.<br>- **Model surgery**: Modifies models with minimal loss in accuracy and makes it suitable for TI device (replaces unsupported operators)<br>- **QAT**: **Quantization Aware Training** to improve accuracy with fixed point quantization<br>- Model Pruning/sparsity: Induces sparsity during training – only applicable for specific devices - this is in development.<br> |- Does not support Tensorflow   |
|Model training code    | [**edgeai-torchvision**](edgeai-torchvision)<br>[**edgeai-mmdetection**](edgeai-mmdetection)<br>[edgeai-mmdetection3d](edgeai-mmdetection3d)<br>[edgeai-hf-transformers](edgeai-hf-transformers)<br>[edgeai-mmpose](edgeai-mmpose)<br>[edgeai-tensorvision](edgeai-tensorvision) | Training repositories for various tasks<br>- Provides extensions of popular training repositories (like mmdetection, torchvision) with lite version of models |- Does not support Tensorflow |
|**End-to-end Model development - Datasets, Training & Compilation**   | [**edgeai-modelmaker**](edgeai-modelmaker)       | **Beginner friendly**, command line, integrated environment for training & compilation<br>- Bring your own data, select a model, perform training and generate artifacts for deployment on SDK<br>- Backend tool for model composer (early availability of features compared to Model Composer ) |- Does not support Bring Your Own Model workflow |
|Example datasets, used in edgeai-modelmaker         | [edgeai-datasets](edgeai-datasets)           | Example datasets   |          |

<hr>

### Deprecations
| Category                                | ToolLink                                     | Purpose                                          | IS NOT    |
|-----------------------------------------|----------------------------------------------|--------------------------------------------------|-----------|
| Model training code     | [edgeai-yolox](edgeai-yolox) is being deprecated - use [edgeai-mmpose](edgeai-mmpose) for Keypoint detection and [edgeai-mmdetection](edgeai-mmdetection) for Object Detection |  |  |


<hr>
<hr>


## Tech Reports

Technical documentation can be found in the documentation of each repository. Here we have a collection of technical reports & tutorials that give high level overview on various topics - see [**Edge AI Tech Reports**](./docs/tech_reports/README.md).

<hr>
<hr>

## Acknowledgements
This umbrella repository uses and modifies several source repositories. The following table can be used to navigate to the source of the original repositories and see the contents & contributors.

| Sub-repository/Sub-directory     | Original source repository   |
|----------------------------------|------------------------------| 
|edgeai-hf-transformers           | [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers) |
|edgeai-mmdeploy                   | [https://github.com/open-mmlab/mmdeploy](https://github.com/open-mmlab/mmdeploy) |
|edgeai-mmdetection                | [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) |
|edgeai-mmdetection3d              | [https://github.com/open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d) |
|edgeai-mmpose                     | [https://github.com/open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) |
|edgeai-torchvision                | [https://github.com/pytorch/vision](https://github.com/pytorch/vision) |
|edgeai-yolox                      | [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) |
|edgeai-benchmark                  |  NA  |
|edgeai-modelzoo                   |  NA  |
|edgeai-modelmaker                 |  NA  |
|edgeai-modeloptimization          |  NA  |
|edgeai-tensorvision               |  NA  |

<hr>
<hr>
