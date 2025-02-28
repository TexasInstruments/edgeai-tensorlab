# edgeai-tensorlab 
Edge AI model training, quantization, compilation/benchmark & Model Zoo

<hr>

## Release 10.1
* Release name: **10.1**
* Git branch: **r10.1**
* tidl_tools version: **10_01_04_01**

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

### New models in edgeai-modelmaker
| Model name                    | Model Type                            | Source repository      |
|-------------------------------|---------------------------------------|------------------------|
| YOLOv7       | Object Detection                      | edgeai-mmdetection     |
| YOLOv9       | Object Detection                      | edgeai-mmdetection     |
| YOLOXPose                     | Keypoint detection / Human Pose       | edgeai-mmpose          |

**More details are in the [Release Notes](./docs/release_notes.md)**

### Tech reports
New Tech report on 3D Object Detection - see the section on "Tech Reports"

<hr>
<hr>

## Notice
Our documentation landing pages are the following:
- https://www.ti.com/edgeai : Technology page summarizing TI’s edge AI software/hardware products 
- **https://github.com/TexasInstruments/edgeai** : Landing page for developers to understand overall software and tools offering. Read it before navigating into this repository.

<hr>
<hr>

## Components
* The subcomponents have detailed documentation. In the browser, navigate into the sub-folders to see detailed documentation. Here is a high level overview.

| Category                                | ToolLink                                     | Purpose                                          | IS NOT    |
|-----------------------------------------|----------------------------------------------|--------------------------------------------------|-----------|
| Model Zoo / Models collection      | [edgeai-modelzoo](edgeai-modelzoo)           | provides collection of pretrained models         |           |
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
This umbrella repository is created using [git subrepo](https://github.com/ingydotnet/git-subrepo) - it doesn't preserve the intermediate git commits of individual sub-repositories/sub-directories here. The following table can be used to navigate to the source of the original repositories and see the contributors.

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
