# edgeai-tensorlab 
Edge AI model training, quantization, compilation/benchmark & Model Zoo

<hr>

## Release 10.0 (10_00)
- Git branch: **r10.0**
### Updates
#### Date: 18 October 2024
- git tag: v10.0.2
- tidl_tools update - bug fixes: edgeai-benchmark setup_pc.sh has been updated to install tidl_tools version **10_00_08_00**. See [release notes of edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools/releases) to understand the bug fixes that it has.
#### Date: September 2024
- git tag: v10.0.1
- Tech Reports added(**Object detection, PyTorch FX based Quantization**) - see this [link](./docs/tech_reports/README.md)
- edgeai-hf-transformers (fork of https://github.com/huggingface/transformers) repository added, enabling training of Transformer based Image Classification, Segmentation and Object Detection Models. 
- Improvements for Model Surgery and Quantization in edgeai-modeloptimization
- Several additional models added in edgeai-modelzoo and edgeai-benchmark
- edgeai-mmpose(fork of open-mmlab/mmpose for efficient keypoint detection), edgeai-hf-transformers(fork of huggingface/transformers)
- Tools: [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) version used: **10_00_06_00**

More details and information about previous releases are available in [**Release Notes**](./docs/release_notes.md)

**It is important to use the correct git branch if you are going to use the compiled models on a device/SDK.**

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
|Model training & code    | [**edgeai-torchvision**](edgeai-torchvision)<br>[**edgeai-mmdetection**](edgeai-mmdetection)<br>[edgeai-mmdetection3d](edgeai-mmdetection3d)<br>[edgeai-hf-transformers](edgeai-hf-transformers)<br>[edgeai-mmpose](edgeai-mmpose)<br>[edgeai-yolox](edgeai-yolox) | Training repositories for various tasks<br>- Provides extensions of popular training repositories (like mmdetection, torchvision) with lite version of models |- Does not support Tensorflow |
|**End-to-end Model development - Datasets, Training & Compilation**   | [**edgeai-modelmaker**](edgeai-modelmaker)       | **Beginner friendly**, command line, integrated environment for training & compilation<br>- Bring your own data, select a model, perform training and generate artifacts for deployment on SDK<br>- Backend tool for model composer (early availability of features compared to Model Composer ) |- Does not support Bring Your Own Model workflow |
|Example datasets, used in edgeai-modelmaker         | [edgeai-datasets](edgeai-datasets)           | Example datasets   |          |

<hr>
<hr>

## Tech Reports

Technical documentation can be found in the documentation of each repository. Here we have a collection of technical reports & tutorials that give high level overview on various topics - see [**Edge AI Tech Reports**](./docs/tech_reports/README.md).

<hr>
