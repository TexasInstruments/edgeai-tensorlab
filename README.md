# edgeai-tensorlab 
Edge AI model training, quantization, benchmark & Model Zoo

## Notice
Our documentation landing pages are the following:
- https://www.ti.com/edgeai : Technology page summarizing TI’s edge AI software/hardware products 
- https://github.com/TexasInstruments/edgeai : Landing page for developers to understand overall software and tools offering. Read this before navigating into this repository.

<hr>

## Components summary
### [edgeai-docs](edgeai-docs)

### [edgeai-modelzoo](edgeai-modelzoo)

### [edgeai-benchmark](edgeai-benchmark)

### [edgeai-torchvision](edgeai-torchvision)

### [edgeai-mmdetection](edgeai-mmdetection)

## [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)

<hr>

## Details of components

The table below provides detailed explanation of each of the tools:

| Category                                                | Tool/Link| Purpose   | IS NOT                |
|---------------------------------------------------------|----------|-----------|-----------------------|
| **Model training & associated tools**                   |[edgeai-modelzoo](edgeai-modelzoo)| **Model Zoo**<br>- To provide collection of pretrained models and documemtation    |      |
|ditto                                                         |[edgeai-torchvision](edgeai-torchvision)<br>[edgeai-mmdetection](edgeai-mmdetection)| Training repositories for various tasks<br>- Provides extensions of popular training repositories (like mmdetection, yolox) with lite version of models  |- Does not support Tensorflow   |
|ditto                                                         |[edgeai-benchmark](edgeai-benchmark)| Bring your own model and compile, benchmark and generate artifacts for deployment on SDK with camera, inference and display (using edgeai-gst-apps)<br>- Comprehends inference pipeline including dataset loading, pre-processing and post-processing<br>- Benchmarking of accuracy and latency with large data sets<br>- Post training quantization<br>- Docker for easy development environment setup  |  |
| **Inference (and compilation) Tools**                   |[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)| To get familiar with model compilation and inference flow<br>- [Post training quantization](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/tidl_fsg_quantization.md)<br>- Benchmark latency with out of box example models (10+)<br>- Compile user / custom model for deployment<br>- Inference of compiled models on X86_PC or TI SOC using file base input and output<br>- Docker for easy development environment setup |- Does not support benchmarking accuracy of models using TIDL with standard datasets, for e.g. - accuracy benchmarking using MS COCO dataset for object detection models. Please refer to edgeai-benchmark for the same.<br>- Does not support Camera, Display and inference based end-to-end pipeline development. Please refer Edge AI SDK for such usage    | 

<hr>
