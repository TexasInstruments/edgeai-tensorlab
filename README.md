# edgeai-tensorlab 
Edge AI model training, quantization, compilation/benchmark & Model Zoo

## Notice
Our documentation landing pages are the following:
- https://www.ti.com/edgeai : Technology page summarizing TI’s edge AI software/hardware products 
- https://github.com/TexasInstruments/edgeai : Landing page for developers to understand overall software and tools offering. Read this before navigating into this repository.
- **Our repositories have been restructured** : Several repositories are now packaged as components inside this repository.

<hr>

## Release Notes
Please see the **[Release Notes](docs/release_notes.md)** to understand what is new.

<hr>

## Components
* The subcomponents have detailed documentation. In the browser, navigate into the sub-folders to see detailed documentation. Here is a high level overview.

| Category                                                | Tool/Link                                                                                                                                                                                                              | Purpose           | IS NOT    |
|---------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------|
| **Model training & associated tools**                   | [edgeai-modelzoo](edgeai-modelzoo)  provides collection of pretrained models                                                                                                                                           |                  |
|ditto                                                         | [Model optimization tools](edgeai-modeloptimization)                                                                                                                                                                   | **Model optimization tools**<br>- **Model surgery**: Modifies models with minimal loss in accuracy and makes it suitable for TI device (replaces unsupported operators)<br>- **Model Pruning/sparsity**: Induces sparsity during training – only applicable for specific devices<br>- **QAT**: **Quantization Aware Training** to improve accuracy with fixed point quantization<br>                    |- Does not support Tensorflow   |
|ditto                                                         | [edgeai-benchmark](edgeai-benchmark)                                                                                                                                                                                   | Bring your own model and compile, benchmark and generate artifacts for deployment on SDK with camera, inference and display (using edgeai-gst-apps)<br>- Comprehends inference pipeline including dataset loading, pre-processing and post-processing<br>- Benchmarking of accuracy and latency with large data sets<br>- Post training quantization<br>- Docker for easy development environment setup |  |
|ditto                                                         | [edgeai-torchvision](edgeai-torchvision)<br>[edgeai-mmdetection](edgeai-mmdetection)<br>[edgeai-yolox](edgeai-yolox)<br>[edgeai-mmdetection3d](edgeai-mmdetection3d)<br>[edgeai-hf-transformers](edgeai-hf-transformers) | Training repositories for various tasks<br>- Provides extensions of popular training repositories (like mmdetection, yolox) with lite version of models      |- Does not support Tensorflow |
|ditto                                                         | [edgeai-datasets](edgeai-datasets)                                                                                                                                                                                     | Example datasets                |  |
|ditto                                                         | [Model Maker](edgeai-modelmaker)                                                                                                                                                                                       | Command line Integrated environment for training & compilation<br>- Bring your own data, select a model, perform training and generate artifacts for deployment on SDK<br>- Backend tool for model composer (early availability of features compared to Model Composer )    |- Does not support Bring Your Own Model workflow |


<hr>
