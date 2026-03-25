.. EdgeAI ModelZoo master file

***************************************************************************
ModelZoo - Deep Neural Network (DNN) Models & Artifacts for Jacinto 7 SoCs
***************************************************************************

Pre-Trained Models
===================

Models Collection containing several exported models (onnx, tflite, mxnet) that TIDL can import.

The models support a variety of tasks such as Image Classification, Semantic Segmentation, Object Detection and Keypoint Detection/Human Pose Estimation. We also provide links to the corresponding training repositories so that the user can go and finetune these models as necessary.

Some models were trained by us - they are good examples on how to design or select models that perform well on our device.

.. toctree::
    :maxdepth: 2

    Models Collection <README.md>


Pre-Imported Model Artifacts
=============================

Several pre-imported model artifacts are provided in this package. These can be used for running inference on device.

.. toctree::
    :maxdepth: 2

    docs/modelartifacts


Using the Pre-Imported Model Artifacts
--------------------------------------

Inference on device can be run using one of the options: (1) EdgeAI-DevKit Software Development Kit (to be announced). (2) `EVM Cloud/Farm for evaluating EdgeAI <https://dev.ti.com/edgeai/>`_ (3) TI Deep Learning Library (TIDL) in `PROCESSOR-SDK-RTOS-J721E <https://www.ti.com/tool/PROCESSOR-SDK-J721E>`_. Please consult the `latest PROCESSOR-SDK-RTOS-J721E documentation <https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/latest/exports/docs/psdk_rtos/docs/user_guide/index.html>`_ for for more information.


EdgeAI-Benchmark Repository
===========================

We also provide a Benchmark Git repository with source code that provides dataset loaders, pre-processing, post processing & metric computations to easily to import, inference and accuracy benchmark of these models.

There is also an example of how custom models can be imported easily to generate model artifacts that can be used for inference on device.

.. toctree::
    :maxdepth: 2

    Models Benchmark <./dependencies/edgeai-benchmark/README.md>


Training Repositories
=====================

.. toctree::
    :maxdepth: 2

    EdgeAI-Torchvision for Training Classification, Segmentation & Multi-Task models <https://github.com/TexasInstruments/edgeai-torchvision>
    Quantization Aware Training (QAT) using EdgeAI-Torchvision <https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Quantization.md>
    Object Detection Training using PyTorch-MMDetection <https://github.com/TexasInstruments/edgeai-mmdetection>


References
==========

.. toctree::
    :maxdepth: 2

    docs/references
