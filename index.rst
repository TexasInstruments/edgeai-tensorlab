.. Jacinto AI ModelZoo master file

--------------------------------------------------------------------------
ModelZoo - Deep Neural Network (DNN) Models & Artifacts for Jacinto 7 SoCs
--------------------------------------------------------------------------


Pre-Imported Model Artifacts
----------------------------

Several Pre-imported model artifacts are provided in this package. These can be used for running real-time inference on device.

.. toctree::
    :maxdepth: 3

    docs/artifacts


Using the Pre-Imported Model Artifacts
--------------------------------------

Inference on device can be run using one of the options:

- EdgeAI-Devkit (documentation link to be added here).
- `EVM Cloud/Farm for evaluating EdgeAI <https://dev.ti.com/edgeai/>`_
- TI Deep Learning Library (TIDL) in `PROCESSOR-SDK-RTOS-J721E <https://www.ti.com/tool/PROCESSOR-SDK-J721E>`_. Please consult the `latest PROCESSOR-SDK-RTOS-J721E documentation <https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/latest/exports/docs/psdk_rtos/docs/user_guide/index.html>`_ for for more information.


Jacinto-AI-ModelZoo Repository
------------------------------

We have a Models Collection Git repository containing several exported models (onnx, tflite, mxnet) that TIDL can import. We also have links to various popular training repositories. Some models were trained by us - they are good examples of how to select models that perform well on our device. Sometimes model export is not straight forward, as some models can have operators that TIDL do not support. When doing inference on device using one of the open soure front ends of TIDL, those operators will run on ARM (instead of being offloaded to C7x/MMA), slowing down the inference. Wherever necessary, we have more information about how to modify models during training or export to maximize performance.

We also provide a benchmark repository that provides dataset loaders, pre-processing, post processing & metric computations to easily to import, inference and accuracy benchmark of these models.

There is also an example of how custom models can be imported easily to generate model artifacts that can be used for inference on device.

.. toctree::
    :maxdepth: 3

    Models Collection <README.md>


Jacinto-AI-Benchmark Repository
-------------------------------

We also provide a Benchmark Git repository with source code that provides dataset loaders, pre-processing, post processing & metric computations to easily to import, inference and accuracy benchmark of these models.

There is also an example of how custom models can be imported easily to generate model artifacts that can be used for inference on device.

.. toctree::
    :maxdepth: 3

    Models Benchmark <./dependencies/jacinto-ai-benchmark/README.md>


Training Repositories
---------------------

.. toctree::
    :maxdepth: 3

    PyTorch-Jacinto-AI-DevKit for Training Classification, Segmentation & Multi-Task models <https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/>
    Quantization Aware Training (QAT) using PyTorch-Jacinto-AI-DevKit <https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/docs/Quantization.md>
    Object Detection Training using PyTorch-MMDetection <https://git.ti.com/cgit/jacinto-ai/pytorch-mmdetection/about/>


References
----------

.. toctree::
    :maxdepth: 3

    docs/references
