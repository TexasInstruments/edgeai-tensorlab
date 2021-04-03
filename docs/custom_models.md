# Import / Benchmark custom models

Importing a DNN model using TIDL (or one of its open source front ends) is the process of quantizing and converting the model into a format that can be offloaded into c7x/MMA in. The imported artifacts can then be used to run inference.

Out benchmark tools provide utilities for importing and accuracy benchmark of models. These tools include dataset loaders, pre-processing utilities, post-processing utilities and metric computation utilities.

The following example shows how to perform accuracy benchmark of custom models.

[scripts/benchmark_custom.py](../scripts/benchmark_custom.py)

During this accuracy benchmark, import of the model will be performed as the first step. The imported artifacts can be used to run inference on the target device (eg. EVM). 

However, there are a few things to note:

All the step including import, inference and accuracy benchmark can be run on PC (Simulation).

However, the import step cannot be run on the target device (eg. EVM). So that import step has to be run on PC and the resulting artifacts has to be coped to the target device to perform inference or accuracy benchmark.

