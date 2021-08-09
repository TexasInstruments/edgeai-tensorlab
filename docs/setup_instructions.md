## Requirements

PROCESSOR-SDK-RTOS for Jacinto 7 provides TI Deep Learning Library (TIDL), which is an optimized library that can run DNNs on our SoCs. TIDL provides several familiar interfaces for model inference - such as onnxruntime, tflite_runtime, tvm/dlr - apart from its own native interface. All these runtimes that are provided as part of TIDL can offload model execution into our high performance C7x+MMA DSP. For more information how to obtain and use these runtimes, please visit the TIDL documentation in the RTOS SDK. This software depends on TIDL.

#### Environment
We have tested this on Ubuntu 18.04 PC with Anaconda Python 3.6. This is the recommended environment. Create a Python 3.6 environment if you don't have it and activate it.


#### Requirement: ModelZoo
DNN Models and pre-compied model artifacts are provided in another repository called **[EdgeAI-ModelZoo](https://github.com/TexasInstruments/edgeai-modelzoo)**. 

Please clone that repository. After cloning, edgeai-benchmark and edgeai-modelzoo must be inside the same parent folder for the default settings to work.


#### Requirement: PROCESSOR-SDK-RTOS-J721E
PROCESSOR-SDK-RTOS for Jacinto 7 is required to run this package. Please visit the links given at https://github.com/TexasInstruments/edgeai to download and untar/extract the PROCESSOR-SDK-RTOS on your Ubuntu desktop machine.

After extracting, follow the instructions in the RTOS package to download and install the dependencies required for it. The following steps are required:<br>

(1) Install PROCESSOR-SDK-RTOS dependencies - especially graphviz and gcc-arm: Change directory to **psdk_rtos/scripts** inside the extracted SDK and run:

```setup_psdk_rtos.sh```

(2) In the extracted SDK, change directory to tidl folder (it has the form tidl_j7_xx_xx_xx_xx). Inside the tidl folder, change directory to **ti_dl/test/tvm-dlr** and run:

```source prepare_model_compliation_env.sh```
 
 to install TVM Deep Learning compiler, DLR Deep Learning Runtime and their dependencies. In our SDK, we have support to use TVM+DLR to offload part of the graph into the underlying TIDL backend running on the C7x+MMA DSP, while keeping the unsupported layers running on the main ARM processor. <br>

(3) Inside the tidl folder, change directory to **ti_dl/test/tflrt** and run:
 
 ```source prepare_model_compliation_env.sh``` 
 
 to install TI's fork of TFLite Runtime and its dependencies. In our SDK, we have support to use TFLite Runtime to offload part of the graph into the underlying TIDL backend running on the C7x+MMA DSP, while keeping the unsupported layers running on the main ARM processor.<br>

(4) Inside the tidl folder, change directory to **ti_dl/test/onnxrt** and run:
 
 ```source prepare_model_compliation_env.sh``` 
 
 to install TI's fork of ONNX Runtime and its dependencies. In our SDK, we have support to use ONNX Runtime to offload part of the graph into the underlying TIDL backend running on the C7x+MMA DSP, while keeping the unsupported layers running on the main ARM processor.<br>

(5) Finally, create a symbolic link to the SDK inside this code at the path **./dependencies/ti-processor-sdk-rtos** so that this code can find it - for example:

```
ln -sf ../ti-processor-sdk-rtos-j721e-evm-07_03_00_07 ./dependencies/ti-processor-sdk-rtos
```

Where *../ti-processor-sdk-rtos-j721e-evm-07_03_00_07* is just an example - **replace it** with the path where the SDK has been extracted.


#### Requirement: Datasets
This benchmark code can use several datasets. In fact, the design of this code is flexible to add support for additional datasets easily.

We already have support to download several of these datasets automatically - but this may not always work because the source URLs may change. For example the ImageNet download URL has changed recently and the automatic download no longer works. 

If you start the download and interrupt it in between, the datasets may be partially downloaded and it can lead to unexpected failures. If the download of a dataset is interrupted in between, delete that dataset folder manually to start over. 

Also, the download may take several hours even with a good internet connection. 

Because of all these reasons **some datasets may need to be manually downloaded (especially ImageNet).** To make the datasets manually available, they should be placed at the locations specified for each dataset inside the folder **./dependencies/datasets/** - if you have the datasets stored somewhere else, create symbolic links as necessary.

The following link explains how to **[Obtain Datasets](./docs/datasets.md)** for benchmarking.


## Installation Instructions
After cloning this repository, install it as a Python package by running:
```
./setup.sh
```

Open the shell scripts that starts the actual benchmarking run_benchmarks.sh and see the environment variables **PSDK_BASE_PATH** and **TIDL_TOOLS_PATH** being defined. Change these paths appropriately to reflect what is in your PC.

Once installed, the **jai_benchmark** will be a available as a package in your Python environment. It can be imported just like any other Python package in a Python script:<br>
```
import jai_benchmark
```
or
```
from jai_benchmark import *
```
