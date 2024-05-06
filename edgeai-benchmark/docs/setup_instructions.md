## Environment
We have tested this on Ubuntu 18.04 PC with Miniconda from https://docs.conda.io/en/latest/ with Python 3.6. Create a Python 3.6 environment if you don't have it and activate it before following the rest of the instructions.
```
conda create -n benchmark python=3.6
conda activate benchmark
```


## Installation Instructions
After cloning this repository, install it as a Python package by running:
```
./setup.sh
```
In the [requirements_pc.txt](../requirements_pc.txt), we have pillow-simd (as it is faster), but you can replace it with pillow if pillow-simd installation fails.

This setup script downloads several packages from **[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools]** and installs them. These packages are used for model compilation and also inference on PC (simulation).

It also installs the module **jai_benchmark** from this repository. Once installed, the **jai_benchmark** will be available as a module in the Python environment. It can be imported just like any other Python module:<br>
```
import jai_benchmark
```
or
```
from jai_benchmark import *
```


## Additional Requirement: ModelZoo
DNN Models and pre-compied model artifacts are provided in another repository called **[EdgeAI-ModelZoo](https://github.com/TexasInstruments/edgeai-modelzoo)**. 

DNN models are in the folder edgeai-modelzoo/models and pre-compiled artifacts are in the folder edgeai-modelzoo/modelartifacts/8bits. 

Please clone that repository. After cloning, edgeai-benchmark and edgeai-modelzoo must be inside the same parent folder for the default settings to work.


## Additional Requirement: Datasets
This benchmark code can use several datasets. In fact, the design of this code is flexible to add support for additional datasets easily.

We already have support to download several of these datasets automatically - but this may not always work because the source URLs may change. For example the ImageNet download URL has changed recently and the automatic download no longer works. 

If you start the download and interrupt it in between, the datasets may be partially downloaded and it can lead to unexpected failures. If the download of a dataset is interrupted in between, delete that dataset folder manually to start over. 

Also, the download may take several hours even with a good internet connection. 

Because of all these reasons **some datasets may need to be manually downloaded (especially ImageNet).** To make the datasets manually available, they should be placed at the locations specified for each dataset inside the folder **./dependencies/datasets/** - if you have the datasets stored somewhere else, create symbolic links as necessary.

The following link explains how to **[Obtain Datasets](./datasets.md)** for benchmarking.
