## Environment
We have tested this on Ubuntu 22.04 OS and pyenv Python environment manager. Here are the setup instructions.

Make sure that you are using bash shell. If it is not bash shell, change it to bash. Verify it by typing:
```
echo ${SHELL}
```

Install system packages
```
sudo apt update
sudo apt install build-essential curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev wget xz-utils zlib1g-dev
```

Install pyenv using the following command.
```
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

echo '# pyenv settings ' >> ${HOME}/.bashrc
echo 'command -v pyenv >/dev/null || export PATH=":${HOME}/.pyenv/bin:$PATH"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME}/.bashrc
echo '' >> ${HOME}/.bashrc

exec ${SHELL}
```

From SDK/TIDL version 9.0, the Python version required is 3.10. Create a Python 3.10 environment if you don't have it and activate it before following the rest of the instructions.
```
pyenv install 3.10
pyenv virtualenv 3.10 benchmark
pyenv activate benchmark
pip install --upgrade pip setuptools
```

Note: Prior to SDK/TIDL version 9.0, the Python version required was 3.6

Activation of Python environment - this activation step needs to be done everytime one starts a new terminal or shell. (Alternately, this also can be written to the .bashrc, so that this will be the default penv environment).
```
pyenv activate benchmark
```


## Installation Instructions
After cloning this repository, install it as a Python package by running:
```
./setup_pc.sh
```

In the [requirements_pc.txt](../requirements_pc.txt), we have pillow-simd (as it is faster), but you can replace it with pillow if pillow-simd installation fails.

This setup script downloads several packages from **[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools]** and installs them. These packages are used for model compilation and also inference on PC (simulation).

It also installs the module **edgeai_benchmark** from this repository. Once installed, the **edgeai_benchmark** will be available as a module in the Python environment. It can be imported just like any other Python module:<br>
```
import edgeai_benchmark
```
or
```
from edgeai_benchmark import *
```

## Version compatibility
The setup script setup_pc.sh by default installs the tidl_tools for the latest SDK version. Have a look at the environmnet variable TIDL_TOOLS_RELEASE_NAME defined inside the setup script. There is limitied support to install tidl_tools for previous SDK versions by specifying a version_tag. For example:
```
./setup_pc.sh <version_tag>
```
Please go through the script to understand more details and version_tag to be used for previous releases. However this is for reference rather than for actual usage as the Python requirements for previous releases may not be taken care correctly in this branch. Use the correct git branch for this repository to work correctly for a specific version. For example, git branches such as r9.0, r8.6, r8.5 etc are vailable for this repository. The same is true for several of our other git repositories.


## Additional Requirement: ModelZoo
DNN Models and pre-compied model artifacts are provided in another repository called **[EdgeAI-ModelZoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modelzoo)**. 

DNN models are in the folder edgeai-modelzoo/models and pre-compiled artifacts are in the folder edgeai-modelzoo/modelartifacts/8bits. 

Please clone that repository. After cloning, edgeai-benchmark and edgeai-modelzoo must be inside the same parent folder for the default settings to work.


## Additional Requirement: Datasets
This benchmark code can use several datasets. In fact, the design of this code is flexible to add support for additional datasets easily.

We already have support to download several of these datasets automatically - but this may not always work because the source URLs may change. For example the ImageNet download URL has changed recently and the automatic download no longer works. 

If you start the download and interrupt it in between, the datasets may be partially downloaded and it can lead to unexpected failures. If the download of a dataset is interrupted in between, delete that dataset folder manually to start over. 

Also, the download may take several hours even with a good internet connection. 

Because of all these reasons **some datasets may need to be manually downloaded (especially ImageNet).** To make the datasets manually available, they should be placed at the locations specified for each dataset inside the folder **./dependencies/datasets/** - if you have the datasets stored somewhere else, create symbolic links as necessary.

The following link explains how to **[Obtain Datasets](./datasets.md)** for benchmarking.
