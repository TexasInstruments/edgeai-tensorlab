# EdgeAI-MMDetection3D

This repository is an extension of the popular [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) open source repository for 3d object detection. While mmdetection3d focuses on a wide variety of models, typically at high complexity, we focus on models that are optimized for speed and accuracy so that they run efficiently on embedded devices. For this purpose, we have added a set of embedded friendly model configurations and scripts.

This repository also supports Quantization Aware Training (QAT).

<hr>


## Notes
- See notes about recent changes/updates in this repository in [release notes](./docs/det3d_release_notes.md)
- The original documentation of mmdetection3d is at the bottom of this page.


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
pyenv virtualenv 3.10 mmdet3d
pyenv activate mmdet3d
pip install --upgrade pip setuptools
```

Note: Prior to SDK/TIDL version 9.0, the Python version required was 3.6

Activation of Python environment - this activation step needs to be done everytime one starts a new terminal or shell. (Alternately, this also can be written to the .bashrc, so that this will be the default penv environment).
```
pyenv activate mmdet3d
```


## Installation Instructions
After cloning this repository, install it as a Python package by running:
```
./setup.sh
```

## PointPillars

Data preparation, training and testing of PointPillars can be found in [PointPillars](./projects/PointPillars/README.md).
