#!/usr/bin/env bash

# system packages
sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
pip3 install --no-input --upgrade pip setuptools

######################################################################
echo "installing pytorch - use the applopriate index-url from https://pytorch.org/get-started/locally/"
pip3 install --no-input torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

echo 'Installing python packages...'
# there as issue with installing pillow-simd through requirements - force it here
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd
pip3 install --no-input cython wheel numpy==1.23.0
pip3 install --no-input torchinfo pycocotools opencv-python

echo "installing requirements"
pip3 install --no-input -r requirements.txt

######################################################################
echo "Installing mmcv"
pip3 install --no-input mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html

######################################################################
# building onnx from soure requires carefull steps
# make sure that we are using system cmake
pip uninstall --yes cmake
# pybind11[global] is needed for building the onnx package.
# for some reason, this has to be installed before the requirements file is used.
pip3 install --no-input pybind11[global] protobuf==3.19.4
pybind11_DIR=$(pybind11-config --cmakedir) pip3 install --no-input https://github.com/TexasInstruments/onnx/archive/tidl-j7.zip

######################################################################
echo "Installing mmdetection"
echo "For more details, see: https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md and https://github.com/open-mmlab/mmdetection"
python3 setup.py develop

######################################################################
echo "This package depends on edegeai_xvision that is installed from edgeai-torchvision. Please clone and install that package."
echo "URL: https://github.com/TexasInstruments/edgeai-torchvision"

