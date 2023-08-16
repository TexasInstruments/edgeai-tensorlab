#!/usr/bin/env bash

# Dependencies for building pillow-simd, onnx
echo 'Installing dependencies to build pillow-simd'
echo 'If you do not have sudo access, comment the below line and replace pillow-simd with pillow in the requirements file'
sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
pip3 install --no-input --upgrade pip setuptools

######################################################################
echo 'Installing python packages...'
# there as issue with installing pillow-simd through requirements - force it here
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd

echo "installing requirements"
pip3 install --no-input -r requirements.txt

# building onnx from soure requires carefull steps
# make sure that we are using system cmake
pip uninstall --yes cmake
# pybind11[global] is needed for building the onnx package.
# for some reason, this has to be installed before the requirements file is used.
pip3 install --no-input pybind11[global] protobuf==3.19.4
pybind11_DIR=$(pybind11-config --cmakedir) pip3 install --no-input https://github.com/TexasInstruments/onnx/archive/tidl-j7.zip

######################################################################
# may need pytorch nightly to build this package
echo "installing pytorch - use the applopriate index-url from https://pytorch.org/get-started/locally/"
pip3 install --no-input torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# no need to do this install this repo - we can use off-the-shelf torchvision installed by pip
# but if you want to install from this repo (may involve CUDA/C++ compilation), uncomment the following
#python3 setup.py develop
