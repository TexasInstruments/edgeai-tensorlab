#!/usr/bin/env bash

# system packages
sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
pip3 install --no-input --upgrade pip==23.3.1 setuptools==69.0.2

######################################################################
echo "installing pytorch - use the applopriate index-url from https://pytorch.org/get-started/locally/"
pip3 install --no-input torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo 'Installing python packages...'
# there as issue with installing pillow-simd through requirements - force it here
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd
pip3 install --no-input cython wheel numpy==1.23.0
pip3 install --no-input torchinfo pycocotools opencv-python

echo "installing requirements"
pip3 install --no-input -r requirements.txt

######################################################################
# building onnx from soure requires carefull steps
# make sure that we are using system cmake
pip uninstall --yes cmake
# pybind11[global] is needed for building the onnx package.
# for some reason, this has to be installed before the requirements file is used.
pip3 install --no-input pybind11[global] protobuf==3.19.4
pybind11_DIR=$(pybind11-config --cmakedir) pip3 install --no-input https://github.com/TexasInstruments/onnx/archive/tidl-j7.zip

######################################################################
# no need to do this build/install torchvision from this repo - we can use off-the-shelf torchvision installed above along with torch install
# but if you want to install from this repo (may involve C++ compilation), uncomment the following
#python3 setup.py develop

######################################################################
# setup the edgeai_xvision package, which is inside references/edgeailite
pip3 install --no-input -r ./references/edgeailite/requirements.txt
pip3 install -e ./references/edgeailite/


