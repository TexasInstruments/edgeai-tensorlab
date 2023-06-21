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
# can we move this inside the requirements file is used.
pip3 install --no-input protobuf==3.20.2 onnx==1.13.0

######################################################################
# no need to do this build/install torchvision from this repo - we can use off-the-shelf torchvision installed above along with torch install
# but if you want to install from this repo (may involve C++ compilation), uncomment the following
#python3 setup.py develop

######################################################################
# setup the edgeai_xvision package, which is inside references/edgeailite
pip3 install --no-input -r ./references/edgeailite/requirements.txt
pip3 install -e ./references/edgeailite/


