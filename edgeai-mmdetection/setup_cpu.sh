#!/usr/bin/env bash

# system packages
sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
pip3 install --no-input --upgrade pip setuptools

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
echo "Installing mmcv"
FORCE_CUDA=0 pip3 install --no-input mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.1/index.html

######################################################################
# can we move this inside the requirements file is used.
pip3 install --no-input protobuf==3.20.2 onnx==1.13.0

# error when moving this inside requirements file
pip3 install --no-input git+https://github.com/TexasInstruments/edgeai-modeloptimization.git@r9.1#subdirectory=torchmodelopt

######################################################################
echo "Installing mmdetection"
echo "For more details, see: https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md and https://github.com/open-mmlab/mmdetection"
FORCE_CUDA=0 python3 setup.py develop

######################################################################
# apply patch to support latest torch with older mmcv
python3 ./tools/deployment/torch_onnx_patch/torch_onnx_patch.py

######################################################################
echo "This package depends on edegeai_xvision that is installed from edgeai-torchvision. Please clone and run setup_cpu.sh from:"
echo "URL: https://github.com/TexasInstruments/edgeai-torchvision"

