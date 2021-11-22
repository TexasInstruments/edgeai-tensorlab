#!/usr/bin/env bash

# Dependencies for building pillow-simd
echo 'Installing dependencies to build pillow-simd'
echo 'If you do not have sudo access, comment the below line and replace pillow-simd with pillow in the requirements file'
sudo apt-get install libjpeg-dev zlib1g-dev

echo "-----------------------------------------------------------"
echo "installing requirements"
pip install -r requirements.txt
pip install -r references/requirements.txt

echo "-----------------------------------------------------------"
echo "building torchvision"
# may need pytorch nightly to build this package
echo "installing pytorch for cuda 11.1"
echo "other versions can be found here: https://pytorch.org/get-started/locally/"
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

echo "if your CUDA version has changed, please remove 'build' folder before attempting this installatio."
echo "otherwise there can be errors while using torchvision c++ ops."

# an editable install - changes in this local torchvision module immediately takes effect
echo "installing torchvision in develop mode"
pip install -e ./

# final install - the torchvision module is copied to the python folder
#python setup.py install

echo "-----------------------------------------------------------"
echo "copying the .so files to make running from this folder work"
BUILD_LIB_PATH=$(find "build/" -maxdepth 1 |grep "/lib.")
cp -f ${BUILD_LIB_PATH}/torchvision/*.so ./torchvision/
