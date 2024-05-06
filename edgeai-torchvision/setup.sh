#!/usr/bin/env bash

# Dependencies for building pillow-simd
echo 'Installing dependencies to build pillow-simd. If you dont have sudo access, comment the below line and replace pillow-simd with pillow in the requirements file'
sudo apt-get install libjpeg-dev zlib1g-dev

echo "-----------------------------------------------------------"
echo "installing requirements"
pip install -r requirements.txt
pip install -r references/requirements.txt

echo "-----------------------------------------------------------"
echo "building torchvision"
# may need pytorch nightly to build this package
echo "installing pytoch for cuda 11.1"
echo "if your cuda version is different, please change the pip pytorh nightly url"
echo "find the url here: https://pytorch.org/get-started/locally/"
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# an editable install - changes in this local torchvision module immediately takes effect
pip install -e ./

# final install - the torchvision module is copied to the python folder
#python setup.py install

echo "-----------------------------------------------------------"
echo "copying the .so files to make running from this folder work"
BUILD_LIB_PATH=$(find "build/" -maxdepth 1 |grep "/lib.")
cp -f ${BUILD_LIB_PATH}/torchvision/*.so ./torchvision/
