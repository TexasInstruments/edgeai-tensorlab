#!/usr/bin/env bash

# Dependencies for building pillow-simd
echo 'Installing dependencies to build pillow-simd'
echo 'If you do not have sudo access, comment the below line and replace pillow-simd with pillow in the requirements file'
sudo apt-get install -y libjpeg-dev zlib1g-dev

echo "-----------------------------------------------------------"
echo "installing requirements"
pip3 install --no-input -r requirements.txt

# there as issue with installing pillow-simd through requirements - force it here
# 7.2.0.post1 is what works in Python3.6 - newer Python versions may be able to use a more recent one
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd==7.2.0.post1

echo "-----------------------------------------------------------"
echo "building torchvision"
# may need pytorch nightly to build this package
echo "installing pytorch for cuda 11.3"
echo "other versions can be found here: https://pytorch.org/get-started/locally/"
pip3 install --no-input torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install --no-input onnx==1.8.1
pip3 install --no-input torchinfo

echo "if your CUDA version has changed, please remove 'build' folder before attempting this installatio."
echo "otherwise there can be errors while using torchvision c++ ops."

# an editable install - changes in this local torchvision module immediately takes effect
echo "installing torchvision in develop mode"
python3 setup.py develop

# final install - the torchvision module is copied to the python folder
#python3 setup.py install

echo "-----------------------------------------------------------"
echo "copying the .so files to make running from this folder work"
BUILD_LIB_PATH=$(find "build/" -maxdepth 1 |grep "/lib.")
cp -f -r ${BUILD_LIB_PATH}/torchvision/*.so ./torchvision/
