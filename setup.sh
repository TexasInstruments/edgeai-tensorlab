echo "-----------------------------------------------------------"
echo "attempting to install packages required to build pillow-simd"
echo "if you don't have sudo access, uncomment pillow-simd and enable pillow in requirements.txt and references/requirements.txt"
echo "also comment out the below line in that case"
#sudo apt-get install libjpeg-dev zlib1g-dev

echo "-----------------------------------------------------------"
echo "installing requirements"
pip install -r requirements.txt
pip install -r references/requirements.txt

echo "-----------------------------------------------------------"
echo "building torchvision"
# may need pytorch nightly to build this package
echo "installing pytoch-nightly for cuda 10.2."
echo "if your cuda version is different, please change the pip pytorh nightly url"
echo "find the url here: https://pytorch.org/get-started/locally/"
#pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html

# an editable install - changes in this local torchvision module immediately takes effect
pip install -e ./

# final install - the torchvision module is copied to the python folder
#python setup.py install

echo "-----------------------------------------------------------"
echo "copying the .so files to make running from this folder work"
BUILD_LIB_PATH=$(find "build/" -maxdepth 1 |grep "/lib.")
cp -f ${BUILD_LIB_PATH}/torchvision/*.so ./torchvision/
