echo "mmdetection is not installing cython properly - it has to be installed from conda if this is conda python"
conda install -y cython

echo "Installing pytorch and torchvision"
pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing mmcv - Please check your CUDA and PyTorch versions and modify this link appropriately."
echo "More info is available in https://github.com/open-mmlab/mmcv and https://github.com/open-mmlab/mmdetection"
pip3 install openmim
mim install mmdet==2.8.0
mim install mmcv-full==1.2.7

pip3 install onnxruntime

echo "This package depends on edgeai-torchvision:"
echo "Please visit https://github.com/TexasInstruments/edgeai-torchvision and clone and install that package."
