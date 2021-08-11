echo "mmdetection is not installing cython properly - it has to be installed from conda if this is conda python"
conda install -y cython

echo "Installing pytorch and torchvision"
#pip install torch==1.6.0 torchvision==0.7.0
pip3 install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing mmcv - Please check your CUDA and PyTorch versions and modify this link appropriately."
echo "More info is available in https://github.com/open-mmlab/mmcv and https://github.com/open-mmlab/mmdetection"
pip install --force-reinstall mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

echo "Installing mmdetection"
pip install git+https://github.com/open-mmlab/mmdetection.git@v2.8.0

pip install onnxruntime
pip install onnxoptimizer

echo "This package depdends on edgeai-torchvision:"
echo "Please visit https://github.com/TexasInstruments/edgeai-torchvision and clone and install that package."
