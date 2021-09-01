echo "mmdetection is not installing cython properly - it has to be installed from conda if this is conda python"
conda install -y cython

echo "Installing pytorch and torchvision"
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing mmcv - Please check your CUDA and PyTorch versions and modify this link appropriately."
echo "More info is available in https://github.com/open-mmlab/mmcv and https://github.com/open-mmlab/mmdetection"
pip3 install --force-reinstall mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

echo "Installing mmdetection"
pip3 install git+https://github.com/open-mmlab/mmdetection.git@v2.8.0

pip3 install onnxruntime
pip3 install onnxoptimizer

echo "This package depdends on edgeai-torchvision:"
echo "Please visit https://github.com/TexasInstruments/edgeai-torchvision and clone and install that package."
