#echo "mmdetection is not installing cython properly - it has to be installed from conda if this is conda python"
#conda install -y cython

echo "Installing pytorch and torchvision"
pip3 install --no-input torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing mmcv"
pip3 install --no-input mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

echo "Installing mmdetection"
echo "For more details, see: https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md and https://github.com/open-mmlab/mmdetection"
python3 setup.py develop

pip3 install --no-input onnx==1.8.1
pip3 install --no-input torchinfo

echo "This package depends on edgeai-torchvision:"
echo "Please visit https://github.com/TexasInstruments/edgeai-torchvision and clone and install that package."
