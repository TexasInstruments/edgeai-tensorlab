echo "mmdetection is not installing cython properly - it has to be installed from conda if this is conda python"
conda install cython

echo "Installing mmcv - Please check your CUDA and PyTorch versions and modify this link appropriately."
echo "More info is available in https://github.com/open-mmlab/mmcv and https://github.com/open-mmlab/mmdetection"
pip install mmcv-full==latest+torch1.6.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

echo "Installing mmdetection"
pip install git+https://github.com/open-mmlab/mmdetection.git

echo "Please visit https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/"
echo "and https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/ to clone and install that package."

