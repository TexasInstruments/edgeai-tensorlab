echo "Installing mmcv - Please check your CUDA and PyTorch versions and modify this link apprrpriately. More info is available in https://github.com/open-mmlab/mmcv"
pip install mmcv-full==latest+torch1.5.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

echo "Installing mmdetection"
pip install git+https://github.com/open-mmlab/mmdetection.git

echo "Please visit https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/about/ and https://git.ti.com/cgit/jacinto-ai/pytorch-jacinto-ai-devkit/ to clone and install that package."

