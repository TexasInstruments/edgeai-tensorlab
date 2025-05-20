#!/usr/bin/env bash

# Copyright (c) 2018-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################################

# system packages
sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
pip3 install --no-input --upgrade pip==23.0.1 setuptools==60.2.0

######################################################################
echo "installing torch, torchvision"
pip3 install torch==2.4.0 torchvision==0.19.0

echo "installing requirements"
pip3 install --no-input -r requirements.txt

echo "Installing PandaSet"
rm -r -f pandaset-devkit
git clone https://github.com/scaleapi/pandaset-devkit.git
cp -f ./requirements/pandaset_requirements.txt ./pandaset-devkit/python/requirements.txt
cd pandaset-devkit/python
pip install .
cd ../..
rm -r -f pandaset-devkit

pip install -U openmim
mim install mmengine==0.10.4
mim install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
mim install mmdet==3.3.0
mim install mmsegmentation==1.2.2


######################################################################
# can we move this inside the requirements
pip3 install --no-input protobuf==3.20.2 onnx==1.14.0

# error when moving this inside requirements file
pip3 install --no-input git+https://github.com/TexasInstruments/edgeai-modeloptimization.git@r10.1#subdirectory=torchmodelopt
# pip install -e ../edgeai-modeloptimization/torchmodelopt

######################################################################
echo "Installing mmdetection"
echo "For more details, see: https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md and https://github.com/open-mmlab/mmdetection"
pip install -v -e .

######################################################################
# apply patch to support latest torch with older mmcv
python3 ./tools/deployment/torch_onnx_patch/torch_onnx_patch.py


