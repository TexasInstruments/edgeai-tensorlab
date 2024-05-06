#!/usr/bin/env bash

#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

#################################################################################
# internal or external repositories
USE_INTERNAL_REPO=0

# set to 1 to enable additional GPLv3 licensed models
PLUGINS_ENABLE_GPL=0

# set to 1 to enable other extra models
PLUGINS_ENABLE_EXTRA=0

#################################################################################
if [[ ${USE_INTERNAL_REPO} -eq 0 ]]; then
    SOURCE_LOCATION="https://github.com/TexasInstruments/"
    FAST_CLONE_MODELZOO=""
else
    SOURCE_LOCATION="ssh://git@bitbucket.itg.ti.com/edgeai-algo/"
    FAST_CLONE_MODELZOO="--single-branch"
fi
# print
echo "SOURCE_LOCATION="${SOURCE_LOCATION}

#################################################################################
# clone
echo "cloning git repositories. this may take some time..."

if [[ ! -d ../edgeai-benchmark ]]; then git clone --branch r8.6 ${SOURCE_LOCATION}edgeai-benchmark.git ../edgeai-benchmark; fi
if [[ ! -d ../edgeai-mmdetection ]]; then git clone --branch r8.6 ${SOURCE_LOCATION}edgeai-mmdetection.git ../edgeai-mmdetection; fi
if [[ ! -d ../edgeai-torchvision ]]; then git clone --branch r8.6 ${SOURCE_LOCATION}edgeai-torchvision.git ../edgeai-torchvision; fi
if [[ ! -d ../edgeai-modelzoo ]]; then git clone ${FAST_CLONE_MODELZOO} --branch r8.6 ${SOURCE_LOCATION}edgeai-modelzoo.git ../edgeai-modelzoo; fi

if [[ ${PLUGINS_ENABLE_GPL} -ne 0 ]]; then
  if [[ ! -d ../edgeai-yolov5 ]]; then git clone --branch r8.4 ${SOURCE_LOCATION}edgeai-yolov5.git ../edgeai-yolov5; fi
  sed -i s/'PLUGINS_ENABLE_GPL = False'/'PLUGINS_ENABLE_GPL = True'/g ./edgeai_modelmaker/ai_modules/vision/constants.py
fi

if [[ ${PLUGINS_ENABLE_EXTRA} -ne 0 ]]; then
  sed -i s/'PLUGINS_ENABLE_EXTRA = False'/'PLUGINS_ENABLE_EXTRA = True'/g ./edgeai_modelmaker/ai_modules/vision/constants.py
fi

echo "cloning done."

#################################################################################
# upgrade pip
pip install --no-input --upgrade pip setuptools
pip install --no-input --upgrade wheel cython numpy

#################################################################################
echo "preparing environment..."
# for setup.py develop mode to work inside docker environment, this is required
git config --global --add safe.directory $(pwd)

echo "installing repositories..."

echo "installing: https://github.com/TexasInstruments/edgeai-torchvision"
cd ../edgeai-torchvision
./setup.sh

echo "installing: https://github.com/TexasInstruments/edgeai-mmdetection"
cd ../edgeai-mmdetection
./setup.sh

if [[ ${PLUGINS_ENABLE_GPL} -ne 0 ]]; then
  echo "installing: https://github.com/TexasInstruments/edgeai-yolov5 (GPLv3 Licensed)"
  cd ../edgeai-yolov5
  ./setup_for_modelmaker.sh
fi

echo "installing: https://github.com/TexasInstruments/edgeai-benchmark"
cd ../edgeai-benchmark
./setup_pc.sh r8.6

echo "installing edgeai-modelmaker"
cd ../edgeai-modelmaker
./setup.sh

# there as issue with installing pillow-simd through requirements - force it here
# 7.2.0.post1 is what works in Python3.6 - newer Python versions may be able to use a more recent one
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd==7.2.0.post1

ls -d ../edgeai-*

echo "installation done."
