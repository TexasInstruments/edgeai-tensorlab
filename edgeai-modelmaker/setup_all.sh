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

# set to 1 to enable other extra models
PLUGINS_ENABLE_EXTRA=1

#################################################################################
if [[ ${USE_INTERNAL_REPO} -eq 0 ]]; then
    SOURCE_LOCATION="https://github.com/TexasInstruments/"
else
    SOURCE_LOCATION="ssh://git@bitbucket.itg.ti.com/edgeai-algo/"
fi
# print
echo "SOURCE_LOCATION="${SOURCE_LOCATION}

#################################################################################
# clone
echo "cloning/updating git repositories. this may take some time..."
echo "if there is any issue, please remove these folders and try again ../edgeai-benchmark ../edgeai-mmdetection ../edgeai-torchvision ../edgeai-modelzoo ../edgeai-yolox"
if [[ ! -d ../edgeai-benchmark ]]; then git clone --branch r9.1 ${SOURCE_LOCATION}edgeai-benchmark.git ../edgeai-benchmark; else cd ../edgeai-benchmark; git stash; git fetch origin r9.1; git checkout r9.1; git pull --rebase; fi
if [[ ! -d ../edgeai-mmdetection ]]; then git clone --branch r9.1 ${SOURCE_LOCATION}edgeai-mmdetection.git ../edgeai-mmdetection; else cd ../edgeai-mmdetection; git stash; git fetch origin r9.1; git checkout r9.1; git pull --rebase; fi
if [[ ! -d ../edgeai-torchvision ]]; then git clone --branch r9.1 ${SOURCE_LOCATION}edgeai-torchvision.git ../edgeai-torchvision; else cd ../edgeai-torchvision; git stash; git fetch origin r9.1; git checkout r9.1; git pull --rebase; fi
if [[ ! -d ../edgeai-modelzoo ]]; then git clone "--single-branch" --branch r9.1 ${SOURCE_LOCATION}edgeai-modelzoo.git ../edgeai-modelzoo; else cd ../edgeai-modelzoo; git stash; git fetch origin r9.1; git checkout r9.1; git pull --rebase; fi

if [[ ${PLUGINS_ENABLE_EXTRA} -ne 0 ]]; then
  if [[ ! -d ../edgeai-yolox ]]; then git clone --branch r9.1 ${SOURCE_LOCATION}edgeai-yolox.git ../edgeai-yolox; else cd ../edgeai-yolox; git stash; git fetch origin r9.1; git checkout r9.1; git pull --rebase; fi
fi

cd ../edgeai-modelmaker
echo "cloning/updating done."

#################################################################################
# upgrade pip
pip3 install --no-input --upgrade pip==23.3.1 setuptools==69.0.2
pip install --no-input --upgrade wheel cython numpy==1.23.0

#################################################################################
echo "preparing environment..."
# for setup.py develop mode to work inside docker environment, this is required
git config --global --add safe.directory $(pwd)

echo "installing repositories..."

echo "installing: edgeai-torchvision"
cd ../edgeai-torchvision
./setup_cpu.sh

echo "installing: edgeai-mmdetection"
cd ../edgeai-mmdetection
./setup_cpu.sh

if [[ ${PLUGINS_ENABLE_EXTRA} -ne 0 ]]; then
  echo "installing: edgeai-yolox"
  cd ../edgeai-yolox
  ./setup_cpu.sh
fi

# uninstall the onnxruntime was installed by setups above, so that the correct version can be installed.
pip uninstall --yes onnxruntime

echo "installing: edgeai-benchmark"
cd ../edgeai-benchmark
./setup_pc.sh r9.1

echo "installing edgeai-modelmaker"
cd ../edgeai-modelmaker
./setup.sh

# make sure that we are using pillow-simd (which is faster)
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd

ls -d ../edgeai-*

echo "installation done."
