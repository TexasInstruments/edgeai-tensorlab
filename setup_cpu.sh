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

# clone git repositories
CLONE_GIT_REPOS=0

# pull git repositories
UPDATE_GIT_REPOS=0

# use requirements from: pip list --format=freeze
USE_PIP_FREEZE_REQUIREMENTS=1

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
if [[ ${CLONE_GIT_REPOS} -ne 0 ]]; then
    echo "cloning git repositories. this may take some time..."
    echo "if there is any issue, please remove these folders and try again ../edgeai-benchmark ../edgeai-mmdetection ../edgeai-torchvision ../edgeai-modelzoo ../edgeai-yolox"
    if [[ ! -d ../edgeai-benchmark ]]; then git clone --branch r10.0 ${SOURCE_LOCATION}edgeai-benchmark.git ../edgeai-benchmark; fi
    if [[ ! -d ../edgeai-mmdetection ]]; then git clone --branch r10.0 ${SOURCE_LOCATION}edgeai-mmdetection.git ../edgeai-mmdetection; fi
    if [[ ! -d ../edgeai-mmpose ]]; then git clone --branch r10.0 ${SOURCE_LOCATION}edgeai-mmpose.git ../edgeai-mmpose; fi
    if [[ ! -d ../edgeai-torchvision ]]; then git clone --branch r10.0 ${SOURCE_LOCATION}edgeai-torchvision.git ../edgeai-torchvision; fi
    if [[ ! -d ../edgeai-tensorvision ]]; then git clone --branch r10.0 ${SOURCE_LOCATION}edgeai-tensorvision.git ../edgeai-tensorvision; fi
    if [[ ! -d ../edgeai-modelzoo ]]; then git clone "--single-branch" --branch r10.0 ${SOURCE_LOCATION}edgeai-modelzoo.git ../edgeai-modelzoo; fi
    cd ../edgeai-modelmaker
    echo "git clone done."
fi

if [[ ${UPDATE_GIT_REPOS} -ne 0 ]]; then
    echo "pulling git repositories. this may take some time..."
    cd ../edgeai-benchmark; git stash; git fetch origin r10.0; git checkout r10.0; git pull --rebase
    cd ../edgeai-mmdetection; git stash; git fetch origin r10.0; git checkout r10.0; git pull --rebase
    cd ../edgeai-mmpose; git stash; git fetch origin r10.0; git checkout r10.0; git pull --rebase
    cd ../edgeai-torchvision; git stash; git fetch origin r10.0; git checkout r10.0; git pull --rebase
    cd ../edgeai-tensorvision; git stash; git fetch origin r10.0; git checkout r10.0; git pull --rebase
    cd ../edgeai-modelzoo; git stash; git fetch origin r10.0; git checkout r10.0; git pull --rebase
    cd ../edgeai-modelmaker
    echo "git pull done."
fi

#################################################################################
# upgrade pip
pip3 install --no-input --upgrade pip==24.2 setuptools==73.0.0
pip3 install --no-input cython wheel numpy==1.23.0

#################################################################################
echo "preparing environment..."
# for setup.py develop mode to work inside docker environment, this is required
git config --global --add safe.directory $(pwd)

echo "installing repositories..."

echo "installing: edgeai-torchvision"
cd ../edgeai-torchvision
./setup_cpu.sh

echo "installing: edgeai-tensorvision"
cd ../edgeai-tensorvision
./setup_cpu.sh

echo "installing: edgeai-mmdetection"
cd ../edgeai-mmdetection
./setup_cpu.sh

echo "installing: edgeai-mmpose"
cd ../edgeai-mmpose
./setup_cpu.sh

echo "installing: edgeai-mmdeploy"
cd ../edgeai-mmdeploy
./setup_cpu.sh

# uninstall the onnxruntime was installed by setups above, so that the correct version can be installed.
pip uninstall --yes onnxruntime

echo "installing: edgeai-benchmark"
cd ../edgeai-benchmark
./setup_pc.sh r10.1

######################################################################
echo "installing edgeai-modelmaker"
cd ../edgeai-modelmaker
# Installing dependencies
echo 'Installing python packages...'
pip3 install --no-input -r ./requirements.txt

echo 'Installing as a local module using setup.py'
python3 setup.py develop

######################################################################
# make sure that we are using pillow-simd (which is faster)
pip install --no-input -U --force-reinstall pillow-simd

if [[ ${USE_PIP_FREEZE_REQUIREMENTS} -ne 0 ]]; then
    pip install -r requirements_freeze.txt
fi

ls -d ../edgeai-*

echo "installation done."
