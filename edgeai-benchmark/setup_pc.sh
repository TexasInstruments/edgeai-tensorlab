#!/usr/bin/env bash

# Copyright (c) 2018-2021, Texas Instruments
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
# change default tidl_tools version if needed
# examples: 11.0 10.1 10.0
TIDL_TOOLS_VERSION=${TIDL_TOOLS_VERSION:-"11.0"}
echo "TIDL_TOOLS_VERSION=${TIDL_TOOLS_VERSION}"

#######################################################################
# TIDL_TOOLS_TYPE can be "" or "_gpu" from r9.2 onwards
# TIDL_TOOLS_TYPE="_gpu" (while running this setup) to install tidl-tools compiled with CUDA support
# requires nvidia-hpc-sdk to be insalled for it to work: https://developer.nvidia.com/nvidia-hpc-sdk-237-downloads
TIDL_TOOLS_TYPE=${TIDL_TOOLS_TYPE:-""}
echo "TIDL_TOOLS_TYPE=${TIDL_TOOLS_TYPE}"

#######################################################################
echo 'Installing system dependencies...'

# Dependencies for cmake, onnx, pillow-simd, tidl-graph-visualization
sudo apt-get install -y cmake \
                        libffi-dev \
                        libjpeg-dev zlib1g-dev \
                        graphviz graphviz-dev protobuf-compiler


pip3 install --no-input --upgrade pip==24.2 setuptools==73.0.0
pip3 install --no-input cython wheel numpy==1.23.0 scipy==1.10 pyyaml tqdm

######################################################################
CURRENT_WORK_DIR=$(pwd)
TOOLS_BASE_PATH=${CURRENT_WORK_DIR}/tools

./setup_pandaset.sh

######################################################################
if [ -d "${CURRENT_WORK_DIR}/../edgeai-tidl-tools" ]; then
  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Found local edgeai-tidl-tools, installing osrt_model_tools in develop mode"
  echo "--------------------------------------------------------------------------------------------------------------"
  pip3 uninstall -y osrt_model_tools
  cd ${CURRENT_WORK_DIR}/../edgeai-tidl-tools/osrt-model-tools
  python3 setup.py develop
  cd ${CURRENT_WORK_DIR}
fi

echo "--------------------------------------------------------------------------------------------------------------"
echo "INFO: installing tidl-tools-package version: ${TIDL_TOOLS_VERSION}"
cd ${CURRENT_WORK_DIR}
pip3 install -r ./tools/requirements/requirements_${TIDL_TOOLS_VERSION}.txt
TIDL_TOOLS_TYPE=${TIDL_TOOLS_TYPE} TIDL_TOOLS_VERSION=${TIDL_TOOLS_VERSION} python3 ./tools/setup.py develop

cd ${CURRENT_WORK_DIR}
echo 'INFO: installing local module using setup.py...'
# there as issue with installing pillow-simd through requirements - force it here
pip3 uninstall --yes pillow
pip3 install --no-input -U --force-reinstall pillow-simd
pip3 install --no-input onnx==1.14.0 protobuf
pip3 install --no-input -r ./requirements/requirements_pc.txt
pip3 install --no-input onnx_graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com
python3 setup.py develop
echo "--------------------------------------------------------------------------------------------------------------"

######################################################################
# PYTHONPATH
# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

echo 'Completed installation.'
