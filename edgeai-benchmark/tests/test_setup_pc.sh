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
version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) else 1;print(r)'`
if [ $version_match -ne 0 ]; then
echo 'python version must be >= 3.6'
exit 1
fi

######################################################################
# installers for tidl-tools nightly build
OSRT_TOOLS_BUILD_PATH=http://gtweb.dal.design.ti.com/nightly_builds/tidl-osrt-build/328-2022-12-07_02-19-58/artifacts/output #pip whls
TIDL_TOOLS_BUILD_PATH=http://gtweb.dal.design.ti.com/nightly_builds/tidl-osrt-build/327-2022-12-07_01-29-33/artifacts/output #j721e
#TIDL_TOOLS_BUILD_PATH=http://gtweb.dal.design.ti.com/nightly_builds/tidl-osrt-build/326-2022-12-06_23-27-59/artifacts/output #j721s2

######################################################################
function conditional_wget() {
    base_filename=$(basename $1)
    dest_file=$2/$base_filename
    if [ ! -f $dest_file ]
    then
        wget $1 -O $dest_file --no-check-certificate
    fi
}

function conditional_untar() {
    curdir=$(pwd)
    dir_name=$(dirname $1)
    cd ${dir_name}
    tar -xzf $(basename $1) --strip $2
    cd ${curdir}
}

#######################################################################
# Dependencies for building pillow-simd
#echo 'Installing dependencies to build pillow-simd. If you dont have sudo access, comment the below line and replace pillow-simd with pillow in the requirements file'
#sudo apt-get install -y libjpeg-dev zlib1g-dev

# Dependencies for TIDL graph visualization
echo 'Installing dependencies for TIDL graph visualization'
sudo apt-get install -y graphviz graphviz-dev

######################################################################
echo "Installing Python pre-requisits"
#conda install -y cython
pip3 install --no-input -r ./requirements_basic.txt

echo 'Installing python packages...'
pip3 install --no-input -r ./requirements_pc.txt

######################################################################
#NOTE: THIS STEP INSTALLS THE EDITABLE LOCAL MODULE pytidl
echo 'Installing as a local module using setup.py'
python3 setup.py develop

######################################################################
# Installing dependencies
echo 'Installing tidl_tools...'

TVM_DLR_TAR_NAME=dlr_1.10.0_x86_u18
ONNX_TAR_NAME=onnx_1.7.0_x86_u18
TFLITE_TAR_NAME=tflite_2.8_x86_u18
TIDL_TOOLS_NAME="tidl_tools"

#echo 'Cleaning up previous tidl_tools...'
rm -rf ${TIDL_TOOLS_NAME}
mkdir -p ${TIDL_TOOLS_NAME}

conditional_wget ${TIDL_TOOLS_BUILD_PATH}/tidl_tools/${TIDL_TOOLS_NAME}.tar.gz ${TIDL_TOOLS_NAME}
conditional_wget ${OSRT_TOOLS_BUILD_PATH}/dlr/${TVM_DLR_TAR_NAME}.tar.gz ${TIDL_TOOLS_NAME}
conditional_wget ${OSRT_TOOLS_BUILD_PATH}/onnx/${ONNX_TAR_NAME}.tar.gz ${TIDL_TOOLS_NAME}
conditional_wget ${OSRT_TOOLS_BUILD_PATH}/tflite_2.8/${TFLITE_TAR_NAME}.tar.gz ${TIDL_TOOLS_NAME}

# extract
conditional_untar ${TIDL_TOOLS_NAME}/${TIDL_TOOLS_NAME}.tar.gz 1
conditional_untar ${TIDL_TOOLS_NAME}/${TVM_DLR_TAR_NAME}.tar.gz 0
conditional_untar ${TIDL_TOOLS_NAME}/${ONNX_TAR_NAME}.tar.gz 0
conditional_untar ${TIDL_TOOLS_NAME}/${TFLITE_TAR_NAME}.tar.gz 0

pip3 install --no-input ${TIDL_TOOLS_NAME}/${TVM_DLR_TAR_NAME}/dlr-1.10.0-py3-none-any.whl
pip3 install --no-input ${TIDL_TOOLS_NAME}/${TVM_DLR_TAR_NAME}/tvm-0.9.dev0-cp36-cp36m-linux_x86_64.whl
pip3 install --no-input ${TIDL_TOOLS_NAME}/${ONNX_TAR_NAME}/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
pip3 install --no-input ${TIDL_TOOLS_NAME}/${TFLITE_TAR_NAME}/tflite_runtime-2.8.2-cp36-cp36m-linux_x86_64.whl

######################################################################
ARM64_GCC_FILE=gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
echo "[gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu] Checking ..."
cd ${TIDL_TOOLS_NAME}
if [ ! -d ${TIDL_TOOLS_NAME}/${ARM64_GCC_FILE} ]
then
    conditional_wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/${ARM64_GCC_FILE}.tar.xz .
    tar xf ${ARM64_GCC_FILE}.tar.xz > /dev/null
    #rm ${ARM64_GCC_FILE}.tar.xz
fi
cd ..
echo "[${ARM64_GCC_FILE}] Done"

######################################################################
export TIDL_TOOLS_PATH=$(pwd)/${TIDL_TOOLS_NAME}
echo "TIDL_TOOLS_PATH=${TIDL_TOOLS_PATH}"

export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

# needed for TVM compilation
export ARM64_GCC_PATH=$TIDL_TOOLS_PATH/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

echo 'Completed installation.'
