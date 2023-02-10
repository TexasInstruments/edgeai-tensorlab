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
# change default tidl_tools version if needed - examples: latest stable r8.6 r8.5 r8.4
# default is latest
TIDL_TOOLS_VERSION=${1:-stable}

# Select one target SOC: TDA4VM, AM62A, AM68A, AM69A
# default is TDA4VM
TARGET_SOC=${2:-TDA4VM}

######################################################################
CURRENT_WORK_DIR=$(pwd)
TOOLS_BASE_PATH=${CURRENT_WORK_DIR}/tools

# Tools for selected SOC will be here.
TIDL_TOOLS_PREFIX="${TOOLS_BASE_PATH}/${TARGET_SOC}"
mkdir -p ${TIDL_TOOLS_PREFIX}

echo "SOC: ${TARGET_SOC}"
echo "Tools Location: ${TIDL_TOOLS_PREFIX}"
echo "Installing tidl_tools Verion: ${TIDL_TOOLS_VERSION} ..."

######################################################################
# Dependencies for building pillow-simd
#echo 'Installing dependencies to build pillow-simd. If you dont have sudo access, comment the below line and replace pillow-simd with pillow in the requirements file'
sudo apt-get install -y libjpeg-dev zlib1g-dev

# Dependencies for TIDL graph visualization
echo 'Installing dependencies for TIDL graph visualization'
sudo apt-get install -y graphviz graphviz-dev

#################################################################################
# upgrade pip
pip install --upgrade pip
pip install --upgrade setuptools

######################################################################
echo 'Installing python packages...'
pip install --no-input cython numpy wheel
pip3 install --no-input -r ./requirements_pc.txt

# there as issue with installing pillow-simd through requirements - force it here
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd

######################################################################
#NOTE: THIS STEP INSTALLS THE EDITABLE LOCAL MODULE pytidl
echo 'Installing as a local module using setup.py'
python3 setup.py develop

######################################################################
echo 'Cleaning up previous tidl_tools...'
rm -rf tidl_tools.tar.gz tidl_tools ${TIDL_TOOLS_PREFIX}

######################################################################
# Installing dependencies
if [[ $TIDL_TOOLS_VERSION == "latest" || $TIDL_TOOLS_VERSION == "r8.6" ]]; then
  # installers for 8.6 release
  echo 'tidl_tools version 8.6'
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/OSRT_TOOLS/X86_64_LINUX/UBUNTU_18_04/dlr-1.10.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/OSRT_TOOLS/X86_64_LINUX/UBUNTU_18_04/tvm-0.9.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/OSRT_TOOLS/X86_64_LINUX/UBUNTU_18_04/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/OSRT_TOOLS/X86_64_LINUX/UBUNTU_18_04/tflite_runtime-2.8.2-cp36-cp36m-linux_x86_64.whl
  wget -P ${TIDL_TOOLS_PREFIX} https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/TIDL_TOOLS/${TARGET_SOC}/tidl_tools.tar.gz
elif [[ $TIDL_TOOLS_VERSION == "stable" || $TIDL_TOOLS_VERSION == "r8.5" ]]; then
  # installers for 8.5 release
  echo 'tidl_tools version 8.5'
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/pywhl/dlr-1.10.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/pywhl/tvm-0.9.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/pywhl/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/pywhl/tflite_runtime-2.8.2-cp36-cp36m-linux_x86_64.whl
  wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/tidl_tools.tar.gz
elif [[ $TIDL_TOOLS_VERSION == "r8.4" ]]; then
  # installers for 8.4 release
  echo 'tidl_tools version 8.4'
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/x86_64/pywhl/dlr-1.10.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/x86_64/pywhl/tvm-1.11.1.dev335+g13a4007ca-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/x86_64/pywhl/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/x86_64/pywhl/tflite_runtime-2.8.2-cp36-cp36m-linux_x86_64.whl
  wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/tidl_tools.tar.gz
elif [[ $TIDL_TOOLS_VERSION == "r8.2" ]]; then
  # installers for 8.2 release
  echo 'tidl_tools version 8.2'
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/dlr-1.10.0-py3-none-any.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_05/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
  wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/tidl_tools.tar.gz
elif [[ $TIDL_TOOLS_VERSION == "r8.1" ]]; then
  # installers for 8.1 release
  echo 'tidl_tools version 8.1'
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/dlr-1.8.0-py3-none-any.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_01_00_09-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
  wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_01_00_09-rc1/tidl_tools.tar.gz
elif [[ $TIDL_TOOLS_VERSION == "r8.0" ]]; then
  # installers for 8.0 release
  echo 'tidl_tools version 8.0'
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/dlr-1.8.0-py3-none-any.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
  wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tidl_tools.tar.gz
else
  echo "tidl_tools version $TIDL_TOOLS_VERSION was not found"
fi

tar -xzf ${TIDL_TOOLS_PREFIX}/tidl_tools.tar.gz -C ${TIDL_TOOLS_PREFIX}

######################################################################
GCC_ARM_AARCH64_NAME="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
GCC_ARM_AARCH64_FILE="${GCC_ARM_AARCH64_NAME}.tar.xz"
GCC_ARM_AARCH64_PATH="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/${GCC_ARM_AARCH64_FILE}"

# needed for TVM compilation
echo "Checking ${GCC_ARM_AARCH64_NAME}"
if [ ! -d ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_NAME} ]; then
    if [ ! -f ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_FILE} ]; then
        wget -P ${TOOLS_BASE_PATH} ${GCC_ARM_AARCH64_PATH} --no-check-certificate
    fi
    tar xf ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_FILE} -C ${TOOLS_BASE_PATH} > /dev/null
    # rm -f ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_FILE}
fi
echo "Completed ${GCC_ARM_AARCH64_NAME}"

######################################################################
# TIDL_TOOLS_PATH
export TIDL_TOOLS_PATH=${TIDL_TOOLS_PREFIX}/tidl_tools
echo "TIDL_TOOLS_PATH: ${TIDL_TOOLS_PATH}"

# ARM64_GCC_PATH
export ARM64_GCC_PATH=$TIDL_TOOLS_PATH/${GCC_ARM_AARCH64_NAME}
cd ${TIDL_TOOLS_PATH}
ln -snf ../../${GCC_ARM_AARCH64_NAME}
cd ${CURRENT_WORK_DIR}

# LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${TIDL_TOOLS_PATH}
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# PYTHONPATH
# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

echo 'Completed installation.'
