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
# change default tidl_tools version if needed - examples: latest stable r9.0 r8.6 r8.5 r8.4
TIDL_TOOLS_RELEASE_NAME="${1:-r9.0}"
echo "tidl_tools version ${TIDL_TOOLS_RELEASE_NAME}"
#######################################################################
echo 'Installing system dependencies'


# Dependencies for cmake, onnx, pillow-simd, tidl-graph-visualization
sudo apt-get install -y cmake \
                        libffi-dev \
                        libjpeg-dev zlib1g-dev \
                        graphviz graphviz-dev protobuf-compiler

#################################################################################
# upgrade pip
pip3 install --upgrade pip setuptools

######################################################################
echo 'Installing python packages...'
echo 'Installing python packages...'
# there as issue with installing pillow-simd through requirements - force it here
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd

echo "installing requirements"
pip3 install --no-input -r ./requirements_pc.txt

# building onnx from soure requires carefull steps
# make sure that we are using system cmake
pip uninstall --yes cmake
# pybind11[global] is needed for building the onnx package.
# for some reason, this has to be installed before the requirements file is used.
pip3 install --no-input pybind11[global] protobuf==3.19.4
pybind11_DIR=$(pybind11-config --cmakedir) pip3 install --no-input https://github.com/TexasInstruments/onnx/archive/tidl-j7.zip

######################################################################
#NOTE: THIS STEP INSTALLS THE EDITABLE LOCAL MODULE pytidl
echo 'Installing as a local module using setup.py'
python3 setup.py develop

######################################################################
CURRENT_WORK_DIR=$(pwd)
TOOLS_BASE_PATH=${CURRENT_WORK_DIR}/tools

######################################################################
GCC_ARM_AARCH64_NAME="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
GCC_ARM_AARCH64_FILE="${GCC_ARM_AARCH64_NAME}.tar.xz"
GCC_ARM_AARCH64_PATH="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/${GCC_ARM_AARCH64_FILE}"

# needed for TVM compilation
echo "Installing ${GCC_ARM_AARCH64_NAME}"
if [ ! -d ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_NAME} ]; then
    if [ ! -f ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_FILE} ]; then
        wget -P ${TOOLS_BASE_PATH} ${GCC_ARM_AARCH64_PATH} --no-check-certificate
    fi
    tar xf ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_FILE} -C ${TOOLS_BASE_PATH} > /dev/null
    # rm -f ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_FILE}
fi

######################################################################
echo 'Cleaning up previous tidl_tools...'
rm -rf tidl_tools.tar.gz tidl_tools

######################################################################
echo "Installing tidl_tools verion: ${TIDL_TOOLS_RELEASE_NAME} ..."

# an array to keep download links
declare -a TIDL_TOOLS_DOWNLOAD_LINKS

if [[ $TIDL_TOOLS_RELEASE_NAME == "latest" || $TIDL_TOOLS_RELEASE_NAME == "r9.0" ]]; then
  # python version check = 3.10
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,10) and sys.version_info < (3,11) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.10 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  # installers for 9.0 release
  echo 'tidl_tools version 9.0'
  TARGET_SOCS=(AM62A AM68A AM69A TDA4VM)
  TIDL_TOOLS_RELEASE_ID=09_00_00_01
  TIDL_TOOLS_VERSION_NAME=9.0
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.7.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.8.2-cp310-cp310-linux_x86_64.whl
  for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
    TARGET_SOC=${TARGET_SOCS[$soc_idx]}
    TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]="https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/${TARGET_SOC}/tidl_tools.tar.gz"
  done
elif [[ $TIDL_TOOLS_RELEASE_NAME == "stable" || $TIDL_TOOLS_RELEASE_NAME == "r8.6" ]]; then
  # python version check = 3.6
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) and sys.version_info < (3,7) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.6 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  # installers for 8.6 release
  echo 'tidl_tools version 8.6'
  TARGET_SOCS=(AM62A AM68A AM69A TDA4VM)
  TIDL_TOOLS_RELEASE_ID=08_06_00_00
  TIDL_TOOLS_VERSION_NAME=8.6
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/OSRT_TOOLS/X86_64_LINUX/UBUNTU_18_04/dlr-1.10.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/OSRT_TOOLS/X86_64_LINUX/UBUNTU_18_04/tvm-0.9.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/OSRT_TOOLS/X86_64_LINUX/UBUNTU_18_04/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/OSRT_TOOLS/X86_64_LINUX/UBUNTU_18_04/tflite_runtime-2.8.2-cp36-cp36m-linux_x86_64.whl
  for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
    TARGET_SOC=${TARGET_SOCS[$soc_idx]}
    TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]="https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/TIDL_TOOLS_RC5/${TARGET_SOC}/tidl_tools.tar.gz"
  done
elif [[ $TIDL_TOOLS_RELEASE_NAME == "r8.5" ]]; then
  # python version check = 3.6
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) and sys.version_info < (3,7) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.6 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  # installers for 8.5 release
  echo 'tidl_tools version 8.5'
  TARGET_SOCS=(TDA4VM)
  TIDL_TOOLS_RELEASE_ID=08_05_00_00
  TIDL_TOOLS_VERSION_NAME=8.5
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/pywhl/dlr-1.10.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/pywhl/tvm-0.9.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/pywhl/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/pywhl/tflite_runtime-2.8.2-cp36-cp36m-linux_x86_64.whl
  TIDL_TOOLS_DOWNLOAD_LINKS[0]="https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/tidl_tools.tar.gz"
elif [[ $TIDL_TOOLS_RELEASE_NAME == "r8.4" ]]; then
  # python version check = 3.6
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) and sys.version_info < (3,7) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.6 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  # installers for 8.4 release
  echo 'tidl_tools version 8.4'
  TARGET_SOCS=(TDA4VM)
  TIDL_TOOLS_RELEASE_ID=08_04_00_00
  TIDL_TOOLS_VERSION_NAME=8.4
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/x86_64/pywhl/dlr-1.10.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/x86_64/pywhl/tvm-1.11.1.dev335+g13a4007ca-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/x86_64/pywhl/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/x86_64/pywhl/tflite_runtime-2.8.2-cp36-cp36m-linux_x86_64.whl
  TIDL_TOOLS_DOWNLOAD_LINKS[0]="https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/tidl_tools.tar.gz"
elif [[ $TIDL_TOOLS_RELEASE_NAME == "r8.2" ]]; then
  # python version check = 3.6
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) and sys.version_info < (3,7) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.6 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  # installers for 8.2 release
  echo 'tidl_tools version 8.2'
  TARGET_SOCS=(TDA4VM)
  TIDL_TOOLS_RELEASE_ID=08_02_00_01
  TIDL_TOOLS_VERSION_NAME=8.2
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/dlr-1.10.0-py3-none-any.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_05/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
  TIDL_TOOLS_DOWNLOAD_LINKS[0]="https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/tidl_tools.tar.gz"
elif [[ $TIDL_TOOLS_RELEASE_NAME == "r8.1" ]]; then
  # python version check = 3.6
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) and sys.version_info < (3,7) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.6 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  # installers for 8.1 release
  echo 'tidl_tools version 8.1'
  TARGET_SOCS=(TDA4VM)
  TIDL_TOOLS_RELEASE_ID=08_01_00_00
  TIDL_TOOLS_VERSION_NAME=8.1
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/dlr-1.8.0-py3-none-any.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_01_00_09-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
  TIDL_TOOLS_DOWNLOAD_LINKS[0]="https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_01_00_09-rc1/tidl_tools.tar.gz"
elif [[ $TIDL_TOOLS_RELEASE_NAME == "r8.0" ]]; then
  # python version check = 3.6
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) and sys.version_info < (3,7) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.6 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  # installers for 8.0 release
  echo 'tidl_tools version 8.0'
  TARGET_SOCS=(TDA4VM)
  TIDL_TOOLS_RELEASE_ID=08_00_00_00
  TIDL_TOOLS_VERSION_NAME=8.0
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/dlr-1.8.0-py3-none-any.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
  pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
  TIDL_TOOLS_DOWNLOAD_LINKS[0]="https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tidl_tools.tar.gz"
else
  # python version check = 3.10
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,10) and sys.version_info < (3,11) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.10 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  # installers for internal release
  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Important note: The release name provided is not a a known version. Assuming that it is an internal release tag: ${TIDL_TOOLS_RELEASE_NAME}"
  echo "If instead a release version is required, then use the appropriate name. eg: r9.0"
  echo "--------------------------------------------------------------------------------------------------------------"
  TARGET_SOCS=(AM62A AM68A AM69A TDA4VM)
  TIDL_TOOLS_RELEASE_ID=09_00_00_00
  TIDL_TOOLS_VERSION_NAME=${TIDL_TOOLS_RELEASE_NAME}
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.7.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.8.2-cp310-cp310-linux_x86_64.whl
  TIDL_TOOLS_DOWNLOAD_LINKS=("http://edgeaisrv2.dhcp.ti.com/publish/modelzoo/${TIDL_TOOLS_RELEASE_NAME}/am62a/tidl_tools.tar.gz" "http://edgeaisrv1.dhcp.ti.com/publish/modelzoo/${TIDL_TOOLS_RELEASE_NAME}/j721s2/tidl_tools.tar.gz" "http://edgeaisrv1.dhcp.ti.com/publish/modelzoo/${TIDL_TOOLS_RELEASE_NAME}/j784s4/tidl_tools.tar.gz" "http://edgeaisrv2.dhcp.ti.com/publish/modelzoo/${TIDL_TOOLS_RELEASE_NAME}/j721e/tidl_tools.tar.gz")
  for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
    TARGET_SOC=${TARGET_SOCS[$soc_idx]}
    TIDL_TOOLS_DOWNLOAD_LINK=${TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]}
    echo "$TARGET_SOC $TIDL_TOOLS_DOWNLOAD_LINK"
  done
fi

######################################################################
echo "TARGET_SOCS ${TARGET_SOCS[@]}"
for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
  # Tools for selected SOC will be here.
  TARGET_SOC=${TARGET_SOCS[$soc_idx]}
  echo "Installing tidl_tools for TARGET_SOC: ${TARGET_SOC}"
  TIDL_TOOLS_SOC_PREFIX="${TOOLS_BASE_PATH}/${TARGET_SOC}"
  TIDL_TOOLS_DOWNLOAD_LINK=${TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]}

  echo "Installing tidl_tools for SOC: ${TARGET_SOC} to: ${TIDL_TOOLS_SOC_PREFIX} from: ${TIDL_TOOLS_DOWNLOAD_LINK}"
  rm -rf ${TIDL_TOOLS_SOC_PREFIX}
  mkdir -p ${TIDL_TOOLS_SOC_PREFIX}
  wget -O ${TIDL_TOOLS_SOC_PREFIX}/tidl_tools.tar.gz ${TIDL_TOOLS_DOWNLOAD_LINK}
  tar -xzf ${TIDL_TOOLS_SOC_PREFIX}/tidl_tools.tar.gz -C ${TIDL_TOOLS_SOC_PREFIX}

  # note: this is just en example of setting TIDL_TOOLS_PATH and ARM64_GCC_PATH - this will be overwritten in this loop
  # these and the variables dependent on it will need to defined by the program that need to use tidl_tools
  # actually: these need to defined with export prefix
  TIDL_TOOLS_PATH="${TIDL_TOOLS_SOC_PREFIX}/tidl_tools"
  LD_LIBRARY_PATH=${TIDL_TOOLS_PATH}
  ARM64_GCC_PATH="${TIDL_TOOLS_PATH}/${GCC_ARM_AARCH64_NAME}"

  # create symbolic link for the arm-gcc downloaded into a common folder
  cd ${TIDL_TOOLS_PATH}
  ln -snf ../../${GCC_ARM_AARCH64_NAME}
  cd ${CURRENT_WORK_DIR}

  # write version information
  echo "target_device: ${TARGET_SOC}" > ${TIDL_TOOLS_PATH}/version.yaml
  echo "version: ${TIDL_TOOLS_VERSION_NAME}" >> ${TIDL_TOOLS_PATH}/version.yaml
  echo "release_id: ${TIDL_TOOLS_RELEASE_ID}" >> ${TIDL_TOOLS_PATH}/version.yaml
  echo "release_name: ${TIDL_TOOLS_RELEASE_NAME}" >> ${TIDL_TOOLS_PATH}/version.yaml
done

######################################################################
# PYTHONPATH
# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

echo 'Completed installation.'
