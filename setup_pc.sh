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
# change default tidl_tools version if needed - examples: latest stable r10.0 r9.2 r9.1 r9.0
TIDL_TOOLS_RELEASE_NAME="${1:-r10.1}"

echo "--------------------------------------------------------------------------------------------------------------"
echo "Installing tidl_tools version: ${TIDL_TOOLS_RELEASE_NAME}"
echo "--------------------------------------------------------------------------------------------------------------"

#######################################################################
# TIDL_TOOLS_TYPE_SUFFIX can be "" or "_gpu" from r9.2 onwards
# TIDL_TOOLS_TYPE_SUFFIX="_gpu" (while running this setup) to install tidl-tools compiled with CUDA support
# requires nvidia-hpc-sdk to be insalled for it to work: https://developer.nvidia.com/nvidia-hpc-sdk-237-downloads
TIDL_TOOLS_TYPE_SUFFIX=${TIDL_TOOLS_TYPE_SUFFIX:-""}
echo "TIDL_TOOLS_TYPE_SUFFIX=${TIDL_TOOLS_TYPE_SUFFIX}"

#######################################################################
echo 'Installing system dependencies...'

# Dependencies for cmake, onnx, pillow-simd, tidl-graph-visualization
sudo apt-get install -y cmake \
                        libffi-dev \
                        libjpeg-dev zlib1g-dev \
                        graphviz graphviz-dev protobuf-compiler

# upgrade pip
pip3 install --no-input --upgrade pip==24.2 setuptools==73.0.0
pip3 install --no-input cython wheel numpy==1.23.0

######################################################################
CURRENT_WORK_DIR=$(pwd)
TOOLS_BASE_PATH=${CURRENT_WORK_DIR}/tools

echo "--------------------------------------------------------------------------------------------------------------"
echo "Installing gcc arm required for tvm..."
echo "--------------------------------------------------------------------------------------------------------------"
GCC_ARM_AARCH64_NAME="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
GCC_ARM_AARCH64_FILE="${GCC_ARM_AARCH64_NAME}.tar.xz"
GCC_ARM_AARCH64_PATH="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/${GCC_ARM_AARCH64_FILE}"

echo "Installing ${GCC_ARM_AARCH64_NAME}"
if [ ! -d ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_NAME} ]; then
    if [ ! -f ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_FILE} ]; then
        wget -P ${TOOLS_BASE_PATH} ${GCC_ARM_AARCH64_PATH} --no-check-certificate
    fi
    tar xf ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_FILE} -C ${TOOLS_BASE_PATH} > /dev/null
    # rm -f ${TOOLS_BASE_PATH}/${GCC_ARM_AARCH64_FILE}
fi

######################################################################
# an array to keep download links
declare -a TIDL_TOOLS_DOWNLOAD_LINKS

if [[ $TIDL_TOOLS_RELEASE_NAME == "latest" || $TIDL_TOOLS_RELEASE_NAME == "r10.1" ]]; then
  # python version check = 3.10
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,10) and sys.version_info < (3,11) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.10 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi

  # for internal links, no proxy is required
  WGET_PROXY_SETTINGS="--proxy=off"

  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing requirements..."
  echo "--------------------------------------------------------------------------------------------------------------"
  # there as issue with installing pillow-simd through requirements - force it here
  pip3 uninstall --yes pillow
  pip3 install --no-input -U --force-reinstall pillow-simd
  pip3 install --no-input onnx==1.14.0 protobuf
  pip3 install --no-input -r ./requirements_pc.txt

  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing tidl_tools..."
  echo "--------------------------------------------------------------------------------------------------------------"
  TARGET_SOCS=(TDA4VM AM68A AM69A AM62A AM67A)
  TIDL_TOOLS_RELEASE_ID=10_00_08_00
  TIDL_TOOLS_VERSION_NAME="10.0"
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input http://tidl-ud-17.dhcp.ti.com/build/sdk_release/osrt/onnx/x86/onnxruntime_tidl-1.15.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.12.0-cp310-cp310-linux_x86_64.whl
  # these are internal links for now
  TIDL_TOOLS_DOWNLOAD_LINKS=("http://10.24.68.92/OSRT_TOOLS/10_01_00_01/am68pa/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "http://10.24.68.92/OSRT_TOOLS/10_01_00_01/am68a/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "http://10.24.68.92/OSRT_TOOLS/10_01_00_01/am69a/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "http://10.24.68.92/OSRT_TOOLS/10_01_00_01/am62a/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "http://10.24.68.92/OSRT_TOOLS/10_01_00_01/am67a/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz")
  for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
    TARGET_SOC=${TARGET_SOCS[$soc_idx]}
    TIDL_TOOLS_DOWNLOAD_LINK=${TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]}
    echo "$TARGET_SOC $TIDL_TOOLS_DOWNLOAD_LINK"
  done

elif [[ $TIDL_TOOLS_RELEASE_NAME == "stable" || $TIDL_TOOLS_RELEASE_NAME == "r10.0" ]]; then
  # python version check = 3.10
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,10) and sys.version_info < (3,11) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.10 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi

  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing requirements..."
  echo "--------------------------------------------------------------------------------------------------------------"
  # there as issue with installing pillow-simd through requirements - force it here
  pip3 uninstall --yes pillow
  pip3 install --no-input -U --force-reinstall pillow-simd
  pip3 install --no-input protobuf==3.20.2 onnx==1.13.0
  pip3 install --no-input -r ./requirements_pc.txt

  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing tidl_tools..."
  echo "--------------------------------------------------------------------------------------------------------------"
  TARGET_SOCS=(TDA4VM AM68A AM69A AM62A AM67A)
  TIDL_TOOLS_RELEASE_ID=10_00_08_00
  TIDL_TOOLS_VERSION_NAME="10.0"
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.14.0+10000005-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.12.0-cp310-cp310-linux_x86_64.whl
  # these are internal links for now
  TIDL_TOOLS_DOWNLOAD_LINKS=("https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM68PA/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM68A/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM69A/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM62A/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM67A/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz")
  for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
    TARGET_SOC=${TARGET_SOCS[$soc_idx]}
    TIDL_TOOLS_DOWNLOAD_LINK=${TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]}
    echo "$TARGET_SOC $TIDL_TOOLS_DOWNLOAD_LINK"
  done

elif [[ $TIDL_TOOLS_RELEASE_NAME == "r9.2" ]]; then
  # python version check = 3.10
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,10) and sys.version_info < (3,11) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.10 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing requirements..."
  echo "--------------------------------------------------------------------------------------------------------------"
  # there as issue with installing pillow-simd through requirements - force it here
  pip3 uninstall --yes pillow
  pip3 install --no-input -U --force-reinstall pillow-simd
  pip3 install --no-input protobuf==3.20.2 onnx==1.13.0
  pip3 install --no-input -r ./requirements_pc.txt

  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing tidl_tools..."
  echo "--------------------------------------------------------------------------------------------------------------"
  TARGET_SOCS=(TDA4VM AM68A AM69A AM62A AM67A)
  TIDL_TOOLS_RELEASE_ID=09_02_09_00
  TIDL_TOOLS_VERSION_NAME="9.2"
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.14.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.8.2-cp310-cp310-linux_x86_64.whl
  # these are internal links for now
  TIDL_TOOLS_DOWNLOAD_LINKS=("https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM68PA/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM68A/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM69A/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM62A/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM67A/tidl_tools${TIDL_TOOLS_TYPE_SUFFIX}.tar.gz")
  for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
    TARGET_SOC=${TARGET_SOCS[$soc_idx]}
    TIDL_TOOLS_DOWNLOAD_LINK=${TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]}
    echo "$TARGET_SOC $TIDL_TOOLS_DOWNLOAD_LINK"
  done

elif [[ $TIDL_TOOLS_RELEASE_NAME == "r9.1" ]]; then
  # python version check = 3.10
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,10) and sys.version_info < (3,11) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.10 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi
  
  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing requirements..."
  echo "--------------------------------------------------------------------------------------------------------------"
  # there as issue with installing pillow-simd through requirements - force it here
  pip3 uninstall --yes pillow
  pip3 install --no-input -U --force-reinstall pillow-simd
  pip3 install --no-input protobuf==3.20.2 onnx==1.13.0
  pip3 install --no-input -r ./requirements_pc.txt

  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing tidl_tools..."
  echo "--------------------------------------------------------------------------------------------------------------"
  TARGET_SOCS=(TDA4VM AM68A AM69A AM62A)
  TIDL_TOOLS_RELEASE_ID=09_01_00_00
  TIDL_TOOLS_VERSION_NAME="9.1"
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.14.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.8.2-cp310-cp310-linux_x86_64.whl
  # these are internal links for now
  TIDL_TOOLS_DOWNLOAD_LINKS=("https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM68PA/tidl_tools.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM68A/tidl_tools.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM69A/tidl_tools.tar.gz" "https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/AM62A/tidl_tools.tar.gz")
  for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
    TARGET_SOC=${TARGET_SOCS[$soc_idx]}
    TIDL_TOOLS_DOWNLOAD_LINK=${TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]}
    echo "$TARGET_SOC $TIDL_TOOLS_DOWNLOAD_LINK"
  done

elif  [[ $TIDL_TOOLS_RELEASE_NAME == "test9.0.1" ]]; then
  # python version check = 3.10
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,10) and sys.version_info < (3,11) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.10 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi

  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing requirements..."
  echo "--------------------------------------------------------------------------------------------------------------"
  # onnx - override the onnx version installed by onnxsim
  # building onnx from soure requires carefull steps
  # make sure that we are using system cmake
  pip uninstall --yes cmake
  # pybind11[global] is needed for building the onnx package.
  # for some reason, this has to be installed before the requirements file is used.
  pip3 install --no-input pybind11[global] protobuf==3.19.4
  pybind11_DIR=$(pybind11-config --cmakedir) pip3 install --no-input https://github.com/TexasInstruments/onnx/archive/tidl-j7.zip

  # there as issue with installing pillow-simd through requirements - force it here
  pip3 uninstall --yes pillow
  pip3 install --no-input -U --force-reinstall pillow-simd
  pip3 install --no-input -r ./requirements_pc.txt

  # installers for internal release
  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing tidl_tools..."
  echo "--------------------------------------------------------------------------------------------------------------"
  TARGET_SOCS=(TDA4VM AM68A AM69A AM62A)
  TIDL_TOOLS_RELEASE_ID=09_00_00_01
  TIDL_TOOLS_VERSION_NAME="test9.0.1"
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.7.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.8.2-cp310-cp310-linux_x86_64.whl
  # these are internal links
  TIDL_TOOLS_DOWNLOAD_LINKS=("http://edgeaisrv2.dhcp.ti.com/publish/tidl/j721e/09_00_06_00/tidl_tools.tar.gz" "http://edgeaisrv2.dhcp.ti.com/publish/tidl/j721s2/09_00_06_00/tidl_tools.tar.gz" "http://edgeaisrv2.dhcp.ti.com/publish/tidl/j784s4/09_00_06_00/tidl_tools.tar.gz" "http://edgeaisrv2.dhcp.ti.com/publish/tidl/am62a/09_00_06_01/tidl_tools.tar.gz")
  for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
    TARGET_SOC=${TARGET_SOCS[$soc_idx]}
    TIDL_TOOLS_DOWNLOAD_LINK=${TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]}
    echo "$TARGET_SOC $TIDL_TOOLS_DOWNLOAD_LINK"
  done
  
elif [[ $TIDL_TOOLS_RELEASE_NAME == "r9.0" ]]; then
  # python version check = 3.10
  version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,10) and sys.version_info < (3,11) else 1;print(r)'`
  if [ $version_match -ne 0 ]; then
      echo "python version must be == 3.10 for $TIDL_TOOLS_RELEASE_NAME"
      exit 1
  fi

  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing requirements..."
  echo "--------------------------------------------------------------------------------------------------------------"
  # onnx - override the onnx version installed by onnxsim
  # building onnx from soure requires carefull steps
  # make sure that we are using system cmake
  pip uninstall --yes cmake
  # pybind11[global] is needed for building the onnx package.
  # for some reason, this has to be installed before the requirements file is used.
  pip3 install --no-input pybind11[global] protobuf==3.19.4
  pybind11_DIR=$(pybind11-config --cmakedir) pip3 install --no-input https://github.com/TexasInstruments/onnx/archive/tidl-j7.zip

  # there as issue with installing pillow-simd through requirements - force it here
  pip3 uninstall --yes pillow
  pip3 install --no-input -U --force-reinstall pillow-simd
  pip3 install --no-input -r ./requirements_pc.txt

  # installers for 9.0 release
  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Installing tidl_tools..."
  echo "--------------------------------------------------------------------------------------------------------------"
  TARGET_SOCS=(TDA4VM AM68A AM69A AM62A)
  TIDL_TOOLS_RELEASE_ID=09_00_00_01
  TIDL_TOOLS_VERSION_NAME="9.0"
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.7.0-cp310-cp310-linux_x86_64.whl
  pip3 install --no-input https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.8.2-cp310-cp310-linux_x86_64.whl
  for (( soc_idx=0; soc_idx<"${#TARGET_SOCS[@]}"; soc_idx++ )); do
    TARGET_SOC=${TARGET_SOCS[$soc_idx]}
    TIDL_TOOLS_DOWNLOAD_LINKS[$soc_idx]="https://software-dl.ti.com/jacinto7/esd/tidl-tools/${TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/${TARGET_SOC}/tidl_tools.tar.gz"
  done

else
  echo "tidl_tools version {$TIDL_TOOLS_RELEASE_NAME} was not recognized - todl_tools cannot be installed."
  exit 1
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
  wget ${WGET_PROXY_SETTINGS} -O ${TIDL_TOOLS_SOC_PREFIX}/tidl_tools.tar.gz ${TIDL_TOOLS_DOWNLOAD_LINK}
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

echo "Completed installation of tidl_tools."
echo "Short version name:${TIDL_TOOLS_VERSION_NAME}  Git branch:${TIDL_TOOLS_RELEASE_NAME}  Tools version:${TIDL_TOOLS_RELEASE_ID}"

######################################################################
if [ -d "${CURRENT_WORK_DIR}/../edgeai-tidl-tools" ]; then
  echo "--------------------------------------------------------------------------------------------------------------"
  echo "Found local edgeai-tidl-tools, installing osrt_model_tools in develop mode"
  echo "--------------------------------------------------------------------------------------------------------------"
  pip3 uninstall -y osrt_model_tools
  cd ${CURRENT_WORK_DIR}/../edgeai-tidl-tools/scripts
  python3 setup.py develop
  cd ${CURRENT_WORK_DIR}
fi

######################################################################
echo "--------------------------------------------------------------------------------------------------------------"
echo 'Installing local module using setup.py...'
echo "--------------------------------------------------------------------------------------------------------------"
python3 setup.py develop

######################################################################
# PYTHONPATH
# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

echo 'Completed installation.'
