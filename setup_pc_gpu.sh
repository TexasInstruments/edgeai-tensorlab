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

########################################################
# Install tidl-tools built with gpu support - needs nvidia-hpc-sdk to be installed in system to run
echo "Installing tidl-tools with CUDA GPU support and NVIDIA-HPC-SDK..."
echo "Please ensure that you have NVIDIA CUDA GPUs and that the GPU drivers are installed."

echo "Installing tidl-tools with CUDA GPU support..."
TIDL_TOOLS_TYPE_SUFFIX="_gpu" ./setup_pc.sh "$@"

########################################################
# NVIDIA-HPC-SDK
NVIDIA_HPC_SDK_VERSION="23.7"
NVIDIA_HPC_SDK_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/${NVIDIA_HPC_SDK_VERSION}"
NVIDIA_HPC_SDK_APT_NAME="nvhpc-23-7"

echo "Checking whether NVIDIA_HPC_SDK version ${NVIDIA_HPC_SDK_VERSION} is installed:"
if [ ! -d "${NVIDIA_HPC_SDK_PATH}" ]; then
  echo "${NVIDIA_HPC_SDK_PATH} does not exist. installing it ..."
  # https://developer.nvidia.com/nvidia-hpc-sdk-237-downloads
  curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
  echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
  sudo apt-get update -y
  # sudo apt install -y "${NVIDIA_HPC_SDK_APT_NAME}-cuda-multi"
  sudo apt install -y "${NVIDIA_HPC_SDK_APT_NAME}"
else
  echo "${NVIDIA_HPC_SDK_PATH} already exists - skipping installation."
fi

echo "============================================================"
echo "Please make sure that NVIDIA GPU drivers are installed."
echo "Option 1 - for ubuntu only - details are here: https://ubuntu.com/server/docs/nvidia-drivers-installation"
echo "  sudo ubuntu-drivers list"
echo "  sudo ubuntu-drivers install"
echo "  or a specific version can be installed with additional arguement - for example:"
echo "  sudo ubuntu-drivers install nvidia:550"
echo "Other options:"
echo "This can be ignored is the latest nvidia gpu driver is already installed:"
echo "NVIDIA Kernel drivers (Choose one of the following options)"
echo "Option 2: To install the legacy kernel module flavor:"
echo "  sudo apt-get install -y cuda-drivers"
echo "Option 3: To install the open kernel module flavor:"
echo "  sudo apt-get install -y nvidia-kernel-open-550"
echo "  sudo apt-get install -y cuda-drivers-550"
echo "============================================================"

