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

##################################################################


# for onnxruntime and tflite_runtime, the artifacts are same for pc and j7 devices
# however for tvmdlr, there are two sets of artifacts - one for pc and one for j7 device
# deploy_lib.so.pc is for pc and deploy_lib.so.j7 is for j7 device
# a symbolic link called deploy_lib.so needs to be created, depending on where we plan to run the inference.
# this can be done after the import has been done and artifacts generated.
# by default it points to deploy_lib.so.pc, so nothing needs to be done for inference on pc

if [[ $# -ne 1 ]]; then
  echo "please provide exactly one argument - either pc or j7"
  exit 1
fi

# tvmdlr artifacts are different for pc and j7 device
# point to the right artifact before this script executes
source run_set_target_device.sh $1

# setup the environment
# source run_setupenv_pc.sh
export TIDL_TOOLS_PATH=$(pwd)/tidl_tools
echo "TIDL_TOOLS_PATH=${TIDL_TOOLS_PATH}"

export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

# needed for TVM compilation
export ARM64_GCC_PATH=$TIDL_TOOLS_PATH/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

# additional environment variables for detailed performance measurement
export TIDL_RT_DDR_STATS="1"
export TIDL_RT_PERFSTATS="1"
