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
if [ $# -le 1 ]; then
    echo "help:"
    echo "$0 target_device config_file"
    echo "target_device can be one of AM68A AM69A AM62A AM67A AM62 TDA4VM"
    exit 0
fi

#################################################################################
# optional: check if AVX instructions are available in the machine
# by default AVX is enabled - setting this TIDL_RT_AVX_REF flag to "0" wil disable AVX
CPUINFO_NUM_AVX_CORES=$(cat /proc/cpuinfo|grep avx|wc|tail -n 1|awk '{print $1;}')
if [ ${CPUINFO_NUM_AVX_CORES} -eq 0 ]; then
  export TIDL_RT_AVX_REF="0"
else
  export TIDL_RT_AVX_REF="1"
fi

#################################################################################
# first argument - target_device - use one of: TDA4VM AM62A AM68A AM69A AM67A AM62
TARGET_SOC=${1:-"AM68A"}

# second argument - config_file - use an appropriate config file - see examples given in this folder
CONFIG_FILE=${2:-"config_classification.yaml"}

#################################################################################
export PYTHONPATH=".:${PYTHONPATH}"
export TIDL_TOOLS_PATH="../edgeai-benchmark/tools/tidl_tools_package/${TARGET_SOC}/tidl_tools"
export LD_LIBRARY_PATH="${TIDL_TOOLS_PATH}:${LD_LIBRARY_PATH}"


#################################################################################
# environement variable to help shape exchange between TIDL and onnxruntime
export TIDL_RT_ONNX_VARDIM="1"

#################################################################################
# print some settings
echo "Number of AVX cores detected in PC: ${CPUINFO_NUM_AVX_CORES}"
echo "AVX compilation speedup in PC     : ${TIDL_RT_AVX_REF}"
echo "Target device                     : ${TARGET_SOC}"
echo "PYTHONPATH                        : ${PYTHONPATH}"
echo "TIDL_TOOLS_PATH                   : ${TIDL_TOOLS_PATH}"
echo "LD_LIBRARY_PATH                   : ${LD_LIBRARY_PATH}"

#################################################################################
python3 ./scripts/run_modelmaker.py ${CONFIG_FILE} --target_device ${TARGET_SOC}
