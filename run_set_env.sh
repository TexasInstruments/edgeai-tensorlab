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
# target_device - use one of: TDA4VM AM62A AM68A AM69A AM67A AM62
TARGET_SOC=${1-AM68A}

# "evm" or "pc"
TARGET_MACHINE=${2:-pc}

#################################################################################
# setup the environment
# source run_setupenv_pc.sh
export TIDL_TOOLS_PATH=$(pwd)/tools/tidl_tools_package/${TARGET_SOC}/tidl_tools
echo "TIDL_TOOLS_PATH=${TIDL_TOOLS_PATH}"

export LD_LIBRARY_PATH="${TIDL_TOOLS_PATH}:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

# needed for TVM compilation
export ARM64_GCC_PATH=$TIDL_TOOLS_PATH/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

#################################################################################
# environement variable to help shape exchange between TIDL and onnxruntime
export TIDL_RT_ONNX_VARDIM="1"

# additional environment variables for detailed performance measurement
export TIDL_RT_DDR_STATS="1"
export TIDL_RT_PERFSTATS="1"

#################################################################################
# optional: check if AVX instructions are available in the machine
# by default AVX is enabled - setting this TIDL_RT_AVX_REF flag to "0" wil disable AVX
CPUINFO_NUM_AVX_CORES=$(cat /proc/cpuinfo|grep avx|wc|tail -n 1|awk '{print $1;}')
if [ ${CPUINFO_NUM_AVX_CORES} -eq 0 ]; then
  export TIDL_RT_AVX_REF="0"
else
  export TIDL_RT_AVX_REF="1"
fi


################################################################################
# tvmdlr artifacts are different for pc and evm device
# point to the right artifact before this script executes
if [ "${ARTIFACTS_BASE_PATH}" = "" ]; then
  ARTIFACTS_BASE_PATH="./work_dirs/modelartifacts/${TARGET_SOC}/8bits"
fi

if [ -d "${ARTIFACTS_BASE_PATH}" ]; then
  echo "INFO: settings the correct symlinks in tvmdlr compiled artifacts"

  artifacts_folders=$(find "${ARTIFACTS_BASE_PATH}/" -maxdepth 1 |grep "_tvmdlr_")
  cur_dir=$(pwd)

  declare -a artifact_files=("deploy_lib.so" "deploy_graph.json" "deploy_params.params")

  for artifact_folder in ${artifacts_folders}
  do
    echo "Entering: ${artifact_folder}"
    cd ${artifact_folder}/"artifacts"
    for artifact_file in "${artifact_files[@]}"
    do
      if [[ -f ${artifact_file}.${TARGET_MACHINE} ]]; then
        echo "creating symbolic link to ${artifact_file}.${TARGET_MACHINE}"
        ln -snf ${artifact_file}.${TARGET_MACHINE} ${artifact_file}
      fi
    done
    cd ${cur_dir}
  done

  # TIDL_ARTIFACT_SYMLINKS is used to indicate that the symlinks have been set to evm
  # affects only artifacts created by/for TVM/DLR
  export TIDL_ARTIFACT_SYMLINKS=1
fi
################################################################################
