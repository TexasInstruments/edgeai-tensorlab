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
# setup the environment

echo "==================================================================="
DATE_TIME=$(date +'%Y%m%d-%H%M%S')
LOG_FILE=./work_dirs/run_benchmarks_"$DATE_TIME".log
exec &> >(tee -a "$LOG_FILE")
echo Logging output to: "$LOG_FILE"
echo "-------------------------------------------------------------------"

echo "==================================================================="
echo "IMPORTANT: make sure that these environment variables are correct."
echo "-------------------------------------------------------------------"

# make sure current directory is visible for python import
#echo "Setting PYTHONPATH"
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

#echo "Setting PSDK_BASE_PATH"
export PSDK_BASE_PATH="./dependencies/ti-processor-sdk-rtos"
export PSDK_BASE_PATH=$(realpath -s ${PSDK_BASE_PATH})
echo "PSDK_BASE_PATH=${PSDK_BASE_PATH}"

# Note: if the following fails to find the correct tidl path, assign it explicitly
# To know if the correct tidl path is found, see what is printed from the following echo
#echo "Setting TIDL_BASE_PATH"
TIDL_BASE_PATH=$(find "${PSDK_BASE_PATH}/" -maxdepth 1 |grep "/tidl_")
TIDL_BASE_PATH=$(realpath -s ${TIDL_BASE_PATH})
export TIDL_TOOLS_PATH=${TIDL_BASE_PATH}/"tidl_tools"
echo "TIDL_TOOLS_PATH=${TIDL_TOOLS_PATH}"

#echo "Setting TIDL_RT_PERFSTATS"
export TIDL_RT_PERFSTATS="1"
echo "TIDL_RT_PERFSTATS=${TIDL_RT_PERFSTATS}"

#echo "Setting ARM64_GCC_PATH"
export ARM64_GCC_PATH=$(find "${PSDK_BASE_PATH}/" -maxdepth 1 |grep "/gcc-arm-")
export ARM64_GCC_PATH=$(realpath -s ${ARM64_GCC_PATH})
echo "ARM64_GCC_PATH=${ARM64_GCC_PATH}"

#echo "Setting LD_LIBRARY_PATH"
import_path="${TIDL_BASE_PATH}/ti_dl/utils/tidlModelImport/out"
rt_path="${TIDL_BASE_PATH}/ti_dl/rt/out/PC/x86_64/LINUX/release"
tfl_delegate_path="${TIDL_BASE_PATH}/ti_dl/tfl_delegate/out/PC/x86_64/LINUX/release"
onnxrt_ep_path="${TIDL_BASE_PATH}/ti_dl/onnxrt_EP/out/PC/x86_64/LINUX/release"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${import_path}:${rt_path}:${tfl_delegate_path}:${onnxrt_ep_path}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "-------------------------------------------------------------------"

# increase the stack size as it can help in some models
ulimit -s 32768

# specify one of the following - additional options can be changed inside the yaml
#settings_file=accuracy_import_for_j7.yaml
#settings_file=accuracy_infer_on_j7.yaml
#settings_file=accuracy_import_infer_pc.yaml
settings_file=accuracy_import_infer_pc.yaml

echo "==================================================================="
# run all the shortlisted models with these settings
python3 ./scripts/benchmark_accuracy.py ${settings_file} \
        --session_type_dict {'onnx': 'onnxrt', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'}
echo "-------------------------------------------------------------------"

echo "==================================================================="
## run few selected models with these settings
python3 ./scripts/benchmark_accuracy.py ${settings_file} \
        --session_type_dict {'onnx': 'tvmdlr', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'} \
        --task_selection classification segmentation \
        --model_selection onnx
echo "-------------------------------------------------------------------"

echo "==================================================================="
# generate the final report with results for all the artifacts generated
python3 ./scripts/generate_report.py
echo "-------------------------------------------------------------------"
