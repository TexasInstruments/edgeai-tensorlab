#!/usr/bin/env bash

##################################################################
# setup the environment

# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

echo "Setting PSDK_BASE_PATH"
export PSDK_BASE_PATH="/user/a0393608/files/work/ti/bitbucket/processor-sdk-vision/ti-processor-sdk-rtos-j721e-evm-07_02_00_06"
echo "PSDK_BASE_PATH=${PSDK_BASE_PATH}"

echo "Setting TIDL_BASE_PATH"
export TIDL_BASE_PATH="${PSDK_BASE_PATH}/tidl_j7_01_04_00_08"
echo "TIDL_BASE_PATH=${TIDL_BASE_PATH}"

echo "Setting ARM64_GCC_PATH"
export ARM64_GCC_PATH="${PSDK_BASE_PATH}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
echo "ARM64_GCC_PATH=${ARM64_GCC_PATH}"

echo "Setting LD_LIBRARY_PATH"
import_path="${TIDL_BASE_PATH}/ti_dl/utils/tidlModelImport/out"
rt_path="${TIDL_BASE_PATH}/ti_dl/rt/out/PC/x86_64/LINUX/release"
tfl_delegate_path="${TIDL_BASE_PATH}/ti_dl/tfl_delegate/out/PC/x86_64/LINUX/release"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${import_path}:${rt_path}:${tfl_delegate_path}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

export TIDL_RT_PERFSTATS="1"
echo "TIDL_RT_PERFSTATS=${TIDL_RT_PERFSTATS}"

##################################################################
python ./scripts/benchmark_classification.py
#python ./scripts/benchmark_detection.py
#python ./scripts/example_classification.py


