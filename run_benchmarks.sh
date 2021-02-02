#!/usr/bin/env bash

##################################################################
# setup the environment

# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}

#set LD_LIBRARY_PATHS
echo "Setting TIDL_BASE_PATH"
export TIDL_BASE_PATH='./dependencies/c7x-mma-tidl'
export TIDL_BASE_PATH=$(readlink -- "${TIDL_BASE_PATH}")
echo "TIDL_BASE_PATH=" ${TIDL_BASE_PATH}

echo "Setting LD_LIBRARY_PATH"
import_path="${TIDL_BASE_PATH}/ti_dl/utils/tidlModelImport/out"
rt_path="${TIDL_BASE_PATH}/ti_dl/rt/out/PC/x86_64/LINUX/release"
tfl_delegate_path="${TIDL_BASE_PATH}/ti_dl/tfl_delegate/out/PC/x86_64/LINUX/release"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${import_path}:${rt_path}:${tfl_delegate_path}"
echo "LD_LIBRARY_PATH=" ${LD_LIBRARY_PATH}

export TIDL_RT_PERFSTATS="1"
echo "TIDL_RT_PERFSTATS=" ${TIDL_RT_PERFSTATS}

##################################################################
python ./scripts/benchmark_classification.py
#python ./scripts/example_classification.py


