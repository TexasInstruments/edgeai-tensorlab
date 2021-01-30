#!/usr/bin/env bash

##################################################################
# setup the environment

#set LD_LIBRARY_PATHS
tidl_path='./dependencies/c7x-mma-tidl'

tidl_path=$(readlink -- "${tidl_path}")
import_path=":${tidl_path}/ti_dl/utils/tidlModelImport/out"
rt_path=":${tidl_path}/ti_dl/rt/out/PC/x86_64/LINUX/release"

echo "Setting TIDL_BASE_PATH"
export TIDL_BASE_PATH=${tidl_path}
echo "TIDL_BASE_PATH=" ${TIDL_BASE_PATH}

echo "Setting LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${import_path}:${rt_path}"
echo "LD_LIBRARY_PATH=" ${LD_LIBRARY_PATH}

##################################################################
#python ./scripts/benchmark_classification.py
python ./scripts/example_classification.py


