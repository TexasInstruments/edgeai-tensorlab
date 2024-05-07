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
TARGET_SOC=${1:-TDA4VM}
TARGET_MACHINE=${2:-pc}

##################################################################
for arg in "$@"
do 
    case "$arg" in
        "-h"|"--help")
            cat << EOF
Usage: $0 [TARGET_SOC] [TARGET_MACHINE] [OPTIONS]
This script sets up TIDL tools for usage on a target machine, "pc" or "evm". 
This is a helper script called by the main entrypoint scripts. The script expects
the compilation artifacts to be located at "./work_dirs/modelartifacts/${TARGET_SOC}/8bits". 

Options:
-d, --debug     Launch the Python script with debugpy for remote attach.
-h, --help      Display this help message and exit.

TARGET_SOC:
Specify the target device. Use one of: TDA4VM, AM62A, AM68A, AM69A. Defaults to TDA4VM.
Note: Until r8.5, only TDA4VM was supported.  

TARGET_MACHINE:
Specify the target machine to run benchmarking on: pc or evm.

Example:
$0 # defaults to TDA4VM, no debug
$0 AM62A pc # select AM69A on pc
EOF
            exit 0
            ;;
    esac
done
##################################################################

##################################################################
# for onnxruntime and tflite_runtime, the artifacts are same for pc and evm devices
# however for tvmdlr, there are two sets of artifacts - one for pc and one for evm device
# deploy_lib.so.pc is for pc and deploy_lib.so.evm is for evm device
# a symbolic link called deploy_lib.so needs to be created, depending on where we plan to run the inference.
# this can be done after the import has been done and artifacts generated.
# by default it points to deploy_lib.so.pc, so nothing needs to be done for inference on pc

if [[ $# -ne 2 ]]; then
  echo "Please provide exactly two arguments - TARGET_SOC and TARGET_MACHINE"
  echo "TARGET_MACHINE can be either pc or evm"
  exit 1
fi

# "evm" or "pc"
target_machine=${TARGET_MACHINE}
if [ "$artifacts_base" = "" ]; then
  artifacts_base="./work_dirs/modelartifacts/${TARGET_SOC}/8bits"
fi
artifacts_folders=$(find "${artifacts_base}/" -maxdepth 1 |grep "_tvmdlr_")
cur_dir=$(pwd)

declare -a artifact_files=("deploy_lib.so" "deploy_graph.json" "deploy_params.params")

for artifact_folder in ${artifacts_folders}
do
  echo "Entering: ${artifact_folder}"
  cd ${artifact_folder}/"artifacts"
  for artifact_file in "${artifact_files[@]}"
  do
    if [[ -f ${artifact_file}.${target_machine} ]]; then
      echo "creating symbolic link to ${artifact_file}.${target_machine}"
      ln -snf ${artifact_file}.${target_machine} ${artifact_file}
    fi
  done
  cd ${cur_dir}
done

# TIDL_ARTIFACT_SYMLINKS is used to indicate that the symlinks have been set to evm
# affects only artifacts created by/for TVM/DLR
export TIDL_ARTIFACT_SYMLINKS=1
