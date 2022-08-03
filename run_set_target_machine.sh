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


# for onnxruntime and tflite_runtime, the artifacts are same for pc and evm devices
# however for tvmdlr, there are two sets of artifacts - one for pc and one for evm device
# deploy_lib.so.pc is for pc and deploy_lib.so.evm is for evm device
# a symbolic link called deploy_lib.so needs to be created, depending on where we plan to run the inference.
# this can be done after the import has been done and artifacts generated.
# by default it points to deploy_lib.so.pc, so nothing needs to be done for inference on pc

if [[ $# -ne 1 ]]; then
  echo "please provide exactly one argument - either pc or evm"
  exit 1
fi

# "evm" or "pc"
target_machine=$1
if [ "$artifacts_base" = "" ]; then
  artifacts_base="./work_dirs/modelartifacts/8bits"
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
