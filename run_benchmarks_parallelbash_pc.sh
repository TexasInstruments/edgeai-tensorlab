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
# to run background jobs
set -m
#set -e

##################################################################
# target_device - use one of: TDA4VM AM62A AM68A AM69A
# (Note: until r8.5 only TDA4VM was supported)
TARGET_SOC=${1:-TDA4VM}

# for parallel execution on pc only (cpu or gpu).
# number fo parallel processes to run.
# for example 8 will mean 8 models will run in parallel
# for example 1 will mean one model will run (but in a separae processs from that of the main process)
# null will mean one process will run, in the same process as the main
NUM_PARALLEL_PROCESSES=${NUM_PARALLEL_PROCESSES:-8}

# for parallel execution on CUDA/gpu. if you don't have CUDA/gpu, these don't matter
# if you have gpu's these wil be used for CUDA_VISIBLE_DEVICES. eg. specify 4 will use the gpus: 0,1,2,3
# it can also be specified as a list with actual GPU ids, instead of an integer: [0,1,2,3]
# important note: to use CUDA/gpu, CUDA compiled TIDL (tidl_tools) is required.
NUM_PARALLEL_DEVICES=${NUM_PARALLEL_DEVICES:-4}

# leave this as pc - no change needed
# pc: for model compilation and inference on PC, evm: for model inference on EVM
# after compilation, run_package_artifacts_evm.sh can be used to format and package the compiled artifacts for evm
TARGET_MACHINE=pc

echo #############################################################
echo "target_device/SOC: ${TARGET_SOC}"
echo "Pass the appropriate commandline argument to use another target_device"

##################################################################
# set environment variables
# also point to the right type of artifacts (pc or evm)
source run_set_env.sh ${TARGET_SOC} ${TARGET_MACHINE}

##################################################################
# specify one of the following settings - options can be changed inside the yaml
#settings_file=settings_infer_on_evm.yaml
#settings_file=settings_import_on_pc.yaml
settings_file=settings_import_on_pc.yaml

echo "-------------------------------------------------------------------"
function f_num_running_jobs() {
  n_jobs=$(jobs -r | wc -l)
  echo $n_jobs
}

function f_list_running_jobs() {
  proc_id_jobs=$(pgrep -P $$ -d " ")
  echo $proc_id_jobs
}

function run_model() {
    python3 ./scripts/benchmark_modelzoo.py $@ --run_inference 0 && \
    python3 ./scripts/benchmark_modelzoo.py $@ --run_import 0
}

echo "-------------------------------------------------------------------"
proc_id=$$
# run all the shortlisted models with these settings
modelartifacts_folder="./work_dirs/modelartifacts"
models_list_file="${modelartifacts_folder}/benchmarks_models_list.txt"
mkdir -p ${modelartifacts_folder}
python3 ./scripts/generate_models_list.py ${settings_file} --target_device ${TARGET_SOC} --models_list_file $models_list_file --dataset_loading False ${@:2}
num_lines=$(wc -l < ${models_list_file})
echo $num_lines

parallel_device=0
for model_id in $(cat ${models_list_file}); do
  while [ $(f_num_running_jobs) -ge $NUM_PARALLEL_PROCESSES ]; do
      timestamp=$(date +'%Y%m%d-%H%M%S')
      num_running_jobs=$(f_num_running_jobs)
      echo -ne "\r\e[0K proc_id:$proc_id timestamp:$timestamp num_running_jobs:$num_running_jobs"
      sleep 10
  done
  timestamp=$(date +'%Y%m%d-%H%M%S')
  num_running_jobs=$(f_num_running_jobs)
  parallel_device=$((parallel_device+1))
  parallel_device=$((parallel_device%NUM_PARALLEL_DEVICES))
  echo " "
  echo " ==============================================================="
  echo " proc_id:$proc_id timestamp:$timestamp num_running_jobs:$num_running_jobs running model_id:$model_id on parallel_device:$parallel_device"
  # --parallel_processes 0 is used becuase we don't want to create another process inside.
  # --parallel_devices null is used becuase CUDA_VISIBLE_DEVICES is set here itself - no need to be set inside again
  CUDA_VISIBLE_DEVICES="$parallel_device" run_model "${settings_file}"  --target_device "${TARGET_SOC}" --model_selection "${model_id}" --parallel_processes 0 --parallel_devices null ${@:2} &
  sleep 1
  echo " ==============================================================="
done

echo "-------------------------------------------------------------------"
while [ $(f_num_running_jobs) -ge 1 ]; do
    timestamp=$(date +'%Y%m%d-%H%M%S')
    num_running_jobs=$(f_num_running_jobs)
    echo -ne "\r\e[0K proc_id:$proc_id timestamp:$timestamp num_running_jobs:$num_running_jobs"
    sleep 10
done

echo "-------------------------------------------------------------------"
# generate the final report with results for all the artifacts generated
python3 ./scripts/generate_report.py ${settings_file}
echo "-------------------------------------------------------------------"
