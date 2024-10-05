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
# target_device - use one of: TDA4VM AM62A AM68A AM69A
# (Note: until r8.5 only TDA4VM was supported)
TARGET_SOC=${1:-TDA4VM}

# pc: for model compilation and inference on PC, evm: for model inference on EVM
TARGET_MACHINE=evm


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
settings_file=settings_infer_on_evm.yaml

echo "==================================================================="
# run all the shortlisted models with these settings
python3 ./scripts/benchmark_modelzoo.py ${settings_file} --target_device ${TARGET_SOC}
echo "-------------------------------------------------------------------"

#echo "==================================================================="
### run few selected models with these settings
#python3 ./scripts/benchmark_modelzoo.py ${settings_file} \
#        --session_type_dict {'onnx': 'tvmdlr', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'} \
#        --task_selection classification segmentation \
#        --model_selection onnx
#echo "-------------------------------------------------------------------"

echo "==================================================================="
# generate the final report with results for all the artifacts generated
python3 ./scripts/generate_report.py ${settings_file}
echo "-------------------------------------------------------------------"
