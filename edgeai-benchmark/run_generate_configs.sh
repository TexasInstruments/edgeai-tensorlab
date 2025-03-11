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
# Generate confg yaml files and write into the same location as the the models
# (i.e. inside the edgeai-modelzoo or wherever the model_path is pointing to)
# It will also write a configs.yaml at the location of settings.models_path
# that lists all such config yaml files.
# This configs.yaml file or individial config yaml files can be given to
# run_benchmrks_pc.sh OR run_benchmarks_parallel_pc.sh OR directly to scripts/benchmark_modelzoo.py
# to import or infer the the model using the parameters specified in that config file.

##################################################################
# target_device - use one of: TDA4VM AM62A AM68A AM69A AM67A AM62
TARGET_SOC=${1:-AM68A}


##################################################################
# set environment variables
# also point to the right type of artifacts (pc or evm)
source ./run_set_env.sh ${TARGET_SOC} ${TARGET_MACHINE}

##################################################################
# specify one of the following settings - options can be changed inside the yaml
#SETTINGS_FILE=settings_infer_on_evm.yaml
#SETTINGS_FILE=settings_import_on_pc.yaml
SETTINGS_FILE=settings_import_on_pc.yaml

echo "-------------------------------------------------------------------"
# generate the final report with results for all the artifacts generated
python3 ./scripts/generate_configs.py ${SETTINGS_FILE} --target_device ${TARGET_SOC} ${@:2}
echo "-------------------------------------------------------------------"
