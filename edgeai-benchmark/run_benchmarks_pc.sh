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

# leave this as pc - no change needed
# pc: for model compilation and inference on PC, evm: for model inference on EVM
# after compilation, run_package_artifacts_evm.sh can be used to format and package the compiled artifacts for evm
TARGET_MACHINE=pc

# settigns model_shortlist will cause only selected models to be run
# in ./configs folder, model configs have a model_shortlist associated with them
# in this script, if --model_shortlist is set to 120, only those models with model_shortlist values <= 120 will run
MODEL_SHORTLIST=${MODEL_SHORTLIST:-120}


echo "TARGET_SOC:     ${TARGET_SOC}"
echo "TARGET_MACHINE: ${TARGET_MACHINE}"

##################################################################
# set environment variables
# also point to the right type of artifacts (pc or evm)
source ./run_set_env.sh ${TARGET_SOC} ${TARGET_MACHINE}

# specify settings yaml file - additional options can be changed inside the yaml
SETTINGS_FILE=settings_import_on_pc.yaml

echo "==================================================================="
python3 ./scripts/benchmark_modelzoo.py ${SETTINGS_FILE} --target_device ${TARGET_SOC} --model_shortlist ${MODEL_SHORTLIST} "${@:2}"
python3 ./scripts/generate_report.py ${SETTINGS_FILE}
echo "==================================================================="

