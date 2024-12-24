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
TARGET_SOC=AM68A

# leave this as pc - no change needed
# pc: for model compilation and inference on PC, evm: for model inference on EVM
# after compilation, run_package_artifacts_evm.sh can be used to format and package the compiled artifacts for evm
TARGET_MACHINE=pc

# launch the python script with debugpy for remote attach
DEBUG=false
HOSTNAME=$(hostname)
PORT=5678

##################################################################
CMD_ARGS=()
for arg in "$@"
do 
    case "$arg" in
        "TDA4VM"|"AM68A"|"AM69A"|"AM62A"|"AM67A"|"AM62"|"None")
            TARGET_SOC=$arg
            ;;
        "-d"|"--debug")
            DEBUG=true
            ;;
        "-h"|"--help")
            cat << EOF
Usage: $0 [OPTIONS] [TARGET_SOC]
This script sets up the environment and runs benchmarking on x86 PC for a specified target device by calling the following:
    ./scripts/benchmark_modelzoo.py
    ./scripts/generate_report.py

For more precise configuration of benchmarking, see the CLI options available within ./scripts/benchmark_modelzoo.py.

Options:
-d, --debug     Launch the Python script with debugpy for remote attach.
-h, --help      Display this help message and exit.

TARGET_SOC:
Specify the target device. Use one of: TDA4VM, AM62A, AM68A, AM69A. Defaults to TDA4VM.
Note: Until r8.5, only TDA4VM was supported.

Debug Mode:
If debug mode is enabled, the script will wait for a debugpy to attach at ${HOSTNAME}:${PORT}.
See https://code.visualstudio.com/docs/python/debugging#_example for more info on using debugpy attach with VS Code.

Example:
$0 # defaults to TDA4VM, no debug
$0 [-d|--debug] AM62A # select device with debug
EOF
            exit 0
            ;;
        *) # Catch-all
            CMD_ARGS+=("$arg")
            ;;
    esac
done

echo "TARGET_SOC:     ${TARGET_SOC}"
echo "TARGET_MACHINE: ${TARGET_MACHINE}"
echo "DEBUG MODE:     ${DEBUG} @ ${HOSTNAME}:${PORT}"

echo "=> Recommend to use the alternate script run_benchmarks_parallel_pc.sh"
echo "   to run multiple models in parallel in PC. For example:"
echo "   run_benchmarks_parallel_pc.sh AM68A --parallel_process 8"
##################################################################

# set environment variables
# also point to the right type of artifacts (pc or evm)
source run_set_env.sh ${TARGET_SOC} ${TARGET_MACHINE}

# specify one of the following - additional options can be changed inside the yaml
# SETTINGS=settings_infer_on_evm.yaml
SETTINGS=settings_import_on_pc.yaml
##################################################################

PYARGS1="./scripts/benchmark_modelzoo.py ${SETTINGS} ${CMD_ARGS[@]} --target_device ${TARGET_SOC} --run_inference False"
PYARGS2="./scripts/benchmark_modelzoo.py ${SETTINGS} ${CMD_ARGS[@]} --target_device ${TARGET_SOC} --run_import False"
PYARGS3="./scripts/generate_report.py ${SETTINGS}"
PYDEBUG="python3 -m debugpy --listen ${HOSTNAME}:${PORT} --wait-for-client"

# EX: configure benchmarking with other alternative runtimes, besides ONNXRT
# python3 ./scripts/benchmark_modelzoo.py ${settings_file}  --target_device ${TARGET_SOC} \
#        --session_type_dict {'onnx': 'tvmdlr', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'} \
#        --model_selection onnx  --run_inference False

# python3 ./scripts/benchmark_modelzoo.py ${settings_file}  --target_device ${TARGET_SOC} \
#        --session_type_dict {'onnx': 'tvmdlr', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'} \
#        --model_selection onnx

echo "==================================================================="
if $DEBUG
then
    # Launch script 1, waiting for debugger attachment.
    echo "Waiting for attach @ ${HOSTNAME}:${PORT} to debug the following:"
    echo ${PYARGS1} 
    echo "See --help for more info."
    ${PYDEBUG} ${PYARGS1}
    [ $? -ne 0 ] && exit # Continue only on prior success.
    echo "-------------------------------------------------------------------"

    # Launch script 2, waiting for debugger attachment.
    echo "Waiting for attach @ ${HOSTNAME}:${PORT} to debug the following:"
    echo ${PYARGS2} 
    echo "See --help for more info."
    ${PYDEBUG} ${PYARGS2}            
    [ $? -ne 0 ] && exit # Continue only on prior success.
    echo "-------------------------------------------------------------------"

    # Launch script 3, waiting for debugger attachment.
    echo "Waiting for attach @ ${HOSTNAME}:${PORT} to debug the following:"
    echo ${PYARGS3} 
    echo "See --help for more info."
    ${PYDEBUG} ${PYARGS3}            
    echo "-------------------------------------------------------------------"
else
    # Launch script 1.
    python3 ${PYARGS1}
    [ $? -ne 0 ] && exit # Continue only on prior success.
    echo "-------------------------------------------------------------------"

    # Launch script 2.
    python3 ${PYARGS2}
    [ $? -ne 0 ] && exit # Continue only on prior success.
    echo "-------------------------------------------------------------------"

    # Launch script 3.
    python3 ${PYARGS3}
    echo "-------------------------------------------------------------------"
fi
echo "==================================================================="
