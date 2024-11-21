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
TARGET_SOC=None

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
        "-d"|"--debug")
            DEBUG=true
            ;;
        "TDA4VM"|"AM68A"|"AM69A"|"AM62A"|"AM67A"|"AM62"|"None")
            TARGET_SOC=$arg
            ;;
        "-h"|"--help")
            cat << EOF
Usage: $0 [OPTIONS] [TARGET_SOC]
This script generates a CSV summary of the results of a benchmarking session by calling:
    ./scripts/generate_report.py

Options:
-d, --debug     Launch the Python script with debugpy for remote attach.
-h, --help      Display this help message and exit.

TARGET_SOC:
Specify the target device (optional). Use one of: TDA4VM, AM62A, AM68A, AM69A, AM67A.
Default behaviour is to report all results present in modelartifacts_path

Debug Mode:
If debug mode is enabled, the script will wait for a debugpy to attach at ${HOSTNAME}:${PORT}.
See https://code.visualstudio.com/docs/python/debugging#_example for more info on using debugpy attach with VS Code.

Example:
$0 # defaults to None, no debug
$0 [-d|--debug] AM62A # select device with debug
EOF
            exit 0
            ;;
        *) # Catch-all
            CMD_ARGS+=("$arg")
            ;;
    esac
done
##################################################################

# set environment variables
# also point to the right type of artifacts (pc or evm)
source run_set_env.sh ${TARGET_SOC} ${TARGET_MACHINE}

# specify one of the following - additional options can be changed inside the yaml
# SETTINGS=settings_infer_on_evm.yaml
SETTINGS=settings_import_on_pc.yaml
##################################################################

PYARGS="./scripts/generate_report.py ${SETTINGS} ${CMD_ARGS[@]} --target_device ${TARGET_SOC}"
# add the following to report perfsim results as well.
#--report_perfsim True

PYDEBUG="python3 -m debugpy --listen ${HOSTNAME}:${PORT} --wait-for-client"
echo "==================================================================="

if $DEBUG
then
    echo "Waiting for attach @ ${HOSTNAME}:${PORT} to debug..."
    echo "See --help for more info."
    echo "${PYDEBUG} ${PYARGS}"
    ${PYDEBUG} ${PYARGS}
    echo "-------------------------------------------------------------------"
else
    echo "python3 ${PYARGS}"
    python3 ${PYARGS}
    echo "-------------------------------------------------------------------"
fi
echo "==================================================================="
