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
# until r8.5: TDA4VM
# from r8.6 onwards use one of: AM62A AM68A AM69A TDA4VM
TARGET_SOC=TDA4VM

# pc: for model compilation and inference on PC, evm: for model inference on EVM
TARGET_MACHINE=pc

##################################################################
for arg in "$@"
do 
    case "$arg" in
        "TDA4VM"|"AM68A"|"AM69A"|"AM62A"|"AM67A"|"AM62")
            TARGET_SOC=$arg
            ;;
        "-h"|"--help")
            cat << EOF
Usage: $0 [OPTIONS] [TARGET_SOC]
This script

Options:
-h, --help      Display this help message and exit.

TARGET_SOC:
Specify the target device. Use one of: TDA4VM, AM62A, AM68A, AM69A. Defaults to TDA4VM.
Note: Until r8.5, only TDA4VM was supported.  

Example:
$0 # defaults to TDA4VM
$0 AM62A # select device
EOF
            exit 0
            ;;
    esac
done

echo "TARGET_SOC:     ${TARGET_SOC}"
echo "TARGET_MACHINE: ${TARGET_MACHINE}"
##################################################################

pip3 install jupyter

##################################################################
# set environment variables
# also point to the right type of artifacts (pc or evm)
source run_set_env.sh ${TARGET_SOC} ${TARGET_MACHINE}

# run the script
jupyter notebook --ip=localhost
