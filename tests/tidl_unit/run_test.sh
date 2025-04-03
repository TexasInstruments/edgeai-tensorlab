#!/usr/bin/env bash

# Copyright (c) 2018-2025, Texas Instruments
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
ROOT_DIR=$(realpath "$(dirname -- "$0")")
cd $ROOT_DIR

usage() {
echo \
"Usage:
    Helper script to invoke pytest for tidl unit tests

    To change the custom configuration, modify ./scripts/benchmark_custom.py.

    Options:
    --test_suite        Test suite. Allowed values are (operator)
    --run_compile       Run model compilation test. Allowed values are (0,1). Default=1.
    --run_infer         Run model inference test. Allowed values are (0,1). Default=1.
    --tidl_offload      Enable TIDL Offload. Allowed values are (0,1). Default=1.
    --tests             Specify tests name. If null, will run all test based on test_suite. Default=null.
                        TEST_SUITE:
                            operator: You can specify comma seperated operator name (Ex: Convolution) or specific test (Ex: Softmax_1) 

    Example:
    TEST_SUITE:
        operator: ./run_test.sh --test_suite=operator --tests=Convolution,Softmax_1,Unsqueeze,Flatten_3 --run_compile=1 --run_infer=1
                   This will run all tests under Convolution and Unsqueeze and also Softmax_1 and Flatten_3 test.
    "
}

test_suite=""
tests=""
run_compile=""
run_infer=""
tidl_offload=""

while [ $# -gt 0 ]; do
        case "$1" in
        --test_suite=*)
        test_suite="${1#*=}"
        ;;
        --tests=*)
        tests="${1#*=}"
        ;;
        --run_compile=*)
        run_compile="${1#*=}"
        ;;
        --run_infer=*)
        run_infer="${1#*=}"
        ;;
        --tidl_offload=*)
        tidl_offload="${1#*=}"
        ;;
        --help)
        usage
        exit
        ;;
        *)
        echo "Error: Invalid argument $1 !!"
        usage
        exit
        ;;
        esac
        shift
done

if [[ "$test_suite" != "operator" ]]; then
    echo "[ERROR]: TEST_SUITE: $test_suite is not allowed."
    echo "         Allowed values are (operator)"
    exit 1
fi

if [[ "$run_compile" == "" ]]; then
   run_compile="1"
fi
if [[ "$run_infer" == "" ]]; then
   run_infer="1"
fi
if [[ "$tidl_offload" == "" ]]; then
   tidl_offload="1"
fi

if [ "$run_compile" != "1" -a "$run_compile" != "0" ]; then
    echo "[ERROR]: RUN_COMPILE: $run_compile is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$run_infer" != "1" -a "$run_infer" != "0" ]; then
    echo "[ERROR]: RUN_INFER: $run_infer is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$tidl_offload" != "1" -a "$tidl_offload" != "0" ]; then
    echo "[ERROR]: TIDL_OFFLOAD: $tidl_offload is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi

echo "##################################################################"
echo "TEST_SUITE:   ${test_suite}"
echo "TESTS:        ${tests}"
echo "RUN_COMPILE:  ${run_compile}"
echo "RUN_INFER:    ${run_infer}"
echo "TIDL_OFFLOAD: ${tidl_offload}"
echo "##################################################################"
echo

test_args=""
extra_args=""

##################################################################
# OPERATOR TEST SUITE START
if [[ "$test_suite" == "operator" ]]; then

echo "##################################################################"

OPERATOR_ROOT_FOLDER="${ROOT_DIR}/tidl_unit_test_data/operator"
if [ ! -d "$OPERATOR_ROOT_FOLDER" ]; then
  echo "[ERROR]: $OPERATOR_ROOT_FOLDER does not exist."
  echo "         All the data for operator suite needs to present in this directory. Refer to README for more information."
fi

IFS=',' read -r -a test_array <<< "$tests"
all_test=()
if [ -z "$test_array" ]; then
    echo "[WARNING]: No tests specified. Running all tests under $OPERATOR_ROOT_FOLDER."
    TOTAL=$(find $OPERATOR_ROOT_FOLDER -mindepth 2 -maxdepth 2  -type d | wc -l)
    echo "Total tests:  ${TOTAL}"
else
    for test in "${test_array[@]}"
    do
        # Check if provided test is a directory
        if [ -d "$OPERATOR_ROOT_FOLDER/$test" ]; then
            counter=0
            for D in $(find $OPERATOR_ROOT_FOLDER/$test -mindepth 1 -maxdepth 1 -type d) ; do
                name=`basename $D`
                all_test+=("$name")
                counter=$((counter+1))
            done
            echo "Found ${counter} tests for $test"
        else
            REL_DIR=$(find $OPERATOR_ROOT_FOLDER -mindepth 2 -maxdepth 2  -type d -name $test)
            if [ -z "$REL_DIR" ]; then
                echo "Found 0 test for $test. Skipping."
            else
                echo "Found 1 test for $test"
                all_test+=($test)
            fi
        fi
    done
    echo "Total tests:  ${#all_test[@]}"
fi

echo "##################################################################"
echo

if (( ${#all_test[@]} )); then
    for test in "${all_test[@]}"
    do
        test_args="${test_args} test_tidl_unit.py::test_tidl_unit_operator[$test]"
    done
fi

fi
# OPERATOR TEST SUITE END
##################################################################


##################################################################
# RUN PYTEST START

# COMPILATION
if [[ "$tidl_offload" == "0" ]]; then
   extra_args="--disable-tidl-offload"
fi



if [[ "$run_compile" == "1" ]]; then
    echo "##################################################################"
    echo "Running Compilation..."
    echo
    pytest ${test_args} ${extra_args}
    echo "Compilation test done"
    echo "##################################################################"
    echo
fi

if [[ "$run_infer" == "1" ]]; then
    echo "##################################################################"
    echo "Running Inference..."
    echo
    pytest ${test_args} ${extra_args} --run-infer
    echo "Inference test done"
    echo "##################################################################"
    echo
fi



# RUN PYTEST END
##################################################################

