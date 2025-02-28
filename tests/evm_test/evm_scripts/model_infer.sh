#!/bin/bash
set -e
################################################################################
TARGET_DEVICE=${1:-"AM68A"}
TIMEOUT=${2:-"120"}
GENERATE_REPORT=${3:-"0"}
MODEL_SELECTION=${4:-"null"}
NUM_FRAMES=${5:-"0"}
MODELARTIFACTS_PATH=${6:-"./work_dirs/modelartifacts/${target_device}"}

EXTRA_ARGS="--experimental_models True --additional_models True"
if [ "$MODEL_SELECTION" != "null" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --model_selection ${MODEL_SELECTION}"
fi
if [ "$NUM_FRAMES" != "0" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --num_frames ${NUM_FRAMES}"
fi

echo "TARGET_DEVICE:${TARGET_DEVICE}"
echo "TIMEOUT:${TIMEOUT}"
echo "EXTRA_ARGS:${EXTRA_ARGS}"

# target Inference Steps
cd ~/edgeai-benchmark
timeout -k 5 ${TIMEOUT} python3 ./scripts/benchmark_modelzoo.py settings_infer_on_evm.yaml --target_device ${TARGET_DEVICE} --modelartifacts_path ${MODELARTIFACTS_PATH} ${EXTRA_ARGS}
if [ $? -eq 124 ]; then
    echo "TIMEDOUT after ${TIMEOUT}s"
    exit 1
fi
if [ "$GENERATE_REPORT" == "1" ]; then
    echo "Generating test report..."
    python3 ./scripts/generate_report.py settings_infer_on_evm.yaml
fi

echo "END_OF_MODEL_INFERENCE"