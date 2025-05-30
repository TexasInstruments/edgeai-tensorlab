#!/bin/bash
################################################################################
TEST_NAME=${TEST_NAME:-""}
TEST_COMMAND=${TEST_COMMAND,""}
TIMEOUT=${TIMEOUT,""}

cd ~/edgeai-benchmark/tests/tidl_unit

mkdir -p logs
rm -rf logs/report*
rm -rf ${TEST_NAME}.html

PYTEST_ARGS="--run-infer -n 1 --exit-on-critical-error"

if [ "$TIMEOUT" != "" ]; then
    PYTEST_ARGS="${PYTEST_ARGS} --timeout ${TIMEOUT}"
fi
pytest $TEST_COMMAND $PYTEST_ARGS
cd logs
mv report*.html ${TEST_NAME}.html
cd ../
echo "END_OF_MODEL_INFERENCE"
