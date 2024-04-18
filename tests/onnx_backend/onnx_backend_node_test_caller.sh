#!/bin/bash
# Note: The typical way to call a group of pytest tests would be using the pytest command
#       This script is a workaround to try to get rid of discrepancies observed between running single tests and running large test suites
#            Those discrepancies have not been seen lately (or at least have not been as common) -- delete this script if it doens't prove to be helpful

log_fname=onnx_backend_tests_log.txt
rm $log_fname

for dir in $(python -c "import onnx;import os;print(os.path.join(os.path.dirname(onnx.__file__),'backend/test/data/node/*'))"); do
    cmd="test_onnx_backend.py::test_onnx_backend_node[${dir##*/}] -capture=tee-sys --log-cli-level=DEBUG"
    printf "\n\n$cmd\n" | tee -a $log_fname
    printf "Running import" | tee -a $log_fname
    pytest $cmd 2>&1 | tee -a $log_fname  
    printf "Running inference" | tee -a $log_fname
    pytest $cmd --run-infer 2>&1 | tee -a $log_fname  

done
