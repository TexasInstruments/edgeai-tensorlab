#!/usr/bin/env bash

: '
Invoke this script by running:
./run_operator_test <SOC>
where SOC can be AM62A, AM67A, AM68A, AM69A, TDA4VM 
'

SOC=$1
###### Make sure to update these variables with the actual paths ######
tools_path="<tidl_tools tarball path here>"
#######################################################################

current_dir="$PWD"
path_edge_ai_benchmark="$current_dir/../.."
cd "$path_edge_ai_benchmark" || { echo "Failed to cd to $path_edge_ai_benchmark";}
source ./run_set_env.sh "$SOC"

rm -rf "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC"/*
tools_basename=$(basename "$tools_path")
cp "$tools_path" "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC/"
cd "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC"
tar -xzvf "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC/$tools_basename"
cp -r "$path_edge_ai_benchmark/tools/tidl_tools_package/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu" "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC/tidl_tools/"
cp -r "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC/tidl_tools/ti_cnnperfsim.out" "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC"

cd "$path_edge_ai_benchmark/tests/tidl_unit"

rm -rf "$path_edge_ai_benchmark/tests/tidl_unit/operator_test_reports"
mkdir -p "$path_edge_ai_benchmark/tests/tidl_unit/operator_test_reports"

path_reports="$path_edge_ai_benchmark/tests/tidl_unit/operator_test_reports"

# Add specific operators to run in the array, else it will run all test under tidl_unit_test_data/operator
# Example: OPERATORS=("Softmax" "Convolution" "Sqrt")
OPERATORS=()
if [ -z "$OPERATORS" ]; then
     for D in $(find $path_edge_ai_benchmark/tests/tidl_unit/tidl_unit_test_data/operator -mindepth 1 -maxdepth 1 -type d) ; do
        name=`basename $D`
        OPERATORS+=("$name")
    done
fi

for operator in "${OPERATORS[@]}"
do
    logs_path=$path_reports/$operator
    rm -rf $logs_path
    mkdir -p $logs_path
    echo "Logs will be saved to: $logs_path"

    echo "########################################## $operator TEST (WITH NC) ######################################"
    rm -rf work_dirs/*

    cp -rp "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC/ti_cnnperfsim.out" "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC/tidl_tools/"

    rm -rf logs/*
    ./run_test.sh --test_suite=operator --tests=$operator --run_infer=0
    cp logs/*.html "$logs_path/compile_with_nc.html"

    rm -rf logs/*
    ./run_test.sh --test_suite=operator --tests=$operator --run_compile=0
    cp logs/*.html "$logs_path/infer_with_nc.html"

    echo "########################################## $operator TEST (WITHOUT NC) ######################################"
    rm -rf work_dirs/*

    rm -rf "$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC/tidl_tools/ti_cnnperfsim.out"

    rm -rf logs/*
    ./run_test.sh --test_suite=operator --tests=$operator --run_infer=0
    cp logs/*.html "$logs_path/compile_without_nc.html"

    rm -rf logs/*
    ./run_test.sh --test_suite=operator --tests=$operator --run_compile=0
    cp logs/*.html "$logs_path/infer_without_nc.html"

    ##############################################################################################
done

# Generate summary report
python3 report_summary_generation.py