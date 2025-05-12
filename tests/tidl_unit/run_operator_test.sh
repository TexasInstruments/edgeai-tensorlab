#!/usr/bin/env bash

###############################################################################
# Script: run_operator_test.sh
# Description: Run TIDL operator tests for a specified SoC with and without
# neural compiler (NC), then generate comparison CSV reports.
# Usage: ./run_operator_test.sh <SOC>
# <SOC> must be one of: AM62A, AM67A, AM68A, AM69A, TDA4VM
###############################################################################

ALLOWED_SOC=("AM62A" "AM67A" "AM68A" "AM69A" "TDA4VM")

usage() {
cat <<EOF
Usage: $0 <SOC>
<SOC> must be one of: ${ALLOWED_SOC[*]}
EOF
exit 1
}

# Validate arguments
if [ $# -ne 1 ]; then
echo "Error: Exactly one argument expected." >&2
usage
fi

SOC=$1

# Check SOC validity
if ! printf '%s\n' "${ALLOWED_SOC[@]}" | grep -Fxq "$SOC"; then
echo "Error: Invalid SoC '$SOC'." >&2
usage
fi

###############################################################################
# Configuration
###############################################################################
tools_path="<tidl_tools tarball path here>"
OPERATORS=()
# Single operator like Max - OPERATORS=("Max")
# Multi operator like Softmax, Convolution & Sqrt - OPERATORS=("Softmax" "Convolution" "Sqrt")
# Full suite - OPERATORS=()
#######################################################################


###############################################################################
# Prepare environment and tools
###############################################################################
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

###############################################################################
# Prepare test directories
###############################################################################
rm -rf "$path_edge_ai_benchmark/tests/tidl_unit/operator_test_reports"
mkdir -p "$path_edge_ai_benchmark/tests/tidl_unit/operator_test_reports"

path_reports="$path_edge_ai_benchmark/tests/tidl_unit/operator_test_reports"

if [ -z "$OPERATORS" ]; then
     for D in $(find $path_edge_ai_benchmark/tests/tidl_unit/tidl_unit_test_data/operator/ -mindepth 1 -maxdepth 1 -type d) ; do
        name=`basename $D`
        OPERATORS+=("$name")
    done
fi

###############################################################################
# Run tests for each operator
###############################################################################
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
done

###############################################################################
# Generate CSV reports
###############################################################################
mv operator_test_reports/ report_script/ 

cd report_script
# python3 comparison_test_report_csv.py
python3 absolute_test_report_csv.py
python3 customer_test_report_csv.py

cd ../
rm -rf operator_test_report_csv
mkdir operator_test_report_csv
# mv report_script/comparison_test_reports/ operator_test_report_csv/
mv report_script/complete_test_reports/ operator_test_report_csv/
mv report_script/customer_test_reports/ operator_test_report_csv/

rm -rf operator_test_report_html
mkdir operator_test_report_html
mv report_script/operator_test_reports/* operator_test_report_html/
rm -rf report_script/operator_test_reports/