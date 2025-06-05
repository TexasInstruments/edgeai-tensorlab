#!/usr/bin/env bash

usage() {
echo \
"Usage:
    Helper script to run unit tests operator wise

    Options:
    --SOC                       SOC. Allowed values are (AM62A, AM67A, AM68A, AM69A, TDA4VM)
    --compile_without_nc        Compile models without NC. Allowed values are (0,1). Default=0
    --compile_with_nc           Compile models with NC. Allowed values are (0,1). Default=1
    --run_ref                   Run HOST emulation inference. Allowed values are (0,1). Default=1
    --run_natc                  Run Inference with NATC flow control. Allowed values are (0,1). Default=0
    --run_ci                    Run Inference with CI flow control. Allowed values are (0,1). Default=0
    --run_target                Run Inference on TARGET. Allowed values are (0,1). Default=0
    --save_model_artifacts      Whether to save compiled artifacts or not. Allowed values are (0,1). Default=0
    --save_model_artifacts_dir  Path to save model artifacts if save_model_artifacts is 1. Default is work_dirs/modelartifacts
    --temp_buffer_dir           Path to redirect temporary buffers for x86 runs. Default is /dev/shm
    --operators                 List of operators (space separated string) to run. By default every operator under tidl_unit_test_data/operators
    --runtimes                  List of runtimes (space separated string) to run tests. Allowed values are (onnxrt, tvmrt). Default=onnxrt
    --tidl_tools_path           Path of tidl tools tarball (named as tidl_tools.tar.gz)
    --compiled_artifacts_path   Path of compiled model artifacts. Will be used only for TARGET run.

    Example:
        ./run_operator_test.sh --SOC=AM68A --run_ref=1 --run_natc=0 --run_ci=0 --save_model_artifacts=1 --operators=\"Add Mul Sqrt\" --runtimes=\"onnxrt\"
        This will run unit tests for (Add, Mul, Sqrt) operators on AM68A using onnxrt runtime, aritifacts will be saved and will run Host emulation inference 
    "
}

SOC="AM68A"
compile_without_nc="0"
compile_with_nc="1"
run_ref="0"
run_natc="0"
run_ci="0"
run_target="0"
save_model_artifacts="0"
save_model_artifacts_dir=""
temp_buffer_dir="/dev/shm"
OPERATORS=()
RUNTIMES=()
tidl_tools_path=""
compiled_artifacts_path=""

while [ $# -gt 0 ]; do
        case "$1" in
        --SOC=*)
        SOC="${1#*=}"
        ;;
        --compile_without_nc=*)
        compile_without_nc="${1#*=}"
        ;;
        --compile_with_nc=*)
        compile_with_nc="${1#*=}"
        ;;
        --run_ref=*)
        run_ref="${1#*=}"
        ;;
        --run_natc=*)
        run_natc="${1#*=}"
        ;;
        --run_ci=*)
        run_ci="${1#*=}"
        ;;
        --run_target=*)
        run_target="${1#*=}"
        ;;
        --save_model_artifacts=*)
        save_model_artifacts="${1#*=}"
        ;;
        --save_model_artifacts_dir=*)
        save_model_artifacts_dir="${1#*=}"
        ;;
        --temp_buffer_dir=*)
        temp_buffer_dir="${1#*=}"
        ;;
        --tidl_tools_path=*)
        tidl_tools_path="${1#*=}"
        ;;
        --compiled_artifacts_path=*)
        compiled_artifacts_path="${1#*=}"
        ;;
        --operators=*)
        operators="${1#*=}"
        ;;
        --runtimes=*)
        runtimes="${1#*=}"
        ;;
        --help)
        usage
        exit
        ;;
        *)
        echo "[ERROR]: Invalid argument $1"
        usage
        exit
        ;;
        esac
        shift
done

for operator in $operators; do
  OPERATORS+=("$operator")
done
for runtime in $runtimes; do
  RUNTIMES+=("$runtime")
done

if [ "$run_ref" == "0" ] && [ "$run_natc" == "0" ] && [ "$run_ci" == "0" ] && [ "$run_target" == "0" ]; then
    run_ref="1"
fi

# Verify arguments
if [ "$SOC" != "AM62A" ] && [ "$SOC" != "AM67A" ] && [ "$SOC" != "AM68A" ] && [ "$SOC" != "AM69A" ] && [ "$SOC" != "TDA4VM" ]; then
    echo "[ERROR]: SOC: $SOC is not allowed."
    echo "         Allowed values are (AM62A, AM67A, AM68A, AM69A, TDA4VM)"
    exit 1
fi
for runtime in "${RUNTIMES[@]}"
do
    if [ "$runtime" != "onnxrt" ] && [ "$runtime" != "tvmrt" ]; then
        echo "[ERROR]: RUNTIME: $runtime is not allowed."
        echo "         Allowed values are (onnxrt, tvmrt)"
        exit 1
    fi
done
if [ "$compile_without_nc" != "1" ] && [ "$compile_without_nc" != "0" ]; then
    echo "[ERROR]: compile_without_nc: $compile_without_nc is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$compile_with_nc" != "1" ] && [ "$compile_with_nc" != "0" ]; then
    echo "[ERROR]: compile_with_nc: $compile_with_nc is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$run_ref" != "1" ] && [ "$run_ref" != "0" ]; then
    echo "[ERROR]: run_ref: $run_ref is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$run_natc" != "1" ] && [ "$run_natc" != "0" ]; then
    echo "[ERROR]: run_natc: $run_natc is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$run_ci" != "1" ] && [ "$run_ci" != "0" ]; then
    echo "[ERROR]: run_ci: $run_ci is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$run_target" != "1" ] && [ "$run_target" != "0" ]; then
    echo "[ERROR]: run_target: $run_target is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$save_model_artifacts" != "1" ] && [ "$save_model_artifacts" != "0" ]; then
    echo "[ERROR]: save_model_artifacts: $save_model_artifacts is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$save_model_artifacts" == "1" ] && [ "$save_model_artifacts_dir" != "" ]; then
    mkdir -p $save_model_artifacts_dir
    if [ "$?" != "0" ]; then
        echo "[WARNING]: Could not create $save_model_artifacts_dir. Using default location to save model artifacts"
        save_model_artifacts_dir=""
    fi
fi

if [ "$compiled_artifacts_path" != "" ]; then
    if [ "$compile_without_nc" == "1" ] || [ "$compile_with_nc" == "1" ]; then
        echo "[WARNING]: Using $compiled_artifacts_path for compiled model artifacts, model compilation will not happen"
        compile_without_nc="0"
        compile_with_nc="0"
    fi
    if [ "$run_target" != "1" ]; then
        echo "[ERROR]: Compiled model artofacts path: $compiled_artifacts_path is given. TARGET run is expected. Exiting"
        exit 1
    fi
    if [ ! -d $compiled_artifacts_path ]; then
        echo "[ERROR]: $compiled_artifacts_path does not exist. Exiting"
        exit 1
    fi
fi

if [ ${#OPERATORS[@]} -eq 0 ]; then
    OPERATORS=()
fi

if [ ${#RUNTIMES[@]} -eq 0 ]; then
    RUNTIMES=('onnxrt')
fi

# Printing options
echo "SOC                       = $SOC"
echo "compile_without_nc        = $compile_without_nc"
echo "compile_with_nc           = $compile_with_nc"
echo "run_ref                   = $run_ref"
echo "run_natc                  = $run_natc"
echo "run_ci                    = $run_ci"
echo "run_target                = $run_target"
echo "save_model_artifacts      = $save_model_artifacts"
echo "save_model_artifacts_dir  = $save_model_artifacts_dir"
echo "temp_buffer_dir           = $temp_buffer_dir"

current_dir="$PWD"
path_edge_ai_benchmark="$current_dir/../../.."
cd "$path_edge_ai_benchmark" 
source ./run_set_env.sh "$SOC"

if [ "$tidl_tools_path" != "" ] && [ ! -f $tidl_tools_path ]; then
    echo "[WARNING]: $tidl_tools_path does not exist. Default tools will be used"
    tidl_tools_path=$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC/tidl_tools.tar.gz
fi
if [ "$tidl_tools_path" == "" ]; then
    tidl_tools_path=$path_edge_ai_benchmark/tools/tidl_tools_package/$SOC/tidl_tools.tar.gz
fi
if [ ! -f $tidl_tools_path ]; then
    echo "[ERROR]: $tidl_tools_path does not exist. Exiting"
    exit 1
fi

cd "$path_edge_ai_benchmark/tests/tidl_unit"

# Set up tidl_tools
mkdir -p temp
cd temp && rm -rf tidl_tools.tar.gz && rm -rf tidl_tools
cp "$tidl_tools_path" ./
tar -xzf tidl_tools.tar.gz 
if [ "$?" -ne 0 ]; then
    echo "[ERROR]: Could not untar $tidl_tools_path. Make sure it is a tarball"
    exit 1
fi
cp -r tidl_tools/ti_cnnperfsim.out ./
cd ../
export TIDL_TOOLS_PATH="$(pwd)/temp/tidl_tools"
export LD_LIBRARY_PATH="${TIDL_TOOLS_PATH}"
echo "TIDL_TOOLS_PATH=${TIDL_TOOLS_PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

if [ "$compiled_artifacts_path" == "" ]; then
    operators_path=$path_edge_ai_benchmark/tests/tidl_unit/tidl_unit_test_data/operators/
    if [ -z "$OPERATORS" ]; then
        for D in $(find $operators_path -mindepth 1 -maxdepth 1 -type d) ; do
            name=`basename $D`
            OPERATORS+=("$name")
        done
    else
        temp_array=()
        for operator in "${OPERATORS[@]}"
        do
            if [ ! -d $operators_path/$operator ]; then
                echo "[WARNING]: $operators_path/$operator does not exist. Skipping it."
            else
                temp_array+=("$operator")
            fi
        done
        OPERATORS=("${temp_array[@]}")
    fi
else
    operators_path=$compiled_artifacts_path
    temp_operators=()
    for D in $(find $operators_path -mindepth 1 -maxdepth 1 -type d) ; do
        name=`basename $D`
        name="${name%%_*}"
        if [[ ! " ${temp_operators[@]} " =~ " ${name} " ]]; then
            temp_operators+=("$name")
        fi
    done
    if [ -z "$OPERATORS" ]; then
        OPERATORS=("${temp_operators[@]}")
    else
        for item in "${OPERATORS[@]}"; do
            if [[ " ${temp_operators[@]} " =~ " ${item} " ]]; then
                intersection+=("$item")
            else
                echo "[WARNING]: Artifacts for $item not present in $compiled_artifacts_path. Skipping it."
            fi

        done
        OPERATORS=("${intersection[@]}")
    fi
fi

###############################################################################
# Run tests for each runtime
###############################################################################
for runtime in "${RUNTIMES[@]}"
do
    echo ""########################################## RUNNING TESTS FOR $runtime "##########################################"
    base_path_reports="$path_edge_ai_benchmark/tests/tidl_unit/internal/operator_test_reports/$runtime"
    path_reports="$base_path_reports/$SOC"
    rm -rf "$path_reports"
    mkdir -p "$path_reports"

    ###############################################################################
    # Run tests for each operator
    ###############################################################################
    for operator in "${OPERATORS[@]}"
    do
        logs_path=$path_reports/$operator
        rm -rf $logs_path
        mkdir -p $logs_path
        echo "Logs will be saved to: $logs_path"

        if [ "$compile_without_nc" == "1" ]; then
            echo "########################################## $operator TEST (WITHOUT NC) ######################################"
            rm -rf work_dirs/modelartifacts/8bits/${operator}_*

            rm -rf "$TIDL_TOOLS_PATH/ti_cnnperfsim.out"

            rm -rf logs/*
            ./run_test.sh --test_suite=operator --tests=$operator --run_infer=0 --runtime=$runtime
            cp logs/*.html "$logs_path/compile_without_nc.html"

            rm -rf logs/*
            if [ "$run_ref" == "1" ]; then
                ./run_test.sh --test_suite=operator --tests=$operator --run_compile=0 --runtime=$runtime
                cp logs/*.html "$logs_path/infer_ref_without_nc.html"
            fi

            if [ "$run_natc" == "1" ]; then
                echo "[WARNING]: NATC will not run without nc"
            fi

            if [ "$run_ci" == "1" ]; then
                echo "[WARNING]: CI will not run without nc"
            fi

            if [ "$run_target" == "1" ]; then
                echo "[WARNING]: TARGET will not run without nc"
            fi
        fi

        if [ "$compile_with_nc" == "1" ]; then
            echo "########################################## $operator TEST (WITH NC) ######################################"
            rm -rf work_dirs/modelartifacts/8bits/${operator}_*

            cp -rp "$TIDL_TOOLS_PATH/../ti_cnnperfsim.out" "$TIDL_TOOLS_PATH"

            rm -rf logs/*
            ./run_test.sh --test_suite=operator --tests=$operator --run_infer=0 --temp_buffer_dir=$temp_buffer_dir --runtime=$runtime
            cp logs/*.html "$logs_path/compile_with_nc.html"

            rm -rf logs/*
            if [ "$run_ref" == "1" ]; then
                ./run_test.sh --test_suite=operator --tests=$operator --run_compile=0 --flow_ctrl=1 --temp_buffer_dir=$temp_buffer_dir --runtime=$runtime
                cp logs/*.html "$logs_path/infer_ref_with_nc.html"
            fi

            rm -rf logs/*
            if [ "$run_natc" == "1" ]; then
                ./run_test.sh --test_suite=operator --tests=$operator --run_compile=0 --flow_ctrl=12 --temp_buffer_dir=$temp_buffer_dir --runtime=$runtime
                cp logs/*.html "$logs_path/infer_natc_with_nc.html"
            fi

            rm -rf logs/*
            if [ "$run_ci" == "1" ]; then
                ./run_test.sh --test_suite=operator --tests=$operator --run_compile=0 --flow_ctrl=0 --temp_buffer_dir=$temp_buffer_dir --runtime=$runtime
                cp logs/*.html "$logs_path/infer_ci_with_nc.html"
            fi

            rm -rf logs/*
            if [ "$run_target" == "1" ]; then
                cd $path_edge_ai_benchmark/tests/evm_test/
                python3 main.py --test_suite=TIDL_UNIT_TEST --soc=$SOC --uart=/dev/am68a-sk-00-usb2 --pc_ip=192.168.46.0 --evm_local_ip=192.168.46.100 --reboot_type=hard --relay_type=ANEL --relay_trigger_mechanism=EXE --relay_exe_path=/work/ti/UNIT_TEST/net-pwrctrl.exe --relay_ip_address=10.24.69.252 --relay_power_port=8 --dataset_dir=/work/ti/UNIT_TEST/tidl_models/unitTest/onnx/tidl_unit_test_assets --operators=$operator
                cd -
                cp logs/*.html "$logs_path"
                cd $logs_path
                rm -rf temp
                mkdir -p temp
                mv ./*_Chunk_*.html temp
                cd temp
                pytest_html_merger -i ./ -o ../infer_target_with_nc.html
                cd ../
                rm -rf temp
                cd $path_edge_ai_benchmark/tests/tidl_unit
            fi

            if [ "$save_model_artifacts" == "0" ]; then
                rm -rf work_dirs/modelartifacts/8bits/${operator}_*
            elif [ "$save_model_artifacts_dir" != "" ]; then
                mv work_dirs/modelartifacts/8bits/${operator}_* $save_model_artifacts_dir
            fi
        fi

        if [ "$compiled_artifacts_path" != "" ]; then
            echo "[INFO]: Using $compiled_artifacts_path for model artifacts to run inference on TARGET"

            rm -rf logs/*
            if [ "$run_target" == "1" ]; then
                cd $path_edge_ai_benchmark/tests/evm_test/
                python3 main.py --test_suite=TIDL_UNIT_TEST --soc=$SOC --uart=/dev/am68a-sk-00-usb2 --pc_ip=192.168.46.0 --evm_local_ip=192.168.46.100 --reboot_type=hard --relay_type=ANEL --relay_trigger_mechanism=EXE --relay_exe_path=/work/ti/UNIT_TEST/net-pwrctrl.exe --relay_ip_address=10.24.69.252 --relay_power_port=8 --dataset_dir=/work/ti/UNIT_TEST/tidl_models/unitTest/onnx/tidl_unit_test_assets --operators=$operator --artifacts_folder=$compiled_artifacts_path
                cd -
                cp logs/*.html "$logs_path"
                cd $logs_path
                rm -rf temp
                mkdir -p temp
                mv ./*_Chunk_*.html temp
                cd temp
                pytest_html_merger -i ./ -o ../infer_target_with_nc.html
                cd ../
                rm -rf temp
                cd $path_edge_ai_benchmark/tests/tidl_unit
            fi
        fi
    done

    # Generate summary report
    cd internal
    python3 report_summary_generation.py --reports_path=$base_path_reports
    cd ../
done

# Clear temporary files
cd "$path_edge_ai_benchmark/tests/tidl_unit"
rm -rf temp
