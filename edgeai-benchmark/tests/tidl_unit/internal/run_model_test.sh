#!/usr/bin/env bash

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

usage() {
echo \
"Usage:
    Helper script to run unit tests model wise

    Options:
    --SOC                       SOC. Allowed values are (AM62A, AM67A, AM68A, AM69A, TDA4VM)
    --tidl_offload              Offload tests to TIDL. Allowed values are (0,1). Default=1
    --compile_without_nc        Compile models without NC. Allowed values are (0,1). Default=0
    --compile_with_nc           Compile models with NC. Allowed values are (0,1). Default=1
    --run_ref                   Run HOST emulation inference. Allowed values are (0,1). Default=1
    --run_natc                  Run Inference with NATC flow control. Allowed values are (0,1). Default=0
    --run_ci                    Run Inference with CI flow control. Allowed values are (0,1). Default=0
    --work_dir                  Full path to save model artifacts during compilation. Same will be used to fetch compiled model artifacts during inference
                                Default is work_dirs/modelartifacts
    --save_model_artifacts      Whether to preserve compiled artifacts or not in work_dir. Allowed values are (0,1). Default=0
    --temp_buffer_dir           Path to redirect temporary buffers for x86 runs. Default is /dev/shm
    --temp_nc_dir               Path to redirect temporary NC buffers. Default is /tmp
    --nmse_threshold            Normalized Mean Squared Error (NMSE) threshold for inference output. Default: Parsed from config.yaml of model is present else 0.5
    --runtimes                  List of runtimes (space separated string) to run tests. Allowed values are (onnxrt, tvmrt). Default=onnxrt
    --tensor_bits               8/16/32. Default: 8
    --num_threads               Number of threads for test run. Default=auto
    --tidl_tools_path           Path of tidl tools tarball

    --test_file                 Path to text file containg test to run. Default=null
    --models                    List of models (space separated string) to run. By default every model under tidl_unit_test_data/models

    NOTE: If 'test_file' is defined, it will take precedence over 'models' option.

    Example:
        ./run_model_test.sh --SOC=AM68A --run_ref=1 --run_natc=0 --run_ci=0 --save_model_artifacts=1 --test_file=abc.txt --runtimes=\"onnxrt\"
        This will run all tests defined in abc.txt on AM68A using onnxrt runtime, aritifacts will be saved and will run Host emulation inference 

        ./run_model_test.sh --SOC=AM68A --run_ref=1 --run_natc=0 --run_ci=0 --save_model_artifacts=1 --models=\"Add Mul Sqrt\" --runtimes=\"onnxrt\"
        This will run unit tests for (Add, Mul, Sqrt) models on AM68A using onnxrt runtime, aritifacts will be saved and will run Host emulation inference 
    "
}

SOC="AM68A"
tidl_offload="1"
compile_without_nc="0"
compile_with_nc="1"
run_ref="1"
run_natc="0"
run_ci="0"
work_dir=""
test_file=""
save_model_artifacts="0"
temp_buffer_dir="/dev/shm"
temp_nc_dir="/tmp"

MODELS=()
RUNTIMES=()
tidl_tools_path=""
nmse_threshold=""
num_threads=""
disable_plot=""
tensor_bits="8"

while [ $# -gt 0 ]; do
        case "$1" in
        --SOC=*)
        SOC="${1#*=}"
        ;;
        --tidl_offload=*)
        tidl_offload="${1#*=}"
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
        --work_dir=*)
        work_dir="${1#*=}"
        ;;
        --save_model_artifacts=*)
        save_model_artifacts="${1#*=}"
        ;;
        --temp_buffer_dir=*)
        temp_buffer_dir="${1#*=}"
        ;;
        --temp_nc_dir=*)
        temp_nc_dir="${1#*=}"
        ;;
        --tidl_tools_path=*)
        tidl_tools_path="${1#*=}"
        ;;
        --nmse_threshold=*)
        nmse_threshold="${1#*=}"
        ;;
        --tensor_bits=*)
        tensor_bits="${1#*=}"
        ;;
        --models=*)
        models="${1#*=}"
        ;;
        --test_file=*)
        test_file="${1#*=}"
        ;;
        --runtimes=*)
        runtimes="${1#*=}"
        ;;
        --num_threads=*)
        num_threads="${1#*=}"
        ;;
        --disable_plot=*)
        disable_plot="${1#*=}"
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

TEMP_TEST_FILE_DIR="$SCRIPT_DIR/temp_test_files"
rm -rf $TEMP_TEST_FILE_DIR
mkdir -p $TEMP_TEST_FILE_DIR

USE_TEST_FILE=0
if [[ "$test_file" != "" ]]; then
    if [[ -f "$test_file" ]]; then
        USE_TEST_FILE=1
    else
        echo "[WARNING]: $test_file not found. Using 'models' option."
        USE_TEST_FILE=0
    fi
fi

# Parse models from test_file or models option
if [[ $USE_TEST_FILE -eq 1 ]]; then
    while IFS= read -r line; do
        if [[ -z "$(echo "$line" | tr -d '[:space:]')" ]]; then
            continue
        fi
        if [[ "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        trimmed_line=$(echo "$line" | sed -e 's/^[[:space:]]*//')
        parent=$(echo "$trimmed_line" | cut -d'/' -f1)
        model=$(echo "$trimmed_line" | cut -d'/' -f2)

        MODELS+=("$parent")
        echo "$model" >> $TEMP_TEST_FILE_DIR/$parent.txt
    done < "$test_file"
else
    for model in $models; do
        MODELS+=("$model")
    done
fi

# Remove duplicated model
declare -A seen_elements
UNIQUE_MODELS=()
for element in "${MODELS[@]}"; do
    if [[ -z "${seen_elements[$element]}" ]]; then
        UNIQUE_MODELS+=("$element")
        seen_elements[$element]=1
    fi
done
MODELS=("${UNIQUE_MODELS[@]}")

# Parse runtimes
for runtime in $runtimes; do
  RUNTIMES+=("$runtime")
done


# Verify arguments
if [ "$SOC" != "AM62A" ] && [ "$SOC" != "AM67A" ] && [ "$SOC" != "AM68A" ] && [ "$SOC" != "AM69A" ] && [ "$SOC" != "TDA4VM" ]; then
    echo "[ERROR]: SOC: $SOC is not allowed."
    echo "         Allowed values are (AM62A, AM67A, AM68A, AM69A, TDA4VM)"
    exit 1
fi
if [ "$tidl_offload" != "1" ] && [ "$tidl_offload" != "0" ]; then
    echo "[ERROR]: tidl_offload: $tidl_offload is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$tensor_bits" != "8" ] && [ "$tensor_bits" != "16" ] && [ "$tensor_bits" != "32" ]; then
    echo "[WARNING]: tensor_bits: $tidl_offload is not allowed. Defaulting to 8."
    echo "           Allowed values are (8, 16, 32)"
    tensor_bits="8"
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
if [ "$save_model_artifacts" != "1" ] && [ "$save_model_artifacts" != "0" ]; then
    echo "[ERROR]: save_model_artifacts: $save_model_artifacts is not allowed."
    echo "         Allowed values are (0,1)"
    exit 1
fi
if [ "$work_dir" != "" ]; then
    mkdir -p $work_dir
    if [ "$?" != "0" ]; then
        echo "[WARNING]: Could not create $work_dir. Using default location for model artifacts"
        work_dir=""
    fi
fi

disable_plot_ref="0"
disable_plot_non_ref="1"
if [ "$disable_plot" == "1" ]; then
    disable_plot_ref="1"
    disable_plot_non_ref="1"
elif [ "$disable_plot" == "0" ]; then
    disable_plot_ref="0"
    disable_plot_non_ref="0"
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=()
fi

if [ ${#RUNTIMES[@]} -eq 0 ]; then
    RUNTIMES=('onnxrt')
fi

# Printing options
echo "SOC                       = $SOC"
echo "tidl_offload              = $tidl_offload"
echo "tensor_bits               = $tensor_bits"
echo "compile_without_nc        = $compile_without_nc"
echo "compile_with_nc           = $compile_with_nc"
echo "run_ref                   = $run_ref"
echo "run_natc                  = $run_natc"
echo "run_ci                    = $run_ci"
echo "work_dir                  = $work_dir"
echo "save_model_artifacts      = $save_model_artifacts"
echo "temp_buffer_dir           = $temp_buffer_dir"
echo "temp_nc_dir               = $temp_nc_dir"
echo "num_threads               = $num_threads"
echo "test_file                 = $test_file"
echo "nmse_threshold            = $nmse_threshold"


current_dir="$PWD"
path_edge_ai_benchmark="$current_dir/../../.."
default_work_dir="$current_dir/../work_dirs/modelartifacts"
cd "$path_edge_ai_benchmark" 
source ./run_set_env.sh "$SOC"

if [ "$tidl_tools_path" != "" ] && [ ! -f $tidl_tools_path ] && [ ! -d $tidl_tools_path ]; then
    echo "[WARNING]: $tidl_tools_path does not exist. Default tools will be used"
    tidl_tools_path=$path_edge_ai_benchmark/tools/tidl_tools_package/bin/$SOC/tidl_tools
fi
if [ "$tidl_tools_path" == "" ]; then
    tidl_tools_path=$path_edge_ai_benchmark/tools/tidl_tools_package/bin/$SOC/tidl_tools
fi

if [ ! -f $tidl_tools_path ] && [ ! -d $tidl_tools_path ]; then
    echo "[ERROR]: $tidl_tools_path does not exist. Exiting"
    exit 1
fi

cd "$path_edge_ai_benchmark/tests/tidl_unit"

# Set up tidl_tools
mkdir -p temp
cd temp && rm -rf *.tar.gz && rm -rf tidl_tools

# Extract the filename from the path
if [ -f $tidl_tools_path ]; then
    tarball_name=$(basename "$tidl_tools_path")
    cp "$tidl_tools_path" ./
    tar -xzf "$tarball_name"
    if [ "$?" -ne 0 ]; then
        echo "[ERROR]: Could not untar $tidl_tools_path. Make sure it is a tarball"
        exit 1
    fi
else
    mkdir -p tidl_tools
    cp -r $tidl_tools_path/* tidl_tools/
fi

# Check if tidl_tools directory was created after extraction
if [ ! -d "tidl_tools" ]; then
    echo "[ERROR]: tidl_tools directory not found after extracting $tidl_tools_path. The tarball may not contain the expected directory structure"
    exit 1
fi
cp -r tidl_tools/ti_cnnperfsim.out ./
cd ../
export TIDL_TOOLS_PATH="$(pwd)/temp/tidl_tools"
export LD_LIBRARY_PATH="${TIDL_TOOLS_PATH}"
echo "TIDL_TOOLS_PATH=${TIDL_TOOLS_PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

if [ "$work_dir" == "" ] || [ $compile_with_nc == "1" ] || [ "$compile_without_nc" == "1" ]; then
    models_path=$path_edge_ai_benchmark/tests/tidl_unit/tidl_unit_test_data/models/
    if [ -z "$MODELS" ]; then
        for D in $(find $models_path -mindepth 1 -maxdepth 1 -type d) ; do
            name=`basename $D`
            MODELS+=("$name")
        done
    else
        temp_array=()
        for model in "${MODELS[@]}"
        do
            if [ ! -d $models_path/$model ]; then
                echo "[WARNING]: $models_path/$model does not exist. Skipping it."
            else
                temp_array+=("$model")
            fi
        done
        MODELS=("${temp_array[@]}")
    fi
else
    if [ ! -d $work_dir ]; then
        echo "[ERROR]: $work_dir does not exist.. Exiting"
        exit 1
    fi
    models_path=$work_dir
    temp_models=()
    for D in $(find $models_path -mindepth 1 -maxdepth 1 -type d) ; do
        name=`basename $D`
        name="${name%%_*}"
        if [[ ! " ${temp_models[@]} " =~ " ${name} " ]]; then
            temp_models+=("$name")
        fi
    done
    if [ -z "$MODELS" ]; then
        MODELS=("${temp_models[@]}")
    else
        for item in "${MODELS[@]}"; do
            if [[ " ${temp_models[@]} " =~ " ${item} " ]]; then
                intersection+=("$item")
            else
                echo "[WARNING]: Artifacts for $item not present in $work_dir. Skipping it."
            fi

        done
        MODELS=("${intersection[@]}")
    fi
fi

# Run only inference if tidl_offload is 0
if [ "$tidl_offload" == "0" ]; then
    compile_without_nc="0"
    compile_with_nc="0"
    run_ref="1"
    run_natc="0"
    run_ci="0"
    echo -e "\n[INFO]: tidl_offload is false, Running tests on CPU\n"
fi

# Change yaml file to set tensor bits
sed -i "/tensor_bits/c\tensor_bits : ${tensor_bits}" tidl_unit.yaml

# Add models in remove_list which you don't want to run 
remove_list=()
filtered_list=()
for item in "${MODELS[@]}"; do
    if [[ ! " ${remove_list[@]} " =~ " ${item} " ]]; then
        filtered_list+=("$item")
    fi
done
MODELS=("${filtered_list[@]}")

###############################################################################
# Run tests for each runtime
###############################################################################
for runtime in "${RUNTIMES[@]}"
do
    echo ""########################################## RUNNING TESTS FOR $runtime "##########################################"
    base_path_reports="$path_edge_ai_benchmark/tests/tidl_unit/internal/model_test_reports/$runtime"
    path_reports="$base_path_reports/$SOC"
    rm -rf "$path_reports"
    mkdir -p "$path_reports"

    ###############################################################################
    # Run tests for each model
    ###############################################################################
    for model in "${MODELS[@]}"
    do
        logs_path=$path_reports/$model
        rm -rf $logs_path
        mkdir -p $logs_path
        echo "Logs will be saved to: $logs_path"

        if [[ $USE_TEST_FILE -eq 1 ]]; then
            test_option="--test_file=$TEMP_TEST_FILE_DIR/$model.txt" 
        else
            test_option="--tests=$model" 
        fi

        extra_args="--tidl_offload=$tidl_offload --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads --nmse_threshold=$nmse_threshold"

        if [ "$compile_without_nc" == "1" ]; then
            echo "########################################## $model TEST (WITHOUT NC) ######################################"
            rm -rf ${work_dir}/${model}_*
            rm -rf $default_work_dir

            rm -rf "$TIDL_TOOLS_PATH/ti_cnnperfsim.out"

            rm -rf logs/*
            ./run_test.sh --test_suite=model $test_option --run_infer=0 $extra_args
            cp logs/*.html "$logs_path/compile_without_nc.html"
            if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                rm -rf $temp_buffer_dir/vashm_buff*
            fi

            rm -rf logs/*
            if [ "$run_ref" == "1" ]; then
                ./run_test.sh --test_suite=model $test_option --run_compile=0 $extra_args --disable_plot=$disable_plot_ref
                cp logs/*.html "$logs_path/infer_ref_without_nc.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            if [ "$run_natc" == "1" ]; then
                echo "[WARNING]: NATC will not run without nc"
            fi

            if [ "$run_ci" == "1" ]; then
                echo "[WARNING]: CI will not run without nc"
            fi
        fi

        if [ "$compile_with_nc" == "1" ]; then
            echo "########################################## $model TEST (WITH NC) ######################################"
            rm -rf ${work_dir}/${model}_*
            rm -rf $default_work_dir

            cp -rp "$TIDL_TOOLS_PATH/../ti_cnnperfsim.out" "$TIDL_TOOLS_PATH"

            rm -rf logs/*
            ./run_test.sh --test_suite=model $test_option --run_infer=0 $extra_args
            cp logs/*.html "$logs_path/compile_with_nc.html"
            if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                rm -rf $temp_buffer_dir/vashm_buff*
            fi

            rm -rf logs/*
            if [ "$run_ref" == "1" ]; then
                ./run_test.sh --test_suite=model $test_option --run_compile=0 --flow_ctrl=1 $extra_args --disable_plot=$disable_plot_ref
                cp logs/*.html "$logs_path/infer_ref_with_nc.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_natc" == "1" ]; then
                ./run_test.sh --test_suite=model $test_option --run_compile=0 --flow_ctrl=12 $extra_args --disable_plot=$disable_plot_non_ref
                cp logs/*.html "$logs_path/infer_natc_with_nc.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_ci" == "1" ]; then
                ./run_test.sh --test_suite=model $test_option --run_compile=0 --flow_ctrl=0 $extra_args --disable_plot=$disable_plot_non_ref
                cp logs/*.html "$logs_path/infer_ci_with_nc.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi
        fi

        # Run inference using artifacts present under work_dir
        if [ "$compile_without_nc" == "0" ] && [ "$compile_with_nc" == "0" ]; then

            rm -rf logs/*
            if [ "$run_ref" == "1" ]; then
                ./run_test.sh --test_suite=model $test_option --run_compile=0 --flow_ctrl=1 $extra_args --disable_plot=$disable_plot_ref
                cp logs/*.html "$logs_path/infer_ref.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_natc" == "1" ]; then
                ./run_test.sh --test_suite=model $test_option --run_compile=0 --flow_ctrl=12 $extra_args --disable_plot=$disable_plot_non_ref
                cp logs/*.html "$logs_path/infer_natc.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_ci" == "1" ]; then
                ./run_test.sh --test_suite=model $test_option --run_compile=0 --flow_ctrl=0 $extra_args --disable_plot=$disable_plot_non_ref
                cp logs/*.html "$logs_path/infer_ci.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi
        fi

        if [ "$save_model_artifacts" == "0" ]; then
            rm -rf ${work_dir}/${model}_*
            rm -rf $default_work_dir
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
