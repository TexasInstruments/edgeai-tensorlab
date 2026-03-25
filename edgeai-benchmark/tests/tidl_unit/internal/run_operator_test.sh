#!/usr/bin/env bash

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

usage() {
echo \
"Usage:
    Helper script to run unit tests operator wise

    Options:
    --SOC                       SOC. Allowed values are (AM62A, AM67A, AM68A, AM69A, TDA4VM)
    --tidl_offload              Offload tests to TIDL. Allowed values are (0,1). Default=1
    --compile_without_nc        Compile models without NC. Allowed values are (0,1). Default=0
    --compile_with_nc           Compile models with NC. Allowed values are (0,1). Default=1
    --run_ref                   Run HOST emulation inference. Allowed values are (0,1). Default=1
    --run_natc                  Run Inference with NATC flow control. Allowed values are (0,1). Default=0
    --run_ci                    Run Inference with CI flow control. Allowed values are (0,1). Default=0
    --run_target                Run Inference on TARGET. Allowed values are (0,1). Default=0
    --work_dir                  Full path to save model artifacts during compilation. Same will be used to fetch compiled model artifacts during inference
                                Default is work_dirs/modelartifacts
    --save_model_artifacts      Whether to preserve compiled artifacts or not in work_dir. Allowed values are (0,1). Default=0
    --temp_buffer_dir           Path to redirect temporary buffers for x86 runs. Default is /dev/shm
    --temp_nc_dir               Path to redirect temporary NC buffers. Default is /tmp
    --nmse_threshold            Normalized Mean Squared Error (NMSE) threshold for inference output. Default: 0.5
    --runtimes                  List of runtimes (space separated string) to run tests. Allowed values are (onnxrt, tvmrt). Default=onnxrt
    --tensor_bits               8/16/32. Default: 8
    --num_threads               Number of threads for test run. Default=auto
    --tidl_tools_path           Path of tidl tools tarball

    --test_file                 Path to text file containg test to run. Default=null
    --operators                 List of operators (space separated string) to run. By default every operator under tidl_unit_test_data/operators

    NOTE: If 'test_file' is defined, it will take precedence over 'operators' option.

    Example:
        ./run_operator_test.sh --SOC=AM68A --run_ref=1 --run_natc=0 --run_ci=0 --save_model_artifacts=1 --test_file=abc.txt --runtimes=\"onnxrt\"
        This will run all tests defined in abc.txt on AM68A using onnxrt runtime, aritifacts will be saved and will run Host emulation inference 

        ./run_operator_test.sh --SOC=AM68A --run_ref=1 --run_natc=0 --run_ci=0 --save_model_artifacts=1 --operators=\"Add Mul Sqrt\" --runtimes=\"onnxrt\"
        This will run unit tests for (Add, Mul, Sqrt) operators on AM68A using onnxrt runtime, aritifacts will be saved and will run Host emulation inference 
    "
}

SOC="AM68A"
tidl_offload="1"
compile_without_nc="0"
compile_with_nc="1"
run_ref="1"
run_natc="0"
run_ci="0"
run_target="0"
work_dir=""
test_file=""
save_model_artifacts="0"
temp_buffer_dir="/dev/shm"
temp_nc_dir="/tmp"

OPERATORS=()
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
        --run_target=*)
        run_target="${1#*=}"
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
        --operators=*)
        operators="${1#*=}"
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
        echo "[WARNING]: $test_file not found. Using 'operators' option."
        USE_TEST_FILE=0
    fi
fi

# Parse operators from test_file or operators option
if [[ $USE_TEST_FILE -eq 1 ]]; then
    while IFS= read -r line; do
        if [[ -z "$(echo "$line" | tr -d '[:space:]')" ]]; then
            continue
        fi
        if [[ "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        trimmed_line=$(echo "$line" | sed -e 's/^[[:space:]]*//')
        op=$(echo "$trimmed_line" | cut -d'_' -f1)

        OPERATORS+=("$op")
        echo "$trimmed_line" >> $TEMP_TEST_FILE_DIR/$op.txt
    done < "$test_file"
else
    for operator in $operators; do
        OPERATORS+=("$operator")
    done
fi

# Remove duplicated operator
declare -A seen_elements
UNIQUE_OPERATORS=()
for element in "${OPERATORS[@]}"; do
    if [[ -z "${seen_elements[$element]}" ]]; then
        UNIQUE_OPERATORS+=("$element")
        seen_elements[$element]=1
    fi
done
OPERATORS=("${UNIQUE_OPERATORS[@]}")

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

# Operator specific nmse_threshold
threshold_e_minus1="0.1"
threshold_e_minus2="0.01"
threshold_e_minus3="0.001"
declare -A ops_nmse_threshold
ops_nmse_threshold=(
    ["ArgMax"]="0"
    ["ArgMin"]="0"
    # 1e-3
    ["Abs"]=$threshold_e_minus3
    ["Acos"]=$threshold_e_minus3
    ["Asin"]=$threshold_e_minus3
    ["Asinh"]=$threshold_e_minus3
    ["Atan"]=$threshold_e_minus3
    ["Clip"]=$threshold_e_minus3
    ["Concat"]=$threshold_e_minus3
    ["Cos"]=$threshold_e_minus3
    ["DepthToSpace"]=$threshold_e_minus3
    ["Elu"]=$threshold_e_minus3
    ["Erf"]=$threshold_e_minus3
    ["Flatten"]=$threshold_e_minus3
    ["Gemm"]=$threshold_e_minus3
    ["LeakyRelu"]=$threshold_e_minus3
    ["Neg"]=$threshold_e_minus3
    ["Pad"]=$threshold_e_minus3
    ["Pow"]=$threshold_e_minus3
    ["Relu"]=$threshold_e_minus3
    ["Reshape"]=$threshold_e_minus3
    ["Sigmoid"]=$threshold_e_minus3
    ["Sin"]=$threshold_e_minus3
    ["Slice"]=$threshold_e_minus3
    ["SpaceToDepth"]=$threshold_e_minus3
    ["Squeeze"]=$threshold_e_minus3
    ["Sum"]=$threshold_e_minus3
    ["Tanh"]=$threshold_e_minus3
    ["Transpose"]=$threshold_e_minus3
    ["Unsqueeze"]=$threshold_e_minus3
    # 1e-2
    ["Add"]=$threshold_e_minus2
    ["AveragePool"]=$threshold_e_minus2
    ["BatchNormalization"]=$threshold_e_minus2
    ["Conv"]=$threshold_e_minus2
    ["ConvTranspose"]=$threshold_e_minus2
    ["Expand"]=$threshold_e_minus2
    ["GlobalAveragePool"]=$threshold_e_minus2
    ["HardSwish"]=$threshold_e_minus2
    ["InstanceNormalization"]=$threshold_e_minus2
    ["LayerNormalization"]=$threshold_e_minus2
    ["MatMul"]=$threshold_e_minus2
    ["Max"]=$threshold_e_minus2
    ["MaxPool"]=$threshold_e_minus2
    ["Min"]=$threshold_e_minus2
    ["Mish"]=$threshold_e_minus2
    ["PRelu"]=$threshold_e_minus2
    ["ReduceMax"]=$threshold_e_minus2
    ["ReduceMean"]=$threshold_e_minus2
    ["ReduceMin"]=$threshold_e_minus2
    ["ReduceSum"]=$threshold_e_minus2
    ["Resize"]=$threshold_e_minus2
    ["Softmax"]=$threshold_e_minus2
    ["Sub"]=$threshold_e_minus2
    ["Sqrt"]=$threshold_e_minus2
    ["TopK"]=$threshold_e_minus2
    # 1e-1
    ["Div"]=$threshold_e_minus1
    ["Exp"]=$threshold_e_minus1
    ["Gather"]=$threshold_e_minus1
    ["Mul"]=$threshold_e_minus1
    # Doubt
    ["Floor"]=$threshold_e_minus2
    ["HardSigmoid"]=$threshold_e_minus2
    ["ScatterElements"]=$threshold_e_minus2
    ["ScatterND"]=$threshold_e_minus2
    ["Tan"]=$threshold_e_minus2
    ["Cosh"]=$threshold_e_minus1
    ["GridSample"]=$threshold_e_minus1
    ["Log"]=$threshold_e_minus1
    ["Sinh"]=$threshold_e_minus1
)

if [ ${#OPERATORS[@]} -eq 0 ]; then
    OPERATORS=()
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
echo "run_target                = $run_target"
echo "work_dir                  = $work_dir"
echo "save_model_artifacts      = $save_model_artifacts"
echo "temp_buffer_dir           = $temp_buffer_dir"
echo "temp_nc_dir               = $temp_nc_dir"
echo "num_threads               = $num_threads"
echo "test_file                 = $test_file"


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
    if [ ! -d $work_dir ]; then
        echo "[ERROR]: $work_dir does not exist.. Exiting"
        exit 1
    fi
    operators_path=$work_dir
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
                echo "[WARNING]: Artifacts for $item not present in $work_dir. Skipping it."
            fi

        done
        OPERATORS=("${intersection[@]}")
    fi
fi

# Run only inference if tidl_offload is 0
if [ "$tidl_offload" == "0" ]; then
    compile_without_nc="0"
    compile_with_nc="0"
    run_ref="1"
    run_natc="0"
    run_ci="0"
    run_target="0"
    echo -e "\n[INFO]: tidl_offload is false, Running tests on CPU\n"
fi

# Change yaml file to set tensor bits
sed -i "/tensor_bits/c\tensor_bits : ${tensor_bits}" tidl_unit.yaml

# Add operators in remove_list which you don't want to run 
# "Add" "Conv" "Mul"
# "ScatterElements" "TopK"
remove_list=()
filtered_list=()
for item in "${OPERATORS[@]}"; do
    if [[ ! " ${remove_list[@]} " =~ " ${item} " ]]; then
        filtered_list+=("$item")
    fi
done
OPERATORS=("${filtered_list[@]}")

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
        op_nmse_threshold=$nmse_threshold
        if [[ -v ops_nmse_threshold["$operator"] ]] && [ "$op_nmse_threshold" == "" ]; then
            op_nmse_threshold="${ops_nmse_threshold[$operator]}"
        fi

        logs_path=$path_reports/$operator
        rm -rf $logs_path
        mkdir -p $logs_path
        echo "Logs will be saved to: $logs_path"

        if [[ $USE_TEST_FILE -eq 1 ]]; then
            test_option="--test_file=$TEMP_TEST_FILE_DIR/$operator.txt" 
        else
            test_option="--tests=$operator" 
        fi

        if [ "$compile_without_nc" == "1" ]; then
            echo "########################################## $operator TEST (WITHOUT NC) ######################################"
            rm -rf ${work_dir}/${operator}_*
            rm -rf $default_work_dir

            rm -rf "$TIDL_TOOLS_PATH/ti_cnnperfsim.out"

            rm -rf logs/*
            ./run_test.sh --test_suite=operator $test_option --tidl_offload=$tidl_offload --run_infer=0 --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads
            cp logs/*.html "$logs_path/compile_without_nc.html"
            if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                rm -rf $temp_buffer_dir/vashm_buff*
            fi

            rm -rf logs/*
            if [ "$run_ref" == "1" ]; then
                ./run_test.sh --test_suite=operator $test_option --tidl_offload=$tidl_offload --run_compile=0 --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --nmse_threshold=$op_nmse_threshold --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads --disable_plot=$disable_plot_ref
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

            if [ "$run_target" == "1" ]; then
                echo "[WARNING]: TARGET will not run without nc"
            fi
        fi

        if [ "$compile_with_nc" == "1" ]; then
            echo "########################################## $operator TEST (WITH NC) ######################################"
            rm -rf ${work_dir}/${operator}_*
            rm -rf $default_work_dir

            cp -rp "$TIDL_TOOLS_PATH/../ti_cnnperfsim.out" "$TIDL_TOOLS_PATH"

            rm -rf logs/*
            ./run_test.sh --test_suite=operator $test_option --tidl_offload=$tidl_offload --run_infer=0 --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --nmse_threshold=$op_nmse_threshold --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads
            cp logs/*.html "$logs_path/compile_with_nc.html"
            if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                rm -rf $temp_buffer_dir/vashm_buff*
            fi

            rm -rf logs/*
            if [ "$run_ref" == "1" ]; then
                ./run_test.sh --test_suite=operator $test_option --tidl_offload=$tidl_offload --run_compile=0 --flow_ctrl=1 --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --nmse_threshold=$op_nmse_threshold --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads --disable_plot=$disable_plot_ref
                cp logs/*.html "$logs_path/infer_ref_with_nc.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_natc" == "1" ]; then
                ./run_test.sh --test_suite=operator $test_option --tidl_offload=$tidl_offload --run_compile=0 --flow_ctrl=12 --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --nmse_threshold=$op_nmse_threshold --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads --disable_plot=$disable_plot_non_ref
                cp logs/*.html "$logs_path/infer_natc_with_nc.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_ci" == "1" ]; then
                ./run_test.sh --test_suite=operator $test_option --tidl_offload=$tidl_offload --run_compile=0 --flow_ctrl=0 --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --nmse_threshold=$op_nmse_threshold --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads --disable_plot=$disable_plot_non_ref
                cp logs/*.html "$logs_path/infer_ci_with_nc.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_target" == "1" ]; then
                cd $path_edge_ai_benchmark/tests/evm_test/

                extra_args="--nmse-threshold $op_nmse_threshold"
                if [ "$disable_plot_non_ref" == "1" ]; then
                    extra_args="$extra_args --disable-plot"
                fi

                python3 main.py --test_suite=TIDL_UNIT_TEST --soc=$SOC --uart=/dev/am68a-sk-00-usb2 --pc_ip=192.168.46.0 --evm_local_ip=192.168.46.100 --reboot_type=hard --relay_type=ANEL --relay_trigger_mechanism=EXE --relay_exe_path=/work/ti/UNIT_TEST/net-pwrctrl.exe --relay_ip_address=10.24.69.252 --relay_power_port=8 --dataset_dir=/work/ti/UNIT_TEST/tidl_models/unitTest/onnx/tidl_unit_test_assets --operators=$operator --artifacts_folder=$work_dir --extra_args=$extra_args
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

        # Run inference using artifacts present under work_dir
        if [ "$compile_without_nc" == "0" ] && [ "$compile_with_nc" == "0" ]; then

            rm -rf logs/*
            if [ "$run_ref" == "1" ]; then
                ./run_test.sh --test_suite=operator $test_option --tidl_offload=$tidl_offload --run_compile=0 --flow_ctrl=1 --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --nmse_threshold=$op_nmse_threshold --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads --disable_plot=$disable_plot_ref
                cp logs/*.html "$logs_path/infer_ref.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_natc" == "1" ]; then
                ./run_test.sh --test_suite=operator $test_option --tidl_offload=$tidl_offload --run_compile=0 --flow_ctrl=12 --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --nmse_threshold=$op_nmse_threshold --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads --disable_plot=$disable_plot_non_ref
                cp logs/*.html "$logs_path/infer_natc.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_ci" == "1" ]; then
                ./run_test.sh --test_suite=operator $test_option --tidl_offload=$tidl_offload --run_compile=0 --flow_ctrl=0 --temp_buffer_dir=$temp_buffer_dir --temp_nc_dir=$temp_nc_dir --nmse_threshold=$op_nmse_threshold --runtime=$runtime --work_dir=$work_dir --num_threads=$num_threads --disable_plot=$disable_plot_non_ref
                cp logs/*.html "$logs_path/infer_ci.html"
                if [ "$temp_buffer_dir" != "/dev/shm" ]; then
                    rm -rf $temp_buffer_dir/vashm_buff*
                fi
            fi

            rm -rf logs/*
            if [ "$run_target" == "1" ]; then
                cd $path_edge_ai_benchmark/tests/evm_test/

                extra_args="--nmse-threshold $op_nmse_threshold"
                if [ "$disable_plot_non_ref" == "1" ]; then
                    extra_args="$extra_args --disable-plot"
                fi

                python3 main.py --test_suite=TIDL_UNIT_TEST --soc=$SOC --uart=/dev/am68a-sk-00-usb2 --pc_ip=192.168.46.0 --evm_local_ip=192.168.46.100 --reboot_type=hard --relay_type=ANEL --relay_trigger_mechanism=EXE --relay_exe_path=/work/ti/UNIT_TEST/net-pwrctrl.exe --relay_ip_address=10.24.69.252 --relay_power_port=8 --dataset_dir=/work/ti/UNIT_TEST/tidl_models/unitTest/onnx/tidl_unit_test_assets --operators=$operator --artifacts_folder=$work_dir --extra_args=$extra_args
                cd -
                cp logs/*.html "$logs_path"
                cd $logs_path
                rm -rf temp
                mkdir -p temp
                mv ./*_Chunk_*.html temp
                cd temp
                pytest_html_merger -i ./ -o ../infer_target.html
                cd ../
                rm -rf temp
                cd $path_edge_ai_benchmark/tests/tidl_unit
            fi
        fi

        if [ "$save_model_artifacts" == "0" ]; then
            rm -rf ${work_dir}/${operator}_*
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
