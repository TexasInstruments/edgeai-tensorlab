import json
import sys
import subprocess
import threading
import time
import os
import shutil
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

start_time = datetime.now()

def format_seconds_to_hhmmss(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def execute_scp_command(pc_name, command, timeout=300):
    """
    Copy from remote PC via SCP

    Args:
        pc_name (str): The PC hostname/username combination
        command (str): The command to execute
        command_type (str): Type of command for logging purposes
    
    Returns:
        tuple: (pc_name, success, output, error)
    """
    
    try:
        scp_command = f"scp -r -o StrictHostKeyChecking=no {command}"
        result = subprocess.run(
            scp_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"[SUCCESS][{pc_name}] scp completed successfully")
            return (pc_name, True, result.stdout, result.stderr)
        else:
            print(f"[ERROR][{pc_name}] scp failed with return code {result.returncode}")
            return (pc_name, False, result.stdout, result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR][{pc_name}] scp timed out")
        return (pc_name, False, "", "Command timed out")
    except Exception as e:
        print(f"[ERROR][{pc_name}] Exception occurred while executing scp: {str(e)}")
        return (pc_name, False, "", str(e))


def execute_ssh_command(pc_name, command, timeout=300, command_type="command", verbose=True):
    """
    Execute a command on a remote PC via SSH
    
    Args:
        pc_name (str): The PC hostname/username combination
        command (str): The command to execute
        command_type (str): Type of command for logging purposes
    
    Returns:
        tuple: (pc_name, success, output, error)
    """
    
    try:
        # Execute SSH command
        ssh_command = f"ssh -o StrictHostKeyChecking=no {pc_name} '{command}'"
        result = subprocess.run(
            ssh_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout  # 5 minute timeout
        )
        
        if result.returncode == 0:
            if verbose == True:
                print(f"[SUCCESS][{pc_name}] {command_type} completed successfully")
            return (pc_name, True, result.stdout, result.stderr)
        else:
            if verbose == True:
                print(f"[ERROR][{pc_name}] {command_type} failed with return code {result.returncode}")
            return (pc_name, False, result.stdout, result.stderr)
            
    except subprocess.TimeoutExpired:
        if verbose == True:
            print(f"[ERROR][{pc_name}] {command_type} timed out")
        return (pc_name, False, "", "Command timed out")
    except Exception as e:
        if verbose == True:
            print(f"[ERROR][{pc_name}] Exception occurred while executing {command_type}: {str(e)}")
        return (pc_name, False, "", str(e))

# Parse arguments
arguments_as_string = " ".join(sys.argv[1:])
arguments_as_string = arguments_as_string.strip().split("--")

OPERATORS = ''
SPLIT_ACROSS_PC_OEPRATORS = ''
EDGEAI_BENCHMARK_BRANCH = 'develop'
TIDL_TOOLS_TARBALL = ''
PULL_TIDL_MODELS = '0'
SETUP_EDGEAI_BENCHMARK = '0'
arguments_as_string_filtered = ''
for i in arguments_as_string:
    i = i.strip()
    if i.startswith("operators="):
        OPERATORS = i.split("=")[-1].strip()
    elif i.startswith("split_across_pc="):
        SPLIT_ACROSS_PC_OEPRATORS = i.split("=")[-1].strip()
    elif i.startswith("edgeai_benchmark_branch="):
        EDGEAI_BENCHMARK_BRANCH = i.split("=")[-1].strip()
    elif i.startswith("tidl_tools_path="):
        TIDL_TOOLS_TARBALL=i.split("=")[-1].strip()
    elif i.startswith("pull_tidl_models="):
        PULL_TIDL_MODELS=i.split("=")[-1].strip()
    elif i.startswith("setup_edgeai_benchmark="):
        SETUP_EDGEAI_BENCHMARK=i.split("=")[-1].strip()
    elif i != '' and not i.startswith("temp_buffer_dir=") and not i.startswith("temp_nc_dir=") and not i.startswith("num_threads="):
        arguments_as_string_filtered += f"--{i} "

# Open and read the JSON file
with open('distributed_test_config.json', 'r') as file:
    pc_config = json.load(file)  # Parse JSON data into a Python dictionary

# Access the data
print(f"[INFO] {len(pc_config)} PCs found.")

# Validate
print(f"[INFO] Running validation command for all PCs")
for pc_name in list(pc_config.keys()):
    pc_info = pc_config[pc_name]

    tidl_models_dir = pc_info['tidl_models_dir']
    validation_command = f"git -C {tidl_models_dir} rev-parse"
    validation_result = execute_ssh_command(pc_name, validation_command, timeout=300, command_type="validation_command", verbose=False)
    _, validation_success, validation_stdout, validation_stderr = validation_result
    if not validation_success:
        print(f"[WARNING][{pc_name}] Validation for tidl_models failed. Removing {pc_name} from executers.")
        del pc_config[pc_name]
        continue        

    edgeai_benchmark_dir = pc_info['edgeai_benchmark_dir']
    validation_command = f"git -C {edgeai_benchmark_dir} rev-parse"
    validation_result = execute_ssh_command(pc_name, validation_command, timeout=300, command_type="validation_command", verbose=False)
    _, validation_success, validation_stdout, validation_stderr = validation_result
    if not validation_success:
        print(f"[WARNING][{pc_name}] Validation for edgeai-benchmark failed. Removing {pc_name} from executers.")
        del pc_config[pc_name]

# Setup tidl_models and edgeai-benchmark if specified
executors_to_remove = []
def setup_pc(pc_name, pc_info):
    global executors_to_remove
    if PULL_TIDL_MODELS == '1':
        tidl_models_dir = pc_info['tidl_models_dir']
        tidl_models_pull_command = f"cd {tidl_models_dir} && git stash && git pull && git lfs pull"
        print(f"[INFO][{pc_name}] Pulling tip for tidl_models: {tidl_models_pull_command}")
        tidl_models_pull_result = execute_ssh_command(pc_name, tidl_models_pull_command, timeout=7200, command_type="tidl_models_pull_command")
        _, tidl_models_pull_success, tidl_models_pull_stdout, tidl_models_pull_stderr = tidl_models_pull_result
        if not tidl_models_pull_success:
            print(f"[WARNING][{pc_name}] Pulling tip for tidl_models failed.")
    
    if SETUP_EDGEAI_BENCHMARK == '1':
        edgeai_benchmark_dir = pc_info['edgeai_benchmark_dir']
        pyenv = pc_info['pyenv']
        eai_setup_command = f"cd {edgeai_benchmark_dir} && git fetch --all && git reset --hard origin/{EDGEAI_BENCHMARK_BRANCH} && source {pyenv} && export export HTTPS_PROXY=http://webproxy.ext.ti.com:80 && export https_proxy=http://webproxy.ext.ti.com:80 && export HTTP_PROXY=http://webproxy.ext.ti.com:80 && export http_proxy=http://webproxy.ext.ti.com:80 && export ftp_proxy=http://webproxy.ext.ti.com:80 &&  export FTP_PROXY=http://webproxy.ext.ti.com:80 && export no_proxy=ti.com  && ssh-keyscan -H bitbucket.itg.ti.com >> ~/.ssh/known_hosts && ./setup_pc.sh && pip3 install -r {edgeai_benchmark_dir}/tests/tidl_unit/requirements.txt"
        print(f"[INFO][{pc_name}] Setting up edgeai-benchmark: {eai_setup_command}")
        eai_setup_result = execute_ssh_command(pc_name, eai_setup_command, timeout=1800, command_type="eai_setup_command")
        _, eai_setup_success, eai_setup_stdout, eai_setup_stderr = eai_setup_result
        if not eai_setup_success:
            print(f"[WARNING][{pc_name}] Setting up edgeai-benchmark failed. Removing {pc_name} from executers.")
            executors_to_remove.append(pc_name)
        
        test_data_dir = os.path.join(edgeai_benchmark_dir, "tests/tidl_unit/tidl_unit_test_data")
        tidl_models_dir = pc_info['tidl_models_dir']
        soft_link_command = f"cd {test_data_dir} && ln -sf {tidl_models_dir}/unitTest/onnx/tidl_unit_test_assets/operators ./ && ln -sf {tidl_models_dir}/unitTest/onnx/tidl_unit_test_assets/configs ./"
        print(f"[INFO][{pc_name}] Soft linking tidl_models to tidl_unit_test_assets: {soft_link_command}")
        soft_link_result = execute_ssh_command(pc_name, soft_link_command, timeout=300, command_type="soft_link_command")
        _, soft_link_success, soft_link_stdout, soft_link_stderr = soft_link_result
        if not soft_link_success:
            print(f"[WARNING][{pc_name}] Soft linking tidl_models to tidl_unit_test_assets failed. Removing {pc_name} from executers.")
            executors_to_remove.append(pc_name)

if PULL_TIDL_MODELS == '1' or SETUP_EDGEAI_BENCHMARK == '1':
    active_threads = []
    for pc_name, pc_info in pc_config.items():
        # Create thread
        thread = threading.Thread(
            target=setup_pc,
            args=(pc_name, pc_info),
            name=f"{pc_name}"
        )
        thread.daemon = True
        active_threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    print(f"[INFO] Waiting for all {len(active_threads)} threads to complete the setup...")
    for thread in active_threads:
        thread.join()
    
    for pc_name in executors_to_remove:
        del pc_config[pc_name]

print(f"\nRunning on {len(pc_config)} PCs:")
for pc_name in pc_config:
    print(pc_name)
print("\n")

ALL_OPERATORS = ["Abs", "Acos", "Acosh", "Add", "ArgMax", "ArgMin", "Asin", "Asinh", "Atan", \
                "AveragePool", "BatchNormalization", "Clip", "Concat", "Conv", "ConvTranspose", \
                "Cos", "Cosh", "DepthToSpace", "DequantizeLinear", "Div", "Dropout", "Elu", \
                "Erf", "Exp", "Expand", "Flatten", "Floor", "Gather", "Gemm", "GlobalAveragePool", \
                "GridSample", "HardSigmoid", "HardSwish", "Identity", "InstanceNormalization", \
                "LayerNormalization", "LeakyRelu", "Log", "MatMul", "Max", "MaxPool", "Min", "Mish", \
                "Mul", "Neg", "Pad", "Pow", "PRelu", "QuantizeLinear", "ReduceMax", "ReduceMean", \
                "ReduceMin", "ReduceSum", "Relu", "Reshape", "Resize", "ScatterElements", "ScatterND", \
                "Sigmoid", "Sin", "Sinh", "Slice", "Softmax", "SpaceToDepth", "Sqrt", "Squeeze", "Sub", \
                "Sum", "Tan", "Tanh", "TopK", "Transpose", "Unsqueeze"]


if OPERATORS == '':
    OPERATORS = " ".join(ALL_OPERATORS)
OPERATORS = OPERATORS.split()
OPERATORS = list(set(OPERATORS))

SPLIT_ACROSS_PC_OEPRATORS = SPLIT_ACROSS_PC_OEPRATORS.split()

# Force operators to split across PC since they are huge tests
SPLIT_ACROSS_PC_OEPRATORS += ["Conv", "Add", "Mul", "Div", "Resize", "Squeeze", "Unsqueeze",
                              "Slice", "Transpose", "Reshape", "Sub", "InstanceNormalization"
                              "BatchNormalization", "ScatterND"]

SPLIT_ACROSS_PC_OEPRATORS = list(set(SPLIT_ACROSS_PC_OEPRATORS))

# Remove from SPLIT_ACROSS_PC_OEPRATORS if it is not in operators under test
for i in SPLIT_ACROSS_PC_OEPRATORS[:]:
    if i not in OPERATORS:
        SPLIT_ACROSS_PC_OEPRATORS.remove(i)

# Remove pc_specific_operators if it is not in operators under test
for pc, info in pc_config.items():
    if "pc_specific_operators" in pc_config[pc]:
        for i in pc_config[pc]["pc_specific_operators"][:]:
            if i not in OPERATORS:
                pc_config[pc]["pc_specific_operators"].remove(i)

# Remove unused pc to avoid creating unecessary threads
if len(SPLIT_ACROSS_PC_OEPRATORS) <= 0:
    pc_config_keys_to_remove = []
    for pc, info in pc_config.items():
        if "pc_specific_operators" not in info or len(info['pc_specific_operators']) < 1:
            pc_config_keys_to_remove.append(pc)
    for key in pc_config_keys_to_remove:
        if len(pc_config) <= len(OPERATORS):
            break
        del pc_config[key]

# Remove SPLIT_ACROSS_PC_OEPRATORS from general Operators
for i in SPLIT_ACROSS_PC_OEPRATORS:
    if i in OPERATORS:
        OPERATORS.remove(i)

print(f"[INFO] Total operators to process: {len(OPERATORS) + len(SPLIT_ACROSS_PC_OEPRATORS)}")
print(f"[INFO] Operators to run on single pc: {len(OPERATORS)} - {OPERATORS}")
print(f"[INFO] Operators to split across multiple pc: {len(SPLIT_ACROSS_PC_OEPRATORS)} - {SPLIT_ACROSS_PC_OEPRATORS}")

# Remove pc_specific_operators from SPLIT_ACROSS_PC_OEPRATORS and general operator queue
for pc, info in pc_config.items():
    if "pc_specific_operators" not in info:
        continue
    for i in info["pc_specific_operators"]:
        if i in OPERATORS:
            OPERATORS.remove(i)
        if i in SPLIT_ACROSS_PC_OEPRATORS:
            SPLIT_ACROSS_PC_OEPRATORS.remove(i)

# Query for all tests in SPLIT_ACROSS_PC_OEPRATORS by looking into one of the PC
SPLIT_ACROSS_PC_OEPRATORS_TESTS=[]
for i in SPLIT_ACROSS_PC_OEPRATORS[:]:
    pc_name = list(pc_config.keys())[0]
    test_dir = os.path.join(pc_config[pc_name]["edgeai_benchmark_dir"], "tests/tidl_unit/internal")
    query_command = "find " + test_dir + "/../tidl_unit_test_data/operators/" + i +" -maxdepth 1 -type d -not -name '.' -exec basename {} \;"
    print(f"[INFO][{pc_name}] Running Query command for {i}: {query_command}")
    query_result = execute_ssh_command(pc_name, query_command, timeout=300, command_type="query_command")
    _, query_success, query_stdout, query_stderr = query_result
    if query_success:
        temp = []
        query_stdout = query_stdout.strip().split('\n')
        for j in query_stdout:
            j = j.strip()
            if j.startswith(i) and i != j:
                temp.append(j)
        if len(temp) > 0:
            SPLIT_ACROSS_PC_OEPRATORS_TESTS.append(temp)
    else:
        print(f"[WARNING][{pc_name}] Query for {i} failed. Not splitting across PC.")
        SPLIT_ACROSS_PC_OEPRATORS.remove(i)
        OPERATORS.append(i)

for i in range(len(SPLIT_ACROSS_PC_OEPRATORS_TESTS)):
    n = len(pc_config)
    chunk_size = max(1, len(SPLIT_ACROSS_PC_OEPRATORS_TESTS[i]) // n)
    split_list = []

    SPLIT_ACROSS_PC_OEPRATORS_TESTS[i] = [SPLIT_ACROSS_PC_OEPRATORS_TESTS[i][j:j + chunk_size] for j in range(0, len(SPLIT_ACROSS_PC_OEPRATORS_TESTS[i]), chunk_size)]

# General operator queue
operators_queue = OPERATORS.copy()
operators_lock = threading.Lock()
def get_next_operator():
    """
    Get the next operator from the queue in a thread-safe manner

    Returns:
        str or None: Next operator to process, or None if queue is empty
    """
    global operators_queue, operators_lock

    with operators_lock:
        if operators_queue:
            operator = operators_queue.pop(0)
            return operator
        else:
            return None

split_across_pc_operators_queue = SPLIT_ACROSS_PC_OEPRATORS_TESTS.copy()
split_across_pc_operators_lock = threading.Lock()
def get_next_split_across_pc_operator():
    """
    Get the next operator from the queue in a thread-safe manner

    Returns:
        str or None: Next operator to process, or None if queue is empty
    """
    global split_across_pc_operators_queue, split_across_pc_operators_lock

    with split_across_pc_operators_lock:
        if split_across_pc_operators_queue and len(split_across_pc_operators_queue) > 0:
            if len(split_across_pc_operators_queue[0]) <= 0:
                split_across_pc_operators_queue.pop(0)
            if split_across_pc_operators_queue and split_across_pc_operators_queue[0] and len(split_across_pc_operators_queue[0]) > 0:
                operator = split_across_pc_operators_queue[0].pop(0)
                return operator
            else:
                return None
        else:
            return None

# Global variables for thread synchronization
pc_results = {}
results_lock = threading.Lock()
all_threads_completed = threading.Event()
active_threads = []


def execute_pc_commands(pc_name, pc_info, log_path, result_path):
    """
    Execute setup command once, then continuously process operators from the queue
    Commands are created dynamically within this thread function
    
    Args:
        pc_name (str): The PC hostname/username combination
        pc_info (dict): Dictionary containing metadata
    """
    global pc_results, arguments_as_string_filtered

    print()
    print(f"[INFO][{pc_name}] Starting thread..")

    # Initialize results for this PC
    with results_lock:
        pc_results[pc_name] = {
            'setup_success': False,
            'setup_output': '',
            'setup_error': '',
            'setup_time': '',
            'operators_info': [],
        }
    
    try:
        # Create setup command dynamically
        test_dir = os.path.join(pc_config[pc_name]["edgeai_benchmark_dir"], "tests/tidl_unit/internal")
        pyenv = pc_info["pyenv"]
        setup_command = f"cd {test_dir}/../ && git clean -fxd -e 'tidl_unit_test_data' && cd {test_dir} && git stash && git checkout {EDGEAI_BENCHMARK_BRANCH} && git fetch && git pull --rebase && rm -rf {test_dir}/operator_test_reports/*"
        if  TIDL_TOOLS_TARBALL != '':
            setup_command = f"{setup_command} && rm -rf tidl_tools_tarball && wget -q -O tidl_tools_tarball {TIDL_TOOLS_TARBALL}"
        print(f"[INFO][{pc_name}] Running Setup command : {setup_command}")
        setup_start_time = datetime.now()
        setup_result = execute_ssh_command(pc_name, setup_command, timeout=1200, command_type="setup_command")
        setup_end_time = datetime.now()

        pc_name_result, setup_success, setup_stdout, setup_stderr = setup_result
        setup_time = setup_end_time - setup_start_time
        setup_time = format_seconds_to_hhmmss(setup_time.total_seconds())

        with open(os.path.join(log_path, "setup_stdout.log"), "w+") as f:
            f.write(f"Command: {setup_command}")
            f.write("\n")
            f.write(f"Total Time: {setup_time}")
            f.write("\n")
            f.write(setup_stdout)
        with open(os.path.join(log_path, "setup_stderr.log"), "w+") as f:
            f.write(f"Command: {setup_command}")
            f.write("\n")
            f.write(f"Total Time: {setup_time}")
            f.write("\n")
            f.write(setup_stderr)

        # Update results
        with results_lock:
            pc_results[pc_name]['setup_success'] = setup_success
            pc_results[pc_name]['setup_output'] = setup_stdout
            pc_results[pc_name]['setup_error'] = setup_stderr
            pc_results[pc_name]['setup_time'] = setup_time
        
        if setup_success:
            print(f"[SUCCESS][{pc_name}] Setup completed, starting operator test. Time taken: {setup_time}")
            
            # Process operators from the queue until empty
            operators_run = {}
            while True:
                operator = None
                split_across_pc_operator = False

                if "pc_specific_operators" in pc_info and len(pc_info["pc_specific_operators"]) > 0:
                    operator = pc_info["pc_specific_operators"].pop(0)
                else:
                    # Get from general operator queue
                    operator = get_next_operator()

                    # Get from split_across_pc queue if general operator queue is done
                    if operator is None:
                        operator = get_next_split_across_pc_operator()
                        if operator is None or len(operator) <= 0:
                            operator = None
                        else:
                            print(f"[INFO][{pc_name}] Running {operator}")
                            split_across_pc_operator = True

                if operator is None:
                    print(f"[INFO][{pc_name}] No more operators to process")
                    break

                if split_across_pc_operator == True:
                    all_test = '\n'.join(operator)
                    all_test = all_test.strip()
                    test_file_command = f"rm -rf {test_dir}/test_file.txt && echo \"{all_test}\" >> {test_dir}/test_file.txt"
                    print(f"[INFO][{pc_name}] Executing test_file command.")

                    test_file_result = execute_ssh_command(pc_name, test_file_command, timeout=300, command_type=f"test_file_command")
                    _, test_file_success, test_file_stdout, test_file_stderr = test_file_result

                    operator_name = operator[0].strip().split('_')[0]
                    if operator_name not in operators_run:
                        operators_run[operator_name] = 1
                        count = 1
                    else:
                        operators_run[operator_name] += 1
                        count = operators_run[operator_name]
                    operator = f"{operator_name}_Chunk_{count}"
                else:
                    operator_name = operator
                    operators_run[operator_name] = 1

                
                print(f"[INFO][{pc_name}] Processing : {operator}")
                
                # Create test command dynamically for this operator
                args = f"{arguments_as_string_filtered}"
                if "temp_buffer_dir" in pc_info:
                    temp_buffer_dir = pc_info["temp_buffer_dir"]
                    args = f"{args} --temp_buffer_dir={temp_buffer_dir}" 
                if "temp_nc_dir" in pc_info:
                    temp_nc_dir = pc_info["temp_nc_dir"]
                    args = f"{args} --temp_nc_dir={temp_nc_dir}"
                if "num_threads" in pc_info:
                    num_threads = pc_info["num_threads"]
                    args = f"{args} --num_threads={num_threads}"

                if  TIDL_TOOLS_TARBALL != '':
                    args = f"{args} --tidl_tools_path={test_dir}/tidl_tools_tarball"

                if split_across_pc_operator == True:
                    args = f"{args} --test_file={test_dir}/test_file.txt"

                args = f"{args} --operators={operator_name}"

                test_command = f"cd {test_dir} && source {pyenv} && ./run_operator_test.sh {args}"
                print(f"[INFO][{pc_name}] Executing test command: {test_command}")

                test_start_time = datetime.now()
                test_result = execute_ssh_command(pc_name, test_command, timeout=36000, command_type=f"test_command_{operator}")
                test_end_time = datetime.now()

                pc_name_result, test_success, test_stdout, test_stderr = test_result
                test_time = test_end_time - test_start_time
                test_time = format_seconds_to_hhmmss(test_time.total_seconds())

                if split_across_pc_operator == True:
                    test_result_path = os.path.join(result_path, "chunk_reports", operator)
                    if os.path.exists(test_result_path):
                        try:
                            shutil.rmtree(test_result_path)
                        except OSError as e:
                            pass
                    os.makedirs(test_result_path, exist_ok=True)
                else:
                    test_result_path = result_path

                scp_command = f"{pc_name}:{test_dir}/operator_test_reports {test_result_path}"
                print(f"[INFO][{pc_name}] Executing scp command: {scp_command}")
                scp_result = execute_scp_command(pc_name, scp_command)
                pc_name_result, scp_success, scp_stdout, scp_stderr = scp_result
                
                # Update results
                with results_lock:
                    if test_success:
                        pc_results[pc_name]['operators_info'].append((operator, test_time, "YES"))
                        print(f"[SUCCESS][{pc_name}] {operator} completed. Time taken: {test_time}.")
                    else:
                        pc_results[pc_name]['operators_info'].append((operator, test_time, "NO"))
                        print(f"[ERROR][{pc_name}] {operator} failed. Time taken: {test_time}")
                
                with open(os.path.join(log_path, f"{operator}_stdout.log"), "w+") as f:
                    f.write(f"Command: {test_command}")
                    f.write("\n")
                    f.write(f"Total Time: {test_time}")
                    f.write("\n")
                    f.write(test_stdout)
                with open(os.path.join(log_path, f"{operator}_stderr.log"), "w+") as f:
                    f.write(f"Command: {test_command}")
                    f.write("\n")
                    f.write(f"Total Time: {test_time}")
                    f.write("\n")
                    f.write(test_stderr)
 
        else:
            print(f"[ERROR][{pc_name}] Setup failed, cannot process any operators")
            
    except Exception as e:
        print(f"[ERROR][{pc_name}] Exception in thread: {str(e)}")
        with results_lock:
            pc_results[pc_name]['setup_error'] = str(e)
    
    print(f"[INFO][{pc_name}] Thread completed")


def execute_all_pc_commands(pc_config):
    """
    Create and manage threads for all PCs to execute setup and test commands
    
    Args:
        pc_config (dict): Dictionary containing PC names as keys and some metadata relate to it
    """
    global active_threads, pc_results
    
    print(f"[INFO] Starting parallel execution on {len(pc_config)} PCs...")
    print("[INFO] Each PC will execute setup command followed by test command")

    # Delete existing logs and results
    output_directory_path = f"distributed_test_output"
    if os.path.exists(output_directory_path):
        try:
            shutil.rmtree(output_directory_path)
        except OSError as e:
            pass
    print(f"[INFO] Logs and results will be saved at {output_directory_path}")
    
    # Create and start threads for each PC
    active_threads = []
    for pc_name, pc_info in pc_config.items():

        # Create output directory
        logs_directory_path = os.path.join(output_directory_path, pc_name, "logs")
        results_directory_path = os.path.join(output_directory_path, pc_name)
        os.makedirs(logs_directory_path, exist_ok=True)
        os.makedirs(results_directory_path, exist_ok=True)

        # Create thread
        thread = threading.Thread(
            target=execute_pc_commands,
            args=(pc_name, pc_info, logs_directory_path, results_directory_path),
            name=f"{pc_name}"
        )
        thread.daemon = True  # Allow main program to exit even if threads are running
        active_threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    print(f"[INFO] Waiting for all {len(active_threads)} threads to complete...")
    for thread in active_threads:
        thread.join()
    
    print("[INFO] All PC threads have completed")
    
    # Print summary of results
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    all_setup_successful = True
    total_test_processed = 0
    total_test_failed = 0
    
    for pc_name, result in pc_results.items():
        setup_status = "SUCCESS" if result['setup_success'] else "FAILED"
        setup_time = result['setup_time']

        num_operators_processed = 0
        num_operators_failed = 0
        operators_processed = []
        operators_failed = []
        for i in result['operators_info']:
            if i[2].strip().upper() == "YES":
                num_operators_processed += 1
                operators_processed.append(i[0])
            else:
                num_operators_failed += 1
                operators_failed.append(i[0])

        total_test_processed += num_operators_processed
        total_test_failed += num_operators_failed
        
        print(f"{pc_name}:")
        print(f"  Setup: {setup_status}")
        print(f"  Setup Time: {setup_time}")
        print(f"  Total Operators Processed Successfully: {num_operators_processed}")
        print(f"  Total Operators Not Processed Successfully: {num_operators_failed}")
        
        if len(operators_processed) > 0:
            print(f"  Successful Operators: {', '.join(operators_processed)}")
        if len(operators_failed) > 0:
            print(f"  Unsuccessful Operators: {', '.join(operators_failed)}")

        operator_info_header = ["OPERATOR", "TIME", "RAN SUCCESSFULLY"]
        table = tabulate(result['operators_info'], headers=operator_info_header, tablefmt="grid")
        indented_table = ""
        for line in table.splitlines():
            indented_table += "    " + line + "\n"
        print(f"  Operators Info:")
        print(indented_table)

        if not result['setup_success']:
            all_setup_successful = False
            if result['setup_error']:
                print(f"  Setup Error: {result['setup_error']}")
        
        print()


    print(f"OVERALL STATISTICS:")
    print(f"  Total Test: {total_test_processed + total_test_failed}")
    print(f"  Total Test Ran Successfully: {total_test_processed}")
    print(f"  Total Test Did Not Run Successfully: {total_test_failed}")
    
    status = False
    if not all_setup_successful:
        print("[ERROR] Some setup commands failed")
        status = False
    elif total_test_failed == 0:
        print("[SUCCESS] All operators processed successfully on all PCs!")
        status = True
    else:
        print(f"[PARTIAL SUCCESS] {total_test_processed} operators succeeded, {total_test_failed} operators failed")
        status = False

    print(f"COLLATING REPORTS:")
    collate_directory_path = os.path.join(output_directory_path, "operator_test_reports")
    if os.path.exists(collate_directory_path):
        try:
            shutil.rmtree(collate_directory_path)
        except OSError as e:
            pass
    print(f"[INFO] Collated results will be saved at {collate_directory_path}")

    chunk_reports = {}

    for i in os.listdir(output_directory_path):
        pc_dir = os.path.join(output_directory_path, i)

        # Copy operator test reports from all PC directories
        for j in os.listdir(pc_dir):
            if j == "operator_test_reports":
                operator_test_reports_dir = os.path.join(pc_dir, "operator_test_reports")
                if os.path.exists(operator_test_reports_dir):
                    try:
                        shutil.copytree(operator_test_reports_dir, collate_directory_path, dirs_exist_ok=True)
                    except shutil.Error as e:
                        print(f"Error copying directory: {e}")

        # Store path of all chunk reports per operator
        for j in os.listdir(pc_dir):
            if j == "chunk_reports":
                chunk_reports_dir = os.path.join(pc_dir, "chunk_reports")
                for k in os.listdir(chunk_reports_dir):
                    operator_dir = os.path.join(chunk_reports_dir, k)
                    operator = k.strip().split("_")[0]

                    if operator not in chunk_reports:
                        chunk_reports[operator] = [operator_dir]
                    else:
                        chunk_reports[operator].append(operator_dir)

    # Copy over chunk reports to operator_test_reports
    for operator, chunk_paths in chunk_reports.items():
        count = 0
        for chunk_path in chunk_paths:
            for dirpath, dirnames, filenames in os.walk(chunk_path):
                for dirname in dirnames:
                    if dirname == operator:
                        full_path = os.path.join(dirpath, dirname)
                        path_obj = Path(full_path)
                        parts = path_obj.parts
                        index_of_target = parts.index("operator_test_reports")
                        directories_after_target = parts[index_of_target + 1:]
                        result_path = Path(*directories_after_target)
                        result_path_new = result_path.parent / f"{result_path.parts[-1]}_Chunk_{count}"
                        count += 1
                        copy_path = os.path.join(collate_directory_path, result_path_new)
                        try:
                            shutil.copytree(full_path, copy_path, dirs_exist_ok=True)
                        except shutil.Error as e:
                            print(f"Error copying directory: {e}")

    # Merge HTML reports for chunks
    dir_with_merge_html = []
    for dirpath, dirnames, filenames in os.walk(collate_directory_path):
        if not dirnames:
            parent_dir = Path(dirpath).parent
            basename = os.path.basename(dirpath)
            if "_Chunk_" in basename:
                operator = basename.strip().split('_')[0]
                new_parent_dir = os.path.join(parent_dir, operator)
                os.makedirs(new_parent_dir, exist_ok=True)
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    file_name = os.path.basename(file_path)

                    temp_dir = file_name.strip().split('.')[0]
                    temp_dir = os.path.join(new_parent_dir, temp_dir)
                    os.makedirs(temp_dir, exist_ok=True)

                    new_file_name = f"{basename}_{file_name}"
                    new_file_path = os.path.join(temp_dir, new_file_name)
                    shutil.copy(file_path, new_file_path)

                    dir_with_merge_html.append(temp_dir)

                shutil.rmtree(dirpath)

    dir_with_merge_html = list(set(dir_with_merge_html))
    for dir in dir_with_merge_html:

        # Only merge if the dir has html files
        merge = True
        for file in os.listdir(dir):
            if os.path.splitext(file)[-1].strip().lower() != '.html':
                merge = False
                break
        if merge == False:
            continue

        parent_dir = Path(dir).parent
        basename = os.path.basename(dir)
        output_path = os.path.join(parent_dir, f"{basename}.html")
        command = f"pytest_html_merger -i {dir} -o {output_path}"
        result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
        if result.returncode != 0:
            print(f"HTML Merge Command \"{command}\" failed with return code {result.returncode}")
            print(f"Standard output:\n{result.stdout}")
            print(f"Standard error:\n{result.stderr}")
        else:
            shutil.rmtree(dir)

    return status

# Execute commands on all PCs in parallel
print("\n" + "="*60)
print("EXECUTING SETUP AND TEST COMMANDS ON ALL PCs")
print("="*60)

success = execute_all_pc_commands(pc_config)

end_time = datetime.now()
time_difference = end_time - start_time
total_time = format_seconds_to_hhmmss(time_difference.total_seconds())
print(f"Total time: {total_time}")

if success:
    print("\n[SUCCESS] Distributed test execution completed successfully!")
else:
    print("\n[ERROR] Distributed test execution completed with errors!")
    sys.exit(1)
