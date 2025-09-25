import json
import sys
import subprocess
import threading
import time
import os
import shutil

# Open and read the JSON file
with open('distributed_test_config.json', 'r') as file:
    pc_config = json.load(file)  # Parse JSON data into a Python dictionary

# Access the data
print(f"[INFO] {len(pc_config)} PCs found. Distributing tests across them...")

ALL_OPERATORS = ["Abs", "Acos", "Acosh", "Add", "ArgMax", "Asin", "Asinh", "Atan", \
                "AveragePool", "BatchNorm", "Clip", "Concat", "Convolution", "ConvTranspose", \
                "Cos", "Cosh", "DepthToSpace", "DequantizeLinear", "Div", "Dropout", "Elu", \
                "Erf", "Exp", "Flatten", "Floor", "Gather", "Gemm", "GlobalAveragePool", \
                "GridSample", "HardSigmoid", "HardSwish", "Identity", "InstanceNormalization", \
                "LayerNormalization", "LeakyRelu", "Log", "MatMul", "Max", "MaxPool", "Mish", "Mul", \
                "Neg", "Pad", "Pow", "PRelu", "QuantizeLinear", "ReduceMax", "ReduceMean", "ReduceMin", \
                "Relu", "Reshape", "Resize", "ScatterElements", "ScatterND", "Sigmoid", "Sin", "Sinh", "Slice", \
                "Softmax", "SpaceToDepth", "Sqrt", "Squeeze", "Sub", "Sum", "Tan", "Tanh", "TopK", "Transpose", \
                "Unsqueeze"]

arguments_as_string = " ".join(sys.argv[1:])
arguments_as_string = arguments_as_string.strip().split("--")

OPERATORS = ''
TIDL_TOOLS_TARBALL = ''
arguments_as_string_filtered = ''
for i in arguments_as_string:
    i = i.strip()
    if i.startswith("operators="):
        OPERATORS = i.split("=")[-1].strip()
    elif i.startswith("tidl_tools_path="):
        TIDL_TOOLS_TARBALL=i.split("=")[-1].strip()
    elif i != '' and not i.startswith("temp_buffer_dir=") and not i.startswith("temp_nc_dir=") and not i.startswith("num_threads="):
        arguments_as_string_filtered += f"--{i} "

if OPERATORS == '':
    OPERATORS = " ".join(ALL_OPERATORS)

OPERATORS = OPERATORS.split()
print(f"[INFO] Total operators to process: {len(OPERATORS)} - {OPERATORS}")

# Global variables for thread synchronization and operator management
operators_queue = OPERATORS.copy()  # Create a copy for thread-safe operations
operators_lock = threading.Lock()   # Lock for accessing the operators queue

while len(OPERATORS) < len(pc_config):
    pc_config.popitem()

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

def execute_scp_command(pc_name, command, timeout=300):
    """
    Copy from remote PC via SCP
    
    Args:
        pc_name (str): The PC hostname/username combination (e.g., "tidl@tidl-said-pc02.dhcp.ti.com")
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


def execute_ssh_command(pc_name, command, timeout=300, command_type="command"):
    """
    Execute a command on a remote PC via SSH
    
    Args:
        pc_name (str): The PC hostname/username combination (e.g., "tidl@tidl-said-pc02.dhcp.ti.com")
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
            print(f"[SUCCESS][{pc_name}] {command_type} completed successfully")
            return (pc_name, True, result.stdout, result.stderr)
        else:
            print(f"[ERROR][{pc_name}] {command_type} failed with return code {result.returncode}")
            return (pc_name, False, result.stdout, result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR][{pc_name}] {command_type} timed out")
        return (pc_name, False, "", "Command timed out")
    except Exception as e:
        print(f"[ERROR][{pc_name}] Exception occurred while executing {command_type}: {str(e)}")
        return (pc_name, False, "", str(e))


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
            'operators_processed': [],
            'operators_failed': [],
            'setup_output': '',
            'setup_error': ''
        }
    
    try:
        # Create setup command dynamically
        test_dir = pc_info["test_dir"]
        pyenv = pc_info["pyenv"]
        
        setup_command = f"cd {test_dir} && git stash && git checkout 2025/gourav_unit && git fetch && git pull"
        if  TIDL_TOOLS_TARBALL != '':
            setup_command = f"{setup_command} && rm -rf tidl_tools_tarball && wget -q -O tidl_tools_tarball {TIDL_TOOLS_TARBALL}"

        print(f"[INFO][{pc_name}] Running Setup command : {setup_command}")
        setup_result = execute_ssh_command(pc_name, setup_command, timeout=1200, command_type="setup_command")
        pc_name_result, setup_success, setup_stdout, setup_stderr = setup_result

        with open(os.path.join(log_path, "setup_stdout.log"), "w+") as f:
            f.write(f"Command: {setup_command}")
            f.write("\n")
            f.write(setup_stdout)
        with open(os.path.join(log_path, "setup_stderr.log"), "w+") as f:
            f.write(f"Command: {setup_command}")
            f.write("\n")
            f.write(setup_stderr)

        # Update results
        with results_lock:
            pc_results[pc_name]['setup_success'] = setup_success
            pc_results[pc_name]['setup_output'] = setup_stdout
            pc_results[pc_name]['setup_error'] = setup_stderr
        
        if setup_success:
            print(f"[SUCCESS][{pc_name}] Setup completed, starting operator test")
            
            # Process operators from the queue until empty
            while True:
                operator = get_next_operator()
                if operator is None:
                    print(f"[INFO][{pc_name}] No more operators to process")
                    break
                
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
                args = f"{args} --operators={operator}"

                test_command = f"cd {test_dir} && source {pyenv} && ./run_operator_test.sh {args}"
                print(f"[INFO][{pc_name}] Executing test command: {test_command}")
                test_result = execute_ssh_command(pc_name, test_command, timeout=36000, command_type=f"test_command_{operator}")
                pc_name_result, test_success, test_stdout, test_stderr = test_result

                scp_command = f"{pc_name}:{test_dir}/operator_test_reports {result_path}"
                print(f"[INFO][{pc_name}] Executing scp command: {scp_command}")
                scp_result = execute_scp_command(pc_name, scp_command)
                pc_name_result, scp_success, scp_stdout, scp_stderr = scp_result
                
                # Update results
                with results_lock:
                    if test_success:
                        pc_results[pc_name]['operators_processed'].append(operator)
                        print(f"[SUCCESS][{pc_name}] {operator} completed.")
                    else:
                        pc_results[pc_name]['operators_failed'].append(operator)
                        print(f"[ERROR][{pc_name}] {operator} failed")
                
                with open(os.path.join(log_path, f"{operator}_stdout.log"), "w+") as f:
                    f.write(f"Command: {test_command}")
                    f.write("\n")
                    f.write(test_stdout)
                with open(os.path.join(log_path, f"{operator}_stderr.log"), "w+") as f:
                    f.write(f"Command: {test_command}")
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
    total_operators_processed = 0
    total_operators_failed = 0
    
    for pc_name, result in pc_results.items():
        setup_status = "SUCCESS" if result['setup_success'] else "FAILED"
        operators_processed = len(result['operators_processed'])
        operators_failed = len(result['operators_failed'])
        
        total_operators_processed += operators_processed
        total_operators_failed += operators_failed
        
        print(f"{pc_name}:")
        print(f"  Setup: {setup_status}")
        print(f"  Operators Processed: {operators_processed}")
        print(f"  Operators Failed: {operators_failed}")
        
        if result['operators_processed']:
            print(f"  Successful Operators: {', '.join(result['operators_processed'])}")
        if result['operators_failed']:
            print(f"  Failed Operators: {', '.join(result['operators_failed'])}")
        
        if not result['setup_success']:
            all_setup_successful = False
            if result['setup_error']:
                print(f"  Setup Error: {result['setup_error']}")
        
        print()
    
    print(f"OVERALL STATISTICS:")
    print(f"  Total Operators Processed Successfully: {total_operators_processed}")
    print(f"  Total Operators Failed: {total_operators_failed}")
    print(f"  Total Operators: {total_operators_processed + total_operators_failed}")
    
    if not all_setup_successful:
        print("[ERROR] Some setup commands failed")
        return False
    elif total_operators_failed == 0:
        print("[SUCCESS] All operators processed successfully on all PCs!")
        return True
    else:
        print(f"[PARTIAL SUCCESS] {total_operators_processed} operators succeeded, {total_operators_failed} operators failed")
        return False


# Execute commands on all PCs in parallel
print("\n" + "="*60)
print("EXECUTING SETUP AND TEST COMMANDS ON ALL PCs")
print("="*60)

success = execute_all_pc_commands(pc_config)

if success:
    print("\n[SUCCESS] Distributed test execution completed successfully!")
else:
    print("\n[ERROR] Distributed test execution completed with errors!")
    sys.exit(1)
