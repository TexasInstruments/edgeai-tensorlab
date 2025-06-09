"""
Main script to trigger tests on EVM
"""

import os
import sys
import time
import json
import traceback
import argparse
import wget
import shutil
import tarfile
import socket

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from src.benchmark_evm import BenchmarkEvm
from src.tidl_unit_test_evm import TIDLUnitTestEVM

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

parser.add_argument('--test_suite', help='TEST_SUITE (BENCHMARK|TIDL_UNIT_TEST)', type=str, default="BENCHMARK")

# General arguments
parser.add_argument('--soc', help='SOC (AM68A|AM69A|TDA4VM|AM62A|AM67A)', type=str, default="AM68A")
parser.add_argument('--uart', help='UART port', type=str, default="/dev/ttyUSB2")
parser.add_argument('--logs_dir', help='Optional path to store evm test logs', type=str, default='./evm_test_logs')
parser.add_argument('--pc_ip', help='IP address of the pc', type=str, default=None)
parser.add_argument('--dataset_dir', help='Full path of the dataset from pc', type=str, required=True)
parser.add_argument('--tensor_bits', help='Optional argument to tensor bits', type=str, default='8')
parser.add_argument('--artifacts_folder', help='Optional path to store the model artifacts after EVM run', type=str, default=None)


# EVM arguments
parser.add_argument('--evm_timeout', help='Optional argument to set the timeout for a single test', type=int, default=600)
parser.add_argument('--evm_local_ip', help='Optional argument to set local ip for evm', type=str, default="")
parser.add_argument('--reboot_type', help='Reboot type (hard|soft)', type=str, default="soft")
parser.add_argument('--relay_type', help='Type of ethernet controlled power relay - (ANEL|DLI)', type=str, default="ANEL")
parser.add_argument('--relay_trigger_mechanism', help='ANEL Triggering mechanism - (EXE|LIB)', type=str, default="LIB")
parser.add_argument('--relay_exe_path', help='ANEL power switch exe path. Needed if relay_trigger_mechanism is EXE', type=str, default=None)
parser.add_argument('--relay_ip_address', help='ANEL power switch IP address', type=str, default=None)
parser.add_argument('--relay_power_port', help='ANEL power switch port number', type=str, default=None)

# Benchmark test specific arguments
parser.add_argument('--num_frames', help='The number of frames to run the evaluation', type=str, default=None)
parser.add_argument('--artifacts_tarball', help='Optional path to model artifacts tarball', type=str, default=None)
parser.add_argument('--session_type_dict', help='Optional argument to set runtime to model type mapping', type=str, default="\"{'onnx':'onnxrt' ,'tflite':'tflitert' ,'mxnet':'tvmdlr'}\"")

# TIDL_UNIT_TEST specific arguments
parser.add_argument('--operators', help='Space separated operators to test in TIDL_UNIT_TEST', type=str, default=None)

args = parser.parse_args()

status = True
edgeai_benchmark_path = os.path.join(CURRENT_DIR,"../../")

# Get ip address of the PC
ip_address = args.pc_ip
if ip_address == None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.0.0.0', 0))
        ip_address = s.getsockname()[0]

# Verify args
args.test_suite = args.test_suite.strip().upper()
if args.test_suite not in ("BENCHMARK","TIDL_UNIT_TEST"):
    print(f"[ Error ] Invalid Test suite: {args.test_suite}")
    sys.exit(-1)

args.soc = args.soc.strip().upper()
if args.soc not in ("AM68A","AM69A","TDA4VM","AM62A","AM67A"):
    print(f"[ Error ] Invalid SOC: {args.soc}")
    sys.exit(-1)

if ':' not in args.dataset_dir:
    if not os.path.exists(args.dataset_dir):
        print(f"[ Error ] {args.dataset_dir} does not exist")
        sys.exit(-1)

    if not os.path.isabs(args.dataset_dir):
        print(f"[ Error ] {args.dataset_dir} is not absolute path, Please provide this as an absolute path")
        sys.exit(-1)
else:
    print(f"[ Warning ] dataset_dir is defined as {args.dataset_dir}. Make sure this is valid since check is skipped")

if not os.path.exists(args.uart):
    print(f"[ Error ] {args.uart} does not exist")
    sys.exit(-1)

if args.reboot_type == "hard":
    if args.relay_ip_address == None:
        print(f"[ Error ] Please provide relay IP address")
        sys.exit(-1)
    elif args.relay_power_port == None:
        print(f"[ Error ] Please provide relay power port")
        sys.exit(-1)

# Create config for UART Interface
evm_config = { "soc" : args.soc,
                "dut_uart_info" : {
                    "port": args.uart,
                    "baudrate":115200,
                    "bytesize":8,
                    "parity":"N",
                    "stopbits":1,
                    "timeout":None,
                    "xonxoff":0,
                    "rtscts":0
                },
                "relay_info" : {
                    "relay_type" : args.relay_type,
                    "executable_path": args.relay_exe_path,
                    "ip_address": args.relay_ip_address,
                    "power_port": args.relay_power_port,
                    "relay_trigger_mechanism": args.relay_trigger_mechanism
                }
            }

args.logs_dir = f"{args.logs_dir}/{args.test_suite}/{evm_config['soc']}"
# Print info
print()
print("*" * 80)
print(f"TEST_SUITE      :   {args.test_suite}") 
print(f"SOC             :   {evm_config['soc']}") 
print(f"UART PORT       :   {evm_config['dut_uart_info']['port']}")
print(f"PC IP ADDRESS   :   {ip_address}")
print(f"LOGS DIR        :   {args.logs_dir}")
print()
print(f"REBOOT TYPE     :   {args.reboot_type}")
if args.reboot_type == "hard":
    print(f"RELAY EXE PATH  :   {args.relay_exe_path}")
    print(f"RELAY IP        :   {args.relay_ip_address}")
    print(f"RELAY PORT      :   {args.relay_power_port}")
print()
if args.test_suite == "BENCHMARK":
    print(f"MODEL ARTIFACTS DIR :   {args.artifacts_folder}") 
    print(f"ARTIFACTS TARBALL   :   {args.artifacts_tarball}")
elif args.test_suite == "TIDL_UNIT_TEST":
    print(f"MODEL ARTIFACTS DIR :   {args.artifacts_folder}") 
print("*" * 80)
print()

# Cleanup and create logs directory
try:
    shutil.rmtree(f'{args.logs_dir}')
except:
    pass
try:
    print(f'[ Info ] Creating {args.logs_dir} directory.')
    os.makedirs(f'{args.logs_dir}',exist_ok=True)
except:
    print(f'[ Error ] Cannot create {args.logs_dir} directory.')
    sys.exit(-1)


######################### BENCHMARK TESTS ######################################
if (args.test_suite == "BENCHMARK"):
    # Model artifacts folder
    if args.artifacts_folder is None:
        artifacts_dir = os.path.join(edgeai_benchmark_path, "work_dirs", "modelartifacts", evm_config["soc"], f"{args.tensor_bits}bits")
    else:
        artifacts_dir = args.artifacts_folder

    if (args.artifacts_tarball != None):
        args.artifacts_tarball = args.artifacts_tarball.strip()
        
        # Download tarball if http link provided
        if args.artifacts_tarball.startswith("http"):
            tarball_name =  args.artifacts_tarball.split("/")[-1]
            if os.path.exists(tarball_name):
                print(f"[ Warning ] {tarball_name} already exist reusing.")
                args.artifacts_tarball = tarball_name
            else:
                print(f"[ Info ] Downloading {args.artifacts_tarball}...")
                try:
                    wget.download(args.artifacts_tarball)
                    args.artifacts_tarball = tarball_name
                except Exception as e:
                    print(e)
                    sys.exit(-1)
        # Use local tarball
        else:
            if not os.path.exists(args.artifacts_tarball):
                print(f"[ Error ] {args.artifacts_tarball} does not exist")
                sys.exit(-1) 

        # Make artifacts directory is it does not already exist
        if not os.path.exists(artifacts_dir):
            print(f"[ Info ] Creating {artifacts_dir}...")
            try:
                os.makedirs(artifacts_dir)
            except Exception as e:
                print(e)
                sys.exit(-1)
        # Clean up artifacts directory to tarball artifacts
        else:
            print(f"[ Info ] Cleaning up {artifacts_dir}...")
            shutil.rmtree(artifacts_dir)

        # Untar the tarball into artifacts folder
        try:
            print(f"[ Info ] Untaring  {args.artifacts_tarball} to {artifacts_dir}... this might take some time")
            tarfile = tarfile.open(args.artifacts_tarball) 
            tarfile.extractall(artifacts_dir) 
            tarfile.close()
            os.system(f"cd {artifacts_dir} && bash ./extract.sh > /dev/null && cd -")
        except Exception as e:
            print(e)
            sys.exit(-1)

    else:
        if not os.path.exists(artifacts_dir):
            print(f"[ Error ] {artifacts_dir} does not exist.")
            sys.exit(-1)

    # Generate list of models to test based on the directories in artifacts folder
    model_list = []
    for i in os.listdir(artifacts_dir):
        i = i.strip()
        if i.endswith('.tar.gz'):
            model_list.append(i.split('_')[0])

    # if run.log exists in the directory, move it to run_pc.log
    # Walk through the directory tree
    for root, dirs, files in os.walk(artifacts_dir):
        if 'run.log' in files:
            # Construct full file path
            run_log_path = os.path.join(root, 'run.log')
            # Construct new file path
            run_pc_log_path = os.path.join(root, 'run_pc.log')
            # Move the file
            shutil.move(run_log_path, run_pc_log_path)
            print(f"Moved {run_log_path} to {run_pc_log_path}")
            # Make a new run.log file 
            with open(run_log_path, "w") as file:
                file.write("The EVM Logs will be here. \n")
            continue
        else:
            print(f"run.log not found in {root} subdirectory")

    print(f"run.log not found in {root} subdirectory")

    model_list = list(set(model_list))
    if len(model_list) == 0:
        print(f"[ Error ] No model found for testing.")
        sys.exit(-1)

    # we need to send the relative path instead of absolute because it used in evm, without the 8bits in end
    artifacts_dir = './' + os.path.relpath(artifacts_dir, edgeai_benchmark_path)
    if artifacts_dir.endswith(f'/{args.tensor_bits}bits'):
        artifacts_dir = artifacts_dir[:-len(f'/{args.tensor_bits}bits')]
    # Run the tests
    benchmark_evm = BenchmarkEvm(evm_config=evm_config,
                                edgeai_benchmark_path=edgeai_benchmark_path,
                                ip_address=ip_address,
                                reboot_type=args.reboot_type,
                                logs_dir=args.logs_dir,
                                dataset_dir_path=args.dataset_dir,
                                modelartifacts_path=artifacts_dir,
                                tensor_bits=args.tensor_bits,
                                session_type_dict=args.session_type_dict)

    status = benchmark_evm.init_setup()
    if status:
        print (f"[ Info ] Model under tests are {model_list}")
        benchmark_evm.run_tests(model_list, num_frames=args.num_frames, timeout=args.evm_timeout)
    else:
        sys.exit(-1)

################################################################################


######################### TIDL_UNIT TESTS ######################################
elif (args.test_suite == "TIDL_UNIT_TEST"):
    # Clear the existing logs dir
    CHUNK_SIZE=50
    TEST_MAP = {}
    if args.artifacts_folder is None:
        artifacts_dir = os.path.join(edgeai_benchmark_path, "tests", "tidl_unit", "work_dirs", "modelartifacts", f"{args.tensor_bits}bits")
    else:
        artifacts_dir = args.artifacts_folder

    if not os.path.exists(artifacts_dir):
        print(f"[ Error ] {artifacts_dir} does not exist")
        sys.exit(-1)
    
    operators = []
    if args.operators is not None:
        operators = args.operators.split(" ")
    
    for dir in os.listdir(artifacts_dir):
        operator_name = dir.strip().split("_")
        operator_name = operator_name[0]
        # Skip which are not present in specified operators argument
        if(operators != [] and operator_name not in operators):
            continue

        io_file = os.path.abspath(os.path.join(artifacts_dir, dir, "artifacts/subgraph_0_tidl_io_1.bin"))
        bin_file = os.path.abspath(os.path.join(artifacts_dir, dir, "artifacts/subgraph_0_tidl_net.bin"))
        artifacts_present = os.path.exists(io_file) and os.path.exists(bin_file)
        if artifacts_present == True:
            test_name = dir.strip().split("_")[0]
            if test_name not in TEST_MAP:
                TEST_MAP[test_name] = []
            TEST_MAP[test_name].append(dir)

    for key in TEST_MAP:
        try:
            TEST_MAP[key] = sorted(TEST_MAP[key], key=lambda x: int(x.split('_')[-1]))
        except:
            pass

    TEST_LIST = []
    for key in TEST_MAP:
        chunks = [TEST_MAP[key][x:x+CHUNK_SIZE] for x in range(0, len(TEST_MAP[key]), CHUNK_SIZE)]
        TEST_MAP[key] = chunks
        for idx,i in enumerate(chunks):
            test_name = f'{key}_Chunk_{idx}'
            test_command = ""
            for j in i:
                test_command += f'test_tidl_unit.py::test_tidl_unit_operator[{j}] '
            test_command = test_command.strip()
            TEST_LIST.append((test_name,test_command))

    tidl_unit_test_evm = TIDLUnitTestEVM(evm_config=evm_config,
                                        edgeai_benchmark_path=edgeai_benchmark_path,
                                        ip_address=ip_address,
                                        reboot_type=args.reboot_type,
                                        logs_dir=args.logs_dir,
                                        dataset_dir_path=args.dataset_dir,
                                        model_artifacts_path=args.artifacts_folder,
                                        evm_local_ip=args.evm_local_ip)
    
    status = tidl_unit_test_evm.init_setup()


    if status:
        tidl_unit_test_evm.run_tests(test_list=TEST_LIST, timeout=args.evm_timeout)
    else:
        sys.exit(-1)
