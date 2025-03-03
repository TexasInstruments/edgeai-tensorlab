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
from src.uart_interface import UartInterface
from src.benchmark_evm import BenchmarkEvm

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('--soc', help='SOC (AM68A|AM69A|TDA4VM|AM62A|AM67A)', type=str, default="AM68A")
parser.add_argument('--uart', help='UART port', type=str, default="/dev/ttyUSB2")
parser.add_argument('--logs_dir', help='Optional path to store evm test logs', type=str, default='./evm_test_logs')
parser.add_argument('--artifacts_folder', help='Optional path to store the model artifacts after EVM run', type=str, default=None)
parser.add_argument('--artifacts_tarball', help='Optional path to model artifacts tarball', type=str, default=None)
parser.add_argument('--reboot_type', help='Reboot type (hard|soft)', type=str, default="soft")
parser.add_argument('--relay_exe_path', help='Anel power switch exe path', type=str, default=None)
parser.add_argument('--relay_ip_address', help='Anel power switch IP address', type=str, default=None)
parser.add_argument('--relay_power_port', help='Anel power switch port number', type=str, default=None)
parser.add_argument('--pc_ip', help='IP address of the pc', type=str, default=None)
parser.add_argument('--num_frames', help='The number of frames to run the evaluation', type=str, default=None)
parser.add_argument('--dataset_dir', help='Optional path to get the dataset from pc, used with pc_ip', type=str, default=None)


args = parser.parse_args()

status = True
edgeai_benchmark_path = os.path.join(CURRENT_DIR,"../../")

# Get ip address of the PC
ip_address = args.pc_ip
if ip_address == None:
    #try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.0.0.0', 0))
        ip_address = s.getsockname()[0]
    # except:
    #     print("[ Error ] Could not get IP address of the PC. Please provide as argument using --pc_ip")
    #     sys.exit(-1)

# Verify args
args.soc = args.soc.strip().upper()
if args.soc not in ("AM68A","AM69A","TDA4VM","AM62A","AM67A"):
    print(f"[ Error ] Invalid SOC: {args.soc}")
    sys.exit(-1)

if not os.path.exists(args.uart):
    print(f"[ Error ] {args.uart} does not exist")
    sys.exit(-1)

if args.reboot_type == "hard":
    if args.relay_exe_path == None:
        print(f"[ Error ] Please provide relay executable path")
        sys.exit(-1)
    elif args.relay_ip_address == None:
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
                    "executable_path": args.relay_exe_path,
                    "ip_address": args.relay_ip_address,
                    "power_port": args.relay_power_port
                }
            }

# Model artifacts folder
if args.artifacts_folder is None:
    artifacts_dir = os.path.join(edgeai_benchmark_path, "work_dirs", "modelartifacts", evm_config["soc"], "8bits")
else:
    artifacts_dir = args.artifacts_folder

# Print info
print()
print("*" * 80)
print(f"SOC : {evm_config['soc']}") 
print(f"UART PORT : {evm_config['dut_uart_info']['port']}")
print(f"PC IP ADDRESS : {ip_address}")
print(f"MODEL ARTIFACTS DIR : {artifacts_dir}") 
print(f"ARTIFACTS TARBALL : {args.artifacts_tarball}")
print(f"LOGS DIR : {args.logs_dir}") 
print(f"REBOOT TYPE : {args.reboot_type}")
if args.reboot_type == "hard":
    print(f"RELAY EXE PATH : {args.relay_exe_path}")
    print(f"RELAY IP : {args.relay_ip_address}")
    print(f"RELAY PORT : {args.relay_power_port}")
print("*" * 80)
print()

# Cleanup and create logs directory
try:
    shutil.rmtree(f'{args.logs_dir}/{evm_config["soc"]}')
except:
    pass
try:
    print(f'[ Info ] Creating {args.logs_dir}/{evm_config["soc"]} directory.')
    os.makedirs(f'{args.logs_dir}/{evm_config["soc"]}',exist_ok=True)
except:
    print(f'[ Error ] Cannot create {args.logs_dir}/{evm_config["soc"]} directory.')
    sys.exit(-1)


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
        break
else:
    print("run.log not found in any subdirectory")

model_list = list(set(model_list))
if len(model_list) == 0:
    print(f"[ Error ] No model found for testing.")
    sys.exit(-1)

# we need to send the relative path instead of absolute because it used in evm, without the 8bits in end
artifacts_dir = './' + os.path.relpath(artifacts_dir, edgeai_benchmark_path)
if artifacts_dir.endswith('/8bits'):
    artifacts_dir = artifacts_dir[:-len('/8bits')]
# Run the tests
benchmark_evm = BenchmarkEvm(evm_config=evm_config,
                             edgeai_benchmark_path=edgeai_benchmark_path,
                             ip_address=ip_address,
                             reboot_type=args.reboot_type,
                             logs_dir=args.logs_dir,
                             dataset_dir_path=args.dataset_dir,
                             modelartifacts_path=artifacts_dir)

status = benchmark_evm.init_setup()
if status:
    print (f"[ Info ] Model under tests are {model_list}")
    benchmark_evm.run_tests(model_list, num_frames=args.num_frames)
else:
    sys.exit(-1)