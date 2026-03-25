#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
TIDLBaseEVMTest is responsible for running tests on the evm via UartInterface
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from src.relay_control import PowerRelayControl
from src.uart_interface import UartInterface

class TIDLBaseEVMTest():
    def __init__(self, evm_config, edgeai_benchmark_path, dataset_dir_path, model_artifacts_path, reboot_type, logs_dir, ip_address, evm_local_ip=""):
        self.evm_config = evm_config
        self.soc = self.evm_config["soc"]
        self.eai_benchmark_mount_path = f"{ip_address}:{edgeai_benchmark_path}"
        if ':' in dataset_dir_path:
            self.dataset_dir_mount_path = dataset_dir_path
        else:
            self.dataset_dir_mount_path = f"{ip_address}:{dataset_dir_path}"
        
        if model_artifacts_path is not None:
            if ':' in model_artifacts_path:
                self.model_artifacts_mount_path = model_artifacts_path
            else:
                self.model_artifacts_mount_path = f"{ip_address}:{model_artifacts_path}"
        else:
            self.model_artifacts_mount_path = ""
        
        self.relay = None
        self.reboot_type = reboot_type
        if self.reboot_type == "hard":
            relay_type = self.evm_config["relay_info"]["relay_type"]
            relay_exe = self.evm_config["relay_info"]["executable_path"]
            relay_ip = self.evm_config["relay_info"]["ip_address"]
            relay_number = self.evm_config["relay_info"]["power_port"]
            relay_trigger_mechanism = self.evm_config["relay_info"]["relay_trigger_mechanism"]
            self.relay = PowerRelayControl(relay_exe,relay_ip,relay_number,relay_type,relay_trigger_mechanism)
            self.relay.verify_relay()

        elif self.reboot_type == "soft":
            pass
        else:
            print("[ Error ] Invalid reboot type defined. Allowed values are (soft|hard)")
            sys.exit(-1)
    
        self.logs_dir = logs_dir
        
        self.evm_local_ip = evm_local_ip

        print(f"[ Info ] SOC : {self.soc}")
        print(f"[ Info ] Edgeai Benchmark Path : {self.eai_benchmark_mount_path}")
        print(f"[ Info ] Dataset Path : {self.dataset_dir_mount_path}")
        print(f"[ Info ] Model Artifacts Path : {self.model_artifacts_mount_path}")
        print(f"[ Info ] Logs Dir : {self.logs_dir}")
        print(f"[ Info ] EVM Reboot type : {self.reboot_type}")
        print(f"[ Info ] EVM Local IP : {self.evm_local_ip}")

    def init_setup(self, init_cmd):
        print(f"[ Info ] Restarting the board and doing initial setup")

        status = True

        log_file_path = f'{self.logs_dir}/dut_firmware_update_uart.log'
        uart_interface = UartInterface(self.evm_config["dut_uart_info"],
                                       self.evm_config["dut_uart_info"],
                                       log_file_path)
        
        #Reboot and wait for login prompt
        retries=0
        booted=False
        while retries < 3:
            if(self.reboot_type == "hard"):
                self.relay.switch_relay(operation="toggle")
                print(f"\n[ Info ] Sleeping for 30 seconds after reboot...")
                time.sleep(30)
            else:
                status = uart_interface.send_uart_command('root', '#', 5, True)
                status = uart_interface.send_uart_command('reboot', press_enter=True, retry_count=1)
                del uart_interface
                print(f"\n[ Info ] Sleeping for 30 seconds after reboot...")
                time.sleep(30)
                uart_interface = UartInterface(self.evm_config["dut_uart_info"],
                                            self.evm_config["dut_uart_info"],
                                            log_file_path)
            # Clear buffer
            uart_interface.send_uart_command('', None, 1, True)
            uart_interface.send_uart_command('', None, 1, True)
            uart_interface.send_uart_command('', None, 1, True)
            status = uart_interface.send_uart_command('', 'login:', 5, True)
            if status:
                booted = True
                break
            retries += 1

        if (not booted):
            print(f"[ Error ] Cannot find login prompt.\n")
            return False

        # login as root
        if status:
            uart_interface.send_uart_command('', None, 1, True)
            uart_interface.send_uart_command('', None, 1, True)
            uart_interface.send_uart_command('', None, 1, True)
            status = uart_interface.send_uart_command('root', '#', 5, True)

        time.sleep(2)

        # clear buffer
        if status:
            uart_interface.send_uart_command('', None, 1, True)
            uart_interface.send_uart_command('', None, 1, True)
            uart_interface.send_uart_command('', None, 1, True)
            status = uart_interface.send_uart_command('', '#', 5, True)

        ## mount edgeai-benchmark and run initial setup
        if status:
            command = f"{init_cmd}"
            status = uart_interface.send_uart_command(command, "SCRIPT_EXECUTED_SUCCESSFULLY", 80, True)
            response = uart_interface.log_buffer
            print(f"\n\n*******************************\nLog Buffer : {response}\n*******************************\n\n")

        if status:
            command = f'cd ~/edgeai-benchmark/ && ./run_set_env.sh {self.soc} evm  && source ./run_set_env.sh {self.soc} evm && echo END_OF_UART_COMMAND'
            status = uart_interface.send_uart_command(command, 'END_OF_UART_COMMAND', 5, True)

        if status:
            print(f"[ Info ] Inital setup successful.")
        else:
            response = uart_interface.log_buffer
            print(f"[ Error ] Error in initial setup.\n", response)
        
        return status
