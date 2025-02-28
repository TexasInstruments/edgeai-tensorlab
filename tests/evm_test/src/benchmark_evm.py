#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
BenchmarkEvm is responsible for running tests on the evm via UartInterface
"""

import os
import sys
import time
import traceback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from src.relay_control import AnelRelayControl
from src.uart_interface import UartInterface

class BenchmarkEvm():
    def __init__(self, evm_config, edgeai_benchmark_path, ip_address, reboot_type="soft", logs_dir=None, dataset_dir_path=None, modelartifacts_path=None):
        self.evm_config = evm_config
        self.soc = self.evm_config["soc"]
        self.eai_benchmark_mount_path = f"{ip_address}:{edgeai_benchmark_path}"
        self.dataset_dir_mount_path = f"{ip_address}:{dataset_dir_path}" if dataset_dir_path is not None else None
        self.test_num = 0
        self.pass_num = 0
        self.restarts = 0
        self.relay = None
        self.reboot_type = reboot_type
        if self.reboot_type == "hard":
            relay_exe = self.evm_config["relay_info"]["executable_path"]
            relay_ip = self.evm_config["relay_info"]["ip_address"]
            relay_number = self.evm_config["relay_info"]["power_port"]
            self.relay = AnelRelayControl(relay_exe,relay_ip,relay_number)
            self.relay.verify_relay()
        elif self.reboot_type == "soft":
            pass
        else:
            print("[ Error ] Invalid reboot type defined. Allowed values are (soft|hard)")
            sys.exit(-1)
    
        self.logs_dir = logs_dir
        if self.logs_dir == None:
            self.logs_dir = "evm_test_logs"

        self.modelartifacts_path = modelartifacts_path

        print(f"[ Info ] SOC : {self.soc}")
        print(f"[ Info ] EVM Reboot type : {self.reboot_type}")
        print(f"[ Info ] Logs Dir : {self.logs_dir}")

    def init_setup(self):
        print(f"[ Info ] Restarting the board and doing initial setup")

        status = True

        log_file_path = f'{self.logs_dir}/{self.soc}/dut_firmware_update_uart.log'
        uart_interface = UartInterface(self.evm_config["dut_uart_info"],
                                       self.evm_config["dut_uart_info"],
                                       log_file_path)
    
        # Reboot
        if(self.reboot_type == "hard"):
            self.relay.switch_relay(operation="toggle")
        else:
            status = uart_interface.send_uart_command('reboot')
        
        print(f"[ Info ] Sleeping for 30 seconds after reboot...")
        time.sleep(30)

        # wait for root prompt
        cnt=0
        while True:
            status = uart_interface.send_uart_command('', 'login:', 10, True)
            cnt+=1
            if status:
                break
            if cnt>10 and not status:
                sys.exit(-1)

        # login as root
        if status:
            status = uart_interface.send_uart_command('root', '#', 5, True)

        # clear buffer
        if status:
            time.sleep(5)
            status = uart_interface.send_uart_command('', '#', 5, True)

        ## mount edgeai-benchmark and dataset
        if status:
            if self.dataset_dir_mount_path:
                command = f"cd && ./setup_eai_benchmark.sh {self.eai_benchmark_mount_path} {self.dataset_dir_mount_path}"
            else:
                command = f"cd && ./setup_eai_benchmark.sh {self.eai_benchmark_mount_path}"

            status = uart_interface.send_uart_command(command, "SCRIPT_EXECUTED_SUCCESSFULLY", 60, True)
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
            raise Exception("Error in initial setup.")
        
        return status

    def parse_test_run(self,response):
        '''
        status = 0 [No run], 1 [Successful run], -1 [Critical error]
        '''
        ignore_filters = ["VX_ZONE_ERROR:Enabled","Globally Enabled","Globally Disabled","VX_ZONE_ERROR:[tivxObjectDeInit"]
        critical_errors = ["VX_ZONE_ERROR","dumped core","core dump","Segmentation fault"]
        status = 0
        success_status = 0
        error_status = 0
        critical_error_status = 0
        response = response.split('\n')
        for i in response:
            i = i.strip()
            if "SUCCESS:" in i and "benchmark results" in i:
                success_status = 1
            if "ERROR:" in i:
                error_status = 1

            ignore = False
            for j in ignore_filters:
                if j in i:
                    ignore = True
                    break

            if not ignore:
                for j in critical_errors:
                    if j in i:
                       critical_error_status = 1 

        if error_status == 1:
            status = 0
        if success_status == 1:
            status = 1
        if critical_error_status == 1:
            status = -1
        
        return status

    def run_single_test(self, timeout=300, generate_report='0', model_selection='null', num_frames='0', total_test=1):
        test_itr = self.test_num + 1
        print(f"\n\n[ Info ] Running {test_itr}/{total_test} [{model_selection}]")
        log_file_path = f'{self.logs_dir}/{self.soc}/{test_itr:>04}_{model_selection}.log'

        uart_interface = UartInterface(self.evm_config["dut_uart_info"],
                                       self.evm_config["dut_uart_info"],
                                       log_file_path)

        # clear buffer
        status = uart_interface.send_uart_command('', '#', 2, True)

        if not status:
            status = self.init_setup()

        if status:
            status = uart_interface.send_uart_command('clear', '#', 2, True)
        
        infer_status = False
        if status:
            if self.modelartifacts_path is not None:
                command = f'cd && ./model_infer.sh {self.soc} {timeout} {generate_report} {model_selection} {num_frames} {self.modelartifacts_path}'
            else:
                command = f'cd && ./model_infer.sh {self.soc} {timeout} {generate_report} {model_selection} {num_frames}'

            infer_status = uart_interface.send_uart_command(command, "END_OF_MODEL_INFERENCE", timeout, True, 1)
            response = uart_interface.log_buffer
            print(f"\n\n*******************************\nLog Buffer : {response}\n*******************************\n\n")
            response_status = self.parse_test_run(response)
        
        if not status:
                response = uart_interface.log_buffer
                print(f"[ Error ] Error in Target Inference over UART.\n", response)
                raise Exception("Error in Target Inference over UART.")

        if infer_status:
            print(f"[ Info ] Target Inference is completed successfully.")
        else:
            raise TimeoutError(f"{timeout}s timeout while Target Inference over UART.")

        return response_status


    def run_tests(self, model_list, num_frames='0', timeout=300):
        total_models = len(model_list)
        while self.test_num < total_models:
            print()
            print('*' * 80)
            generate_report = '0'
            if (self.test_num == total_models - 1):
                generate_report = '1'
            try:
                model = model_list[self.test_num]
                status = self.run_single_test(timeout=timeout, model_selection=model, num_frames=num_frames, total_test=total_models, generate_report=generate_report)
                if status == 0:
                    print(f"[ Info ] Test {self.test_num + 1}/{total_models} [{model}] did not run successfully!")
                elif status == -1:
                    print(f"[ Error ] Critical error detected while running test {self.test_num + 1}/{total_models} [{model}]! Rebooting...")
                    self.restarts += 1
                    self.init_setup()
                else:
                    print(f"[ Info ] Test {self.test_num + 1}/{total_models} [{model}] ran successfully!")
                    self.pass_num += 1
            except TimeoutError:
                print("[ Error ] Timeout while run_single_test().")
                print(f"\n{traceback.format_exc()}\n")
                self.restarts += 1
                self.init_setup() 
            except Exception as e:
                print("[ Error ] Error while run_single_test().")
                print(f"\n{traceback.format_exc()}\n")
            self.test_num += 1
            print('*' * 80)
            print()

        print(f"Total test cases                         : {total_models}")
        print(f"Pass                                     : {self.pass_num}/{total_models}")
        print(f"Fail                                     : {total_models - self.pass_num}/{total_models}")
        print(f"Restart due to timeout or critical error : {self.restarts}")
