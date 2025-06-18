#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
TIDLUnitTestEVM is responsible for running pytest based tidl_unit_tests on the evm via UartInterface
"""

import os
import sys
import time
import traceback
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from src.relay_control import PowerRelayControl
from src.uart_interface import UartInterface
from src.base_evm import TIDLBaseEVMTest

class TIDLUnitTestEVM(TIDLBaseEVMTest):
    def __init__(self, evm_config, edgeai_benchmark_path, ip_address, dataset_dir_path, model_artifacts_path=None, reboot_type="soft", evm_local_ip="", logs_dir=None):
        if logs_dir == None:
            logs_dir = f"evm_test_logs/TIDL_UNIT_TEST/{evm_config['soc']}"

        super().__init__(evm_config=evm_config,
                       edgeai_benchmark_path=edgeai_benchmark_path,
                       dataset_dir_path=dataset_dir_path, 
                       model_artifacts_path=model_artifacts_path, 
                       reboot_type=reboot_type,
                       logs_dir=logs_dir, 
                       ip_address=ip_address, 
                       evm_local_ip=evm_local_ip)

        self.test_num = 0
        self.pass_num = 0
        self.restarts = 0

    def init_setup(self):
        init_cmd = f"cd && EAI_BENCHMARK_MOUNT_PATH={self.eai_benchmark_mount_path} TEST_SUITE=TIDL_UNIT_TEST LOCAL_IP={self.evm_local_ip} TIDL_UNIT_TEST_DATASET_MOUNT_PATH={self.dataset_dir_mount_path} TIDL_UNIT_TEST_MODEL_ARTIFACTS_MOUNT_PATH={self.model_artifacts_mount_path} ./setup_eai_benchmark.sh"

        status = super().init_setup(init_cmd)
        return status

    def parse_test_run(self, response):
        '''
        status = 0 [No run], 1 [Successful run], -1 [Critical error], -2 [Timed out]
        '''
        status = 0
        success_status = 0
        error_status = 0
        critical_error_status = 0
        critical_error_test_name=None

        response = response.split('\n')

        for i in response:
            i = i.strip()
            if "PASSED" in i:
                success_status = 1
            if "FAILED" in i:
                error_status = 1
            if "CRITICAL_ERROR" in i:
                critical_error_status = 1
                match = re.search(r'test_tidl_unit(?:\.py)?[^\]]*\]', i)
                if match:
                    critical_error_test_name = match.group()

        if success_status == 1:
            status = 1
        if error_status == 1:
            status = 0
        if critical_error_status == 1:
            status = -1
        
        return (status, critical_error_test_name)

    def run_single_test(self, test_name, test_cmd, total_test=1, timeout=300):
        test_itr = self.test_num + 1
        print(f"\n\n[ Info ] Running {test_itr}/{total_test} [{test_name}]")
        log_file_path = f'{self.logs_dir}/{test_itr:>04}_{test_name}.log'

        chunk_size = len(test_cmd.strip().split(' '))
        command_time_out = int(max(1,(timeout/chunk_size)))
        if(command_time_out >= (timeout - 5)):
            command_time_out = timeout - 5

        test_not_done = True
        first_run = True
        count = 0
        while test_not_done:
            name = f"{test_name}_Run_{count}"
            count += 1
            test_not_done = False

            if not first_run:
                self.init_setup()
                self.restarts += 1

            uart_interface = UartInterface(self.evm_config["dut_uart_info"],
                                           self.evm_config["dut_uart_info"],
                                           log_file_path)

            # clear buffer
            status = uart_interface.send_uart_command('', '#', 5, True)
            status = uart_interface.send_uart_command('', '#', 5, True)
            status = uart_interface.send_uart_command('', '#', 5, True)

            if not status:
                status = self.init_setup()
                uart_interface = UartInterface(self.evm_config["dut_uart_info"],
                                               self.evm_config["dut_uart_info"],
                                               log_file_path)

            if status:
                status = uart_interface.send_uart_command('clear', '#', 5, True)

            infer_status = False
            response_status = 0
            if status:
                command = f'cd ~ && TEST_NAME="{name}" TEST_COMMAND="{test_cmd}" TIMEOUT={command_time_out} ./model_infer_tidl_unit_test.sh'
                infer_status = uart_interface.send_uart_command(command, "END_OF_MODEL_INFERENCE", timeout, True, 1)
                response = uart_interface.log_buffer
                print(f"\n\n*******************************\nLog Buffer : {response}\n*******************************\n\n")
                (response_status, critical_error_test_name) = self.parse_test_run(response)
                if response_status == -1 and critical_error_test_name is not None:
                    remaining_test = test_cmd.split(critical_error_test_name)[-1].strip()
                    if remaining_test != "":
                        test_cmd = remaining_test
                        test_not_done = True
                        first_run = False
                        del uart_interface
                        print(f"[INFO] Critical error at {critical_error_test_name}. Re-running remaining tests in the chunk")

                if infer_status:
                    print(f"[ Info ] Target Inference is completed successfully.")
                else:
                    raise TimeoutError(f"{timeout}s timeout while Target Inference over UART.")

        return response_status


    def run_tests(self, test_list, timeout=300):
        total_tests = len(test_list)
        while self.test_num < total_tests:
            print()
            print('*' * 80)
            try:
                test_name = test_list[self.test_num][0]
                test_cmd = test_list[self.test_num][1]
                status = self.run_single_test(test_name=test_name, test_cmd=test_cmd, total_test=total_tests, timeout=timeout)
                if status == 0:
                    print(f"[ Info ] Test {self.test_num + 1}/{total_tests} [{test_name}] did not run successfully!")
                elif status == -1:
                    print(f"[ Error ] Critical error detected while running test {self.test_num + 1}/{total_tests} [{test_name}]! Rebooting...")
                    self.restarts += 1
                    self.init_setup()
                else:
                    print(f"[ Info ] Test {self.test_num + 1}/{total_tests} [{test_name}] ran successfully!")
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

        print(f"Total test cases                         : {total_tests}")
        print(f"Pass                                     : {self.pass_num}/{total_tests}")
        print(f"Fail                                     : {total_tests - self.pass_num}/{total_tests}")
        print(f"Restart due to timeout or critical error : {self.restarts}")