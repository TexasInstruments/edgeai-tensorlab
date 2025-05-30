#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
UartInterface is responsible for the handling Serial Communication between devices
over UART. It is creating the Serial object and File Descriptor for logging as well
as it is closing the connection with proper exception handling.
"""


from datetime import datetime

import pexpect
import serial
from pexpect import fdpexpect


class UartInterface:
    def __init__(self, uart_conn_port_info={}, uart_log_port_info=None, log_file_path=None):
        # check the fields of the uart_conn_port_info dict
        if not {'port', 'baudrate', 'bytesize', 'parity', 'stopbits'} <= set(uart_conn_port_info.keys()):
            raise Exception('Constructor is missing required information, ' +
                             'make sure instantiation call complies with: ' +
                             "{'port':<serial/tty port>,'baudrate':<bps rate>, " +
                             "'bytesize': <data bits>, 'parity':<N,O, or E>," +
                             "'stopbits':<1,1.5,2> " +
                             "[, 'timeout':<None or # of sec>] " +
                             "[, 'xonxoff':<0 or 1>] [, 'rtscts':<0 or 1>]}")

        self.uart_conn_port_info = uart_conn_port_info
        self.uart_log_port_info = uart_log_port_info
        self.log_file_path = log_file_path

        # object resources variables
        self.serial_conn_obj = None
        self.serial_log_obj = None
        self.file_descriptor_process = None
        self.log_file_obj = None
        self.log_buffer = None

        if uart_log_port_info is None:
            self.uart_log_port_info = self.uart_conn_port_info

        if log_file_path is None:
            print('[ Info ] Log path not defined, using default.log')
            self.log_file_path = 'default.log'

        self._connect()

    def __del__(self):
        self._disconnect()
        print('[ Info ] Deleting UartInterface object and releasing the resources.')

    ##############################################################################
    ########################[ Public functions/Methods ]##########################

    def send_uart_command(self, cmd, expected_string=None, timeout=120, press_enter=False, retry_count=3) -> bool:
        # command success status
        status = False

        try:
            # create serial and file descriptor object
            # self._connect()

            index = None
            # retry 3 times
            for iteration in range(retry_count):
                # run the command
                print('[ Info ] Sending Uart Command : ' + str(cmd))

                if press_enter:
                    self.file_descriptor_process.sendline(cmd + '\n')
                else:
                    self.file_descriptor_process.sendline(cmd)

                # check the output if expected_string is defined
                # skip otherwise
                if expected_string is not None:
                    # check the constraint
                    start_time = datetime.now()
                    index = self.except_string_in_uart_log(expected_string, timeout)
                    end_time = datetime.now()

                    # if command execute successfully then break the loop
                    if index == 0:
                        break
                    elif index == 1:  # Timeout and we did not received expected string
                        print(f"[ Warning ] Did not find expected_string in {timeout}s timeout. \
                        Trying Again... iteration - {iteration + 1}")

            if index == 1:
                raise Exception(f"[ Error ] Timeout while sending command : {cmd}")
            else:
                status = True
                print('[ Info ] Successfully Executed Uart Command.\n')
        except Exception as e:
            print('[ Error ] Error occurred while sending command : ', e)
        finally:
            pass
            # self._disconnect()

        return status

    def except_string(self, expected_string, timeout):
        # expect success status
        status = False

        try:
            # create serial and file descriptor object
            self._connect()
            # check the constraint
            start_time = datetime.now()
            index = self.except_string_in_uart_log(expected_string, timeout)
            end_time = datetime.now()

            # create a dict which contains send command status data
            # send data to plugins
            PluginInterface().run_method('on_constraint_check', {
                "result": {
                    'constraint_msg': expected_string,
                    'timeout': timeout,
                    'status': index == 0,
                },
                "start_time": start_time,
                "end_time": end_time
            })

            # Timeout and we did not received expected string
            if index == 1:
                print(f"[ Warning ] Did not fond expected_string in {timeout}s timeout.")
            else:
                status = True
                print('[ Info ] Expected string matched on uart logs.')
        except Exception as e:
            print('[ Error ] Error occurred while checking uart constraint: ', e)
        finally:
            self._disconnect()

        return status

    def except_string_in_uart_log(self, expected_string, timeout):
        print('[ Info ] Waiting for : ' + expected_string)
        print(' [Info ] Wait timeout : ' + str(timeout))
        index = self.file_descriptor_process.expect([expected_string, pexpect.TIMEOUT, pexpect.EOF], timeout=timeout)

        # checking the return code based on the list provided in expect()
        if index == 0:  # Process is completes and we received expected string
            # Print the DUT UART logs
            cmd_response = self.file_descriptor_process.before + self.file_descriptor_process.after
            cmd_response = cmd_response.decode(encoding='iso8859-1')
            # print(cmd_response)
            self.log_buffer = cmd_response

        return index

    ##############################################################################
    ########################[ Private functions/Methods ]#########################

    def _connect(self):
        self.serial_conn_obj = self._create_serial_conn(self.uart_conn_port_info)

        # if booth logging and connection serial porty same then no need to create
        # separate serial object
        if self.uart_conn_port_info == self.uart_log_port_info:
            self.serial_log_obj = self.serial_conn_obj
        else:
            self.serial_log_obj = self._create_serial_conn(self.uart_log_port_info)

        # create logging for serial_log_obj
        self.file_descriptor_process = self._create_file_descriptor_process(self.serial_log_obj)

    def _create_serial_conn(self, uart_port_info):
        # create serial connection obj and apply settings
        serial_obj = serial.Serial(uart_port_info['port'])
        ser_settings = serial_obj.getSettingsDict()
        ser_settings.update(uart_port_info)
        serial_obj.applySettingsDict(ser_settings)
        print(f"\n[ Info ] Serial connection is opened for port {uart_port_info['port']}")
        return serial_obj

    def _create_file_descriptor_process(self, serial_obj):
        # setup the logger
        if self.log_file_path is not None:
            self.log_file_obj = open(self.log_file_path, 'ab+')
            print('[ Info ] Uart log file opened.')

        # create process child for run command on serial port
        file_descriptor_process = fdpexpect.fdspawn(serial_obj, logfile=self.log_file_obj)
        print("[ Info ] File descriptor process is opened.\n")
        return file_descriptor_process

    def _disconnect(self):
        try:
            if self.serial_conn_obj and self.serial_conn_obj.isOpen():
                self.serial_conn_obj.close()
                print('[ Info ] Serial connection obj is closed.')
            if self.serial_log_obj and self.serial_log_obj.isOpen():
                self.serial_log_obj.close()
                print('[ Info ] Serial connection obj is closed.')
            if self.file_descriptor_process and self.file_descriptor_process.isalive():
                self.file_descriptor_process.close()
                print('[ Info ] File descriptor process is closed.')
            if self.log_file_obj:
                self.log_file_obj.close()
                print('[ Info ] Uart log file closed.')
        except Exception as e:
            print("[ Error ] Error while closing the UartInterface resources :", e)
