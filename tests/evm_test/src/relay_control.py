#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
AnelRelayControl is responsible for switching ANEL Elektronik AG power switch.
This requires an executable provided by anel, ip address of the switch and the port number
"""

import os
import sys
import time

class PowerRelayControl():
    def __init__(self, executable, ip_address, relay_number, relay_type, relay_trigger_mechanism):
        self.relay_exe = executable
        self.relay_ip = ip_address
        self.relay_number = int(relay_number)
        self.relay_type = relay_type
        self.relay_trigger_mechanism = relay_trigger_mechanism
        self.verify_relay()
        if(self.relay_type=="DLI"):
            from dlipower import PowerSwitch
            os.environ['no_proxy'] = '*'
            self.dli_switch = PowerSwitch(hostname=self.relay_ip, userid="admin", password="1234")

    def verify_relay(self):
        print(f"[ Info ] Verifying relay configurations")
        if self.relay_number < 1 and self.relay_number > 8:
            print(f"[ Error ] Relay number invalid")
            sys.exit(-1)
        if self.relay_type != "DLI" and self.relay_type != "ANEL":
            print(f"[ Error ] Relay type invalid. Allowed values are (ANEL, DLI)")
            sys.exit(-1)
        if self.relay_type == "ANEL":
            if self.relay_trigger_mechanism != "LIB" and self.relay_trigger_mechanism != "EXE":
                print(f"[ Error ] Relay trigger mechanism invalid. Allowed values are (LIB, EXE)")
                sys.exit(-1)
            if self.relay_trigger_mechanism == "EXE" and not os.path.exists(self.relay_exe):
                print(f"[ Error ] {self.relay_exe} does not exist. Please provide a valid executable")
                sys.exit(-1)
            if self.relay_trigger_mechanism == "EXE" and not os.path.exists(self.relay_exe):
                print(f"[ Error ] {self.relay_exe} does not exist. Please provide a valid executable")
                sys.exit(-1)

    def switch_relay(self,operation="toggle"):
        print(f"[ Info ] Switching relay [operation={operation}]")

        if operation == "on":
            if self.relay_type=="ANEL":
                self.switch_anel_on()
            elif self.relay_type=="DLI":
                self.switch_dli_on()
        elif operation == "off":
            if self.relay_type=="ANEL":
                self.switch_anel_off()
            elif self.relay_type=="DLI":
                self.switch_dli_off()
        elif operation == "toggle":
            if self.relay_type=="ANEL":
                self.switch_anel_off()
                time.sleep(5)
                self.switch_anel_on()
            elif self.relay_type=="DLI":
                self.switch_dli_off()
                time.sleep(5)
                self.switch_anel_on()
        else:
            print(f"[ Error ] Invalid operation value for relay. Allowed values are (off,on,toggle)")

    def switch_anel_on(self):
        if (self.relay_trigger_mechanism == "LIB"):
            os.system(f"pypwrctrl -d -i 12347 -o 12345 on {self.relay_ip} {self.relay_number}")
        else:
            os.system(f"{self.relay_exe} {self.relay_ip},12345,12347,rel,{self.relay_number},on,admin,anel")
    
    def switch_anel_off(self):
        if (self.relay_trigger_mechanism == "LIB"):
            os.system(f"pypwrctrl -d -i 12347 -o 12345 off {self.relay_ip} {self.relay_number}")
        else:
            os.system(f"{self.relay_exe} {self.relay_ip},12345,12347,rel,{self.relay_number},off,admin,anel")

    def switch_dli_on(self):
        self.dli_switch.on(self.relay_number)
    
    def switch_dli_off(self):
        self.dli_switch.off(self.relay_number)