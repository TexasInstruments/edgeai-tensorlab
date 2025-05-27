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
    def __init__(self, executable, ip_address, relay_number, relay_type):
        self.relay_exe = executable
        self.relay_ip = ip_address
        self.relay_number = int(relay_number)
        self.relay_type = relay_type
        self.verify_relay()
        if(relay_type=="DLI"):
            from dlipower import PowerSwitch
            os.environ['no_proxy'] = '*'
            self.switch = PowerSwitch(hostname=ip_address, userid="admin", password="1234")

    def verify_relay(self):
        print(f"[ Info ] Verifying relay configurations")
        if self.relay_number < 1 and self.relay_number > 8:
            print(f"[ Error ] Relay number invalid")
            sys.exit(-1)
        if self.relay_type not in ["Anel","DLI"]:
            print(f"[ Error ] Relay type invalid - must be Anel or DLI")
            sys.exit(-1)
        # if not os.path.exists(self.relay_exe):
        #     print(f"[ Error ] {self.relay_exe} does not exist. Please provide a valid executable")
        #     sys.exit(-1)

    def switch_relay(self, operation="toggle"):
        print(f"[ Info ] Switching relay [operation={operation}]")

        if operation == "off":
            self.switch_relay_off()
        elif operation == "on":
            self.switch_relay_on()
        elif operation == "toggle":
            self.switch_relay_off()
            time.sleep(5)
            self.switch_relay_on()
        else:
            print(f"[ Error ] Invalid operation value for relay. Allowed values are (off,on,toggle)")
    
    def switch_relay_on(self):
        if(self.relay_type=="Anel"):
            os.system(f"pypwrctrl -d -i 12347 -o 12345 on {self.relay_ip} {self.relay_number}")
        else: # switch on using dlipower
            self.switch.on(self.relay_number)
    
    def switch_relay_off(self):
        if(self.relay_type=="Anel"):
            os.system(f"pypwrctrl -d -i 12347 -o 12345 off {self.relay_ip} {self.relay_number}")
        else: # switch off using dlipower
            self.switch.off(self.relay_number)
