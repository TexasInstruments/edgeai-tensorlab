#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
AnelRelayControl is responsible for switching ANEL Elektronik AG power switch.
This requires an executable provided by anel, ip address of the switch and the port number
"""

import os
import sys
import time

class AnelRelayControl():
    def __init__(self, executable, ip_address, relay_number):
        self.relay_exe = executable
        self.relay_ip = ip_address
        self.relay_number = int(relay_number)
        self.verify_relay()

    def verify_relay(self):
        print(f"[ Info ] Verifying relay configurations")
        if self.relay_number < 1 and self.relay_number > 8:
            print(f"[ Error ] Relay number invalid")
            sys.exit(-1)
        if not os.path.exists(self.relay_exe):
            print(f"[ Error ] {self.relay_exe} does not exist. Please provide a valid executable")
            sys.exit(-1)

    def switch_relay(self, operation="toggle"):
        print(f"[ Info ] Switching realy [operatopn={operation}]")
        if operation == "off":
            os.system(f"{self.relay_exe} {self.relay_ip},75,77,rel,{self.relay_number},off,admin,anel")
        elif operation == "on":
            os.system(f"{self.relay_exe} {self.relay_ip},75,77,rel,{self.relay_number},on,admin,anel")
        elif operation == "toggle":
            os.system(f"{self.relay_exe} {self.relay_ip},75,77,rel,{self.relay_number},off,admin,anel")
            time.sleep(5)
            os.system(f"{self.relay_exe} {self.relay_ip},75,77,rel,{self.relay_number},on,admin,anel")
        else:
            print(f"[ Error ] Invalid operation value for relay. Allowed values are (off,on,toggle)")