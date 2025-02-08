# This document explains how to run automatic benchmark pipeline on EVM

## Overview
The purpose of these scripts are to automate the process of running benchmark pipelines on EVM. UART is used by the script to interface with the EVM.

## EVM SETUP
1. Make sure you have EVM flashed with appropriate image and booted.
2. Connect the UART cable and note the serial port. You can use minicom to figure this out.
3. Copy the all the scripts under ``evm_scripts`` to ``~`` directory of the EVM. You can use ssh to copy or copy it over by mounting the SD card as well.

## PC SETUP
1. Install the dependencies under ``requirements_pc.txt``
2. Modify/Add json file under ``evm_config`` according to your EVM and serial port. Refer to the template under evm_config for more information about json file.


## About the script

**``main.py`` is the main script that needs to be invoked from the PC** . It has following options that can be provided by the user.

| PARAMETER | REQUIRED | DEFAULT | DETAILS |
| :---:     | :---:    | :---:   | :---:   |
| evm_config|True|-|Path to evm config json file|
| logs_dir|False|./evm_test_logs|Path to dum evm test logs in|
| reboot_type|False|soft|Type of EVM reboot to use. Allowed values are soft or hard.Soft reboot signifies using 'reboot' command on evm. Hard reboot signifies using ANEL Power switch to control the power supply |
| artifacts_tarball|False|None|Option to provide artifacts tarball download link or path. If provided, it will delete the existing folders in work_dirs/modelartifacts/*SOC*/8bits and replace with the tarball artifacts. If None, it will use the existing artifacts under work_dirs/modelartifacts/*SOC*/8bits|

## Invoke the script

```console
foo@bar:~/edgeai-benchmark/tests/evm_test$ python3 main.py --evm_config evm_config/evm_0_config
```