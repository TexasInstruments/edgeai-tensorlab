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

| PARAMETER | DEFAULT | DETAILS |
| :---:     | :---:   | :---:   |
| soc|AM68A|Allowed values are AM68A,AM69A,TDA4VM,AM62A,AM67A|
| pc_ip|-| IP Address of your PC |
| artifacts_tarball|None|Option to provide artifacts tarball download link or path. If provided, it will delete the existing folders in work_dirs/modelartifacts/*SOC*/8bits and replace with the tarball artifacts. If None, it will use the existing artifacts under work_dirs/modelartifacts/*SOC*/8bits|
| artifacts_folder |None| Path to store the model artifacts, creates a deafult if not specified. |
| logs_dir|./evm_test_logs|Path to dum evm test logs in|
| uart|/dev/ttyUSB2|UART port of the SOC|
| reboot_type|soft|Type of EVM reboot to use. Allowed values are soft or hard.Soft reboot signifies using 'reboot' command on evm. Hard reboot signifies using ANEL Power switch to control the power supply |
| relay_exe_path|-| ANEL Power supply executable, path in case of using hard reboot |
| relay_ip_address|-| ANEL Power supply ip address, in case of using hard reboot |
| relay_power_port|-| ANEL Power supply port EVM is connect to, in case of using hard reboot |
| num_frames | - | The number of frames for which the evaluation is run | 

## Invoke the script

```console
foo@bar:~/edgeai-benchmark/tests/evm_test$ python3 main.py --soc AM68A --pc_ip xx.xx.xx.xx --artifacts_tarball model_artifacts.tar.gz
--uart /dev/ttyUSB2 --reboot_type soft
```