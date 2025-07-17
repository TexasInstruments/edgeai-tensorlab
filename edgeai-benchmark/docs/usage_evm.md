
## Setup on EVM
## Running inference on EVM or StarterKit board
* Boot the EVM or StarterKit board using an SD card flashed with Edge AI SDK. Connect to the Linux OS in the EVM using ssh or using UART connection.
* Mount or copy the edgeai-benchmark folder on to the EVM. Mount edgeai-modelzoo repository in EVM inside the same folder where edgeai-benchmark is present. Instead of munting these separately, you can also mount the parent folder of edgeai-benchmark as well.
* Copying datasets and compiled artifacts to EVM can be time consuming and can use significant disk space (which may not be available on the EVM). To run on EVM, we recommend to share your the datasets and work_dirs folders in your PC to the EVM and using NFS. [exportfs](https://www.tutorialspoint.com/unix_commands/exportfs.htm) can be used for this NFS sharing.
* Mount the datasets from your PC at the required location in EVM (edgeai-benchmark/dependencies/datasets)

Run the setup script to install the python packages.
```
setup_evm.sh
```

## Usage on EVM
### Running inference / benchmark on *SOC/device* using pre-compiled model artifacts

**Remove all result.yaml files (if present) from all the compiled model artifacts folders - it is not sufficint to remove the results.yaml - you have to actually remove all the result.yaml files as well. 
If these file is present, the below script will just report the result from that file without actually running inference.**

Make sure that the modelartifacts_path in settings_base.yaml is pointing to the place where compiled model artifacts are present. See the "Pre-Complied Model Artifacts" section for more details.

[run_benchmarks_evm.sh](../run_benchmarks_evm.sh) can be used to run these model artifacts on device. 
```
run_benchmarks_evm.sh <SOC>
```

This will generate a report csv file containing the inference time and inference accuracy on the development board/EVM.

run_generate_report.sh can also be used to gather the results after running. 

## EVM Test Automation
Have a look at the test scripts that can automate running the benchmark on EVM in the folder [tests/evm_test](../tests/evm_test) 
