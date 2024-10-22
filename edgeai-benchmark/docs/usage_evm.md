
## Setup on EVM
Mount this repository on EVM. NFS mount can be used for that. Make sure that the dependencies/datasets folder is also mounted properly. 

Install the required python dependencies using requirements_evm.txt
```
pip install -r requirements_evm.txt
```

## Usage on EVM
### Running inference / benchmark on *SOC/device* using pre-compiled model artifacts

Make sure that the modelartifacts_path in settings_base.yaml is pointing to the place where compiled model artifacts are present. See the "Pre-Complied Model Artifacts" section for more details.

Remove result.yaml files (if present) from all the compiled model artifacts folders. If this file is present, the below script will just report the result from that without actually running inference.

[run_benchmarks_evm.sh](../run_benchmarks_evm.sh) can be used to run these model artifacts on device. 
```
run_benchmarks_evm.sh <SOC>
```

This will generate a report csv file containing the inference time and inference accuracy on the development board/EVM.
