# Distrubited test
This script provides a mechanism to distribute execution of test across multiple remote PC using ssh.


## 1. Prerequisites
- Make sure your remote pc has setup completed for running pytest and you are able to run the tests manually
- Make sure the master pc, where you will execute this script from has ssh access without password to the remote pc.
  Use ssh-copy-id to add sshkey of master pc to remote pc.

## 2. Setup
```bash
pip3 install -r requirements.txt
```

## 3. Distributed test config 

The distributed_test_config.json file defines all remote pc you want to distribute the tests across

```json
{
  "user@hostname":                            // Will be used as is for ssh
  {
    "test_dir" : "",                          // Full path to "internal" folder where run_operator_test.sh is present
    "pyenv" : "",                             // Full path to pyenv binary to activate Ex: /home/tidl/.pyenv/versions/ta_unit_test/bin/activate
    "temp_buffer_dir" : "",                   // Optional: Path to redirect x86 buffers to
    "temp_nc_dir" : "",                       // Optional: Path to temporary NC buffers to
    "num_threads" : 8,                        // Optional: Number of threads to be used
    "pc_specific_operators" : ["Sin", "Cos"]  // Optional: Force particular operator to run on this PC
  }
}
```

## 4. Running the test

The python script is invoked with exact same parameters as run_operator_test.sh
It essentially passes on all the arguments (except few) while invoking run_operator_test.sh on remote pc via ssh

Arguments which are defined in json file like `temp_buffer_dir`, `temp_nc_dir`, and `num_threads` modified before passing on to remote PC.

`pc_specific_operatos` defined in json file forces defined operators to run
on particular pc only.

By default, all the test under single operator will run on a single PC.
But the script provides an option to split test under single operator across 
multiple PC as well. `split_across_pc` is an argument to the python file which 
fines test under which operators should be split across all PCs in chunks.
This is very useful for running operators which take lots of time.
By default **Conv, Mul and Add** will always be split.

```bash
#This will run defined operators by each across PC defined by distributed_test_config.json 

python3 distributed_test.py  --operators="Abs ArgMax Cos Sin" --SOC=AM68A --tidl_tools_path=*link to tidl_tools tarball* --compile_without_nc=1 --compile_with_nc=1 --run_ref=1 --run_natc=1 --run_ci=1  --tidl_offload=1
```

```bash
# This will run defined operators by each across PC defined by distributed_test_config.json.
# Tests under operators defined in split_across_pc will be distributed across all PC and will run in chunks

python3 distributed_test.py  --operators="Abs ArgMax Cos Sin" --split_across_pc="ArgMax Sin" --SOC=AM68A --tidl_tools_path=*link to tidl_tools tarball* --compile_without_nc=1 --compile_with_nc=1 --run_ref=1 --run_natc=1 --run_ci=1  --tidl_offload=1
```

> **_NOTE:_** If operators are not defined, by default it will run all operators present in test suite.


## 4. Logs and results
The logs and results will be generated under folder `./distributed_test_output`

```
.
├── distributed_test_output/
│   ├── operator_test_reports/        # Final collated  test report 
|   |
│   ├── user1@hostname1/              # Logs and report for hostname1
│   │   ├── logs
|   |   |    ├── setup_stdout.log     # Setup output log
|   |   |    ├── setup_stderr.log     # Setup error log
|   |   |    ├── test_stdout.log      # Single test output log
|   |   |    └── test_stderr.log      # Single test error log
|   |   ├── operator_test_reports/    # Reports for operators test run on this pc 
|   |   └── chunk_reports/            # Reports for operators run on this pc as chunk 
|   |
│   ├── user2@hostname2/
│   │   ├── logs
|   |   |    ├── setup_stdout.log
|   |   |    ├── setup_stderr.log
|   |   |    ├── test_stdout.log
|   |   |    └── test_stderr.log
|   |   ├── operator_test_reports/
|   |   └── chunk_reports/
```

`distributed_test_output/operator_test_reports` will have the final collated report after testing is done.

