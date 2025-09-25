# Distrubited test
This script provides a mechanism to distribute execution of test across multiple remote PC using ssh.


## 1. Prerequisites
- Make sure your remote pc has setup completed for running pytest and you are able to run the tests manually
- Make sure the master pc, where you will execute this script from has ssh access without password to the remote pc.
  Use ssh-copy-id to add sshkey of master pc to remote pc.

## 2. Distributed test config 

The distributed_test_config.json file defines all remote pc you want to distribute the tests across

```json
{
  "user@hostname":              // Will be used as is for ssh
  {
    "test_dir" : "",            // Full path to "internal" folder where run_operator_test.sh is present
    "pyenv" : "",               // Full path to pyevn binary to activate Ex: /home/tidl/.pyenv/versions/ta_unit_test/bin/activate
    "temp_buffer_dir" : "",     // Optional: Path to redirect x86 buffers to
    "temp_nc_dir" : "",         // Optional: Path to temporary NC buffers to
    "num_threads" : 8           // Optional: Number of threads to be used
  }
}
```

## 3. Running the test

The python script is invoked with exact same parameters as run_operator_test.sh
It essentially passes on all the arguments while invoking run_operator_test.sh on remote pc via ssh
Few arguments defined in json file like temp_buffer_dir, temp_nc_dir, and num_threads are overwritten for
each remote pc

```bash
python3 distributed_test.py --SOC=AM68A --tidl_tools_path=*link to tidl_tools tarball* --compile_without_nc=1 --compile_with_nc=1 --run_ref=1 --run_natc=1 --run_ci=1  --tidl_offload=1  --operators="Abs ArgMax Cos Sin"
```
> **_NOTE:_** If operators are not defined, by default it will run all operators present in test suite.


## 4. Logs and results
The logs and results will be generated under folder `./distributed_test_output`

```
.
├── distributed_test_output/
│   ├── user1@hostname1/
│   │   ├── logs
|   |   |    ├── setup_stdout.log
|   |   |    ├── setup_stderr.log
|   |   |    ├── test_stdout.log
|   |   |    └── test_stderr.log
|   |   └── operator_test_reports/
|   |
│   ├── user2@hostname2/
│   │   ├── logs
|   |   |    ├── setup_stdout.log
|   |   |    ├── setup_stderr.log
|   |   |    ├── test_stdout.log
|   |   |    └── test_stderr.log
|   |   ├── operator_test_reports/
```

