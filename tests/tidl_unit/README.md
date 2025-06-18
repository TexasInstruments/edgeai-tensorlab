# TIDL backend test framework
A unified pytest based test environment to test and verify Texas Instruments Deep-Learning (TIDL) models and library.

## 1. Features
- Evaluate compilation and inference of your own models to check readiness for deployement to TI SOC <br>
- Generates HTML reports and supports pytest-xdist for parallel runs.

## 2. Prerequisites
| Dependency               | Minimum version | Notes |
|--------------------------|-----------------|-------|
| Python                   | 3.x             | Tested on 3.10(x86) and 3.12(EVM) |
| pip                      | latest          | python -m pip install --upgrade pip |
| Python packages          | —               | Install once in a fresh pyenv/conda env: pip install -r requirements.txt |

## 3. Setup on X86_PC

Note: Recommended to run this in a fresh python virtual environment

```bash
# inside edgeai-benchmark directory 
./setup_pc.sh
source ./run_set_env.sh "<SOC>"     # Allowed SOC values are (AM62A, AM67A, AM68A, AM69A, TDA4VM)
cd tests/tidl_unit/
pip install -r requirements.txt
```

## 3. Structuring models and input/output buffer diectory
Three things are needed for running tests on particular model, ``model.onnx``, ``input_0.pb``, ``output_0.pb``.
These files needs to be present in a particular directory structure for pytest framework to retrieve and use them.
```
tidl_unit_test_data 
│
└───operators
    │
    └───<network_group_1>
    │   │
    │   └───<network_1>
    │   |   │   model.onnx
    │   |   └───test_data_set_0
    |   |       │   input_0.pb
    |   |       │   output_0.pb
    |   |
    |   └───<network_2>
    │       │   model.onnx
    │       └───test_data_set_0
    |           │   input_0.pb
    |           │   output_0.pb
    |
    |
    └───<network_group_2>
    │   │
    │   └───<network_3>
    │   |   │   model.onnx
    │   |   └───test_data_set_0
    |   |       │   input_0.pb
    |   |       │   output_0.pb
```
> **_NOTE:_** There can be multiple network groups and each network can have multiple networks to test. Make sure all the ``test names (<network_1>, <network_2>, <network_3>, etc.)`` should be unique across different directories also.


## 4. Running the Tests
You are now ready to run the tests. You can either directly invoke pytest or use a shell script to automatically run the tests. After test runs, an html report will be generated in ``logs`` directory

### Directly using pytest
Here are some ways you might want to run these unit tests directly using pytest:

Allowed arguments:
- ``--run-infer`` - Runs model inference. If not given model compilation runs. Default: False
- ``--runtime=*runtime*`` - Runtime to run tests. Allowed values are (onnxrt, tvmrt). Default: onnxrt
- ``--disable-tidl-offload`` - Disbale tidl-offload, runs natively with given runtime. Default: False
- ``--no-subprocess`` - Does not run tests as subprocess. Default: False
- ``--exit-on-critical-error`` - Force stop pytest on critical error (Seg faults, OpenVX errors, etc.). Default: False
- ``--timeout=*timeout*`` - Timeout per test run in seconds. Default: 300

Along with arguments, you can also modify ``tidl_unit.yaml`` to set some compilation & inference options

#### Compilation test 
- Run compilation test for a single test case: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_operator[<network_1>]
    ```
- Run compilation test for a multiple test case: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_operator[<network_1>] test_tidl_unit.py::test_tidl_unit_operator[<network_2>] test_tidl_unit.py::test_tidl_unit_operator[<network_3>]
    ```
- Run compilation test for all test cases in tidl_unit_test_data/operators directory: 
    ```bash
    pytest
    ```
> **_NOTE:_** After compilation, the compiled model artifacts are present under work_dirs/modelartifacts

#### Inference test 
- Run inference test for a single test case: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_operator[<network_1>] --run-infer
    ```
- Run inference test for a multiple test case: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_operator[<network_1>] test_tidl_unit.py::test_tidl_unit_operator[<network_2>] test_tidl_unit.py::test_tidl_unit_operator[<network_3>]  --run-infer
    ```
- Run inference test for all test cases in tidl_unit_test_data/operators directory: 
    ```bash
    pytest  --run-infer
    ```
> **_NOTE:_** Inference test picks the compiled model artifacts from work_dirs/modelartifacts directory

### Using shell script
You can also use helper script (which internally invoke pytest) to run tests:

Allowed arguments:
- ``--test_suite=*test_suite*`` - Defines test suite. Allowed values are (operator).
- ``--runtime=*runtime*`` - Runtime to run tests. Allowed values are (onnxrt, tvmrt). Default: onnxrt
- ``--run_compile=*run_compile*`` - Runs model compilation. Allowed values are (0, 1). Default: 1
- ``--run_infer=*run_infer*`` - Runs model inference. Allowed values are (0, 1). Default: 1
- ``--tidl_offload=*tidl_offload*`` - Enable tidl-offload. Allowed values are (0, 1). Default: 1  
- ``--tests=*test_names*`` - Comma separated test names to run. If not given, will run all test based on test_suite. Ex: ``--tests="<network_group_1>,<network_1>,<network_3>"``

#### Compilation test 
- Run compilation test for a single test case: 
    ```bash
    ./run_test.sh --test_suite=operator --tests="<network_1>" --run_compile=1 --run_infer=0 
    ```
- Run compilation test for a multiple test case: 
    ```bash
    ./run_test.sh --test_suite=operator --tests="<network_1>,<network_2>,<network_group_2>" --run_compile=1 --run_infer=0 
    ```
- Run compilation test for all test cases in tidl_unit_test_data/operators directory: 
    ```bash
    ./run_test.sh --test_suite=operator --run_compile=1 --run_infer=0 
    ```
> **_NOTE:_** After compilation, the compiled model artifacts are present under work_dirs/modelartifacts

#### Inference test 
- Run inference test for a single test case: 
    ```bash
    ./run_test.sh --test_suite=operator --tests="<network_1>" --run_compile=0 --run_infer=1
    ```
- Run inference test for a multiple test case: 
    ```bash
    ./run_test.sh --test_suite=operator --tests="<network_1>,<network_2>,<network_group_2>" --run_compile=0 --run_infer=1
    ```
- Run inference test for all test cases in tidl_unit_test_data/operators directory: 
    ```bash
    ./run_test.sh --test_suite=operator --run_compile=0 --run_infer=1 
    ```

- Run compilation and inference:
    ```bash
    ./run_test.sh --test_suite=operator --tests="<network_1>,<network_2>,<network_group_2>" 
    ```

> **_NOTE:_** Inference test picks the compiled model artifacts from work_dirs/modelartifacts directory

## 5. Extra Documentation

Pass/Fail Notes: [pass-fail-notes.md](docs/pass-fail-notes.md)<br>
Code Outline: [code-outline.md](docs/code-outline.md)
