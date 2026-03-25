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
├───operators
│   │
│   └───<network_group_1>
│   │   │
│   │   └───<network_1>
│   │   |   │   model.onnx
│   │   |   └───test_data_set_0
│   |   |       │   input_0.pb
│   |   |       │   output_0.pb
│   |   |
│   │   └───<network_2>
│   │       │   model.onnx
│   │       └───test_data_set_0
│   |           │   input_0.pb
│   |           │   output_0.pb
│   |
│   |
│   └───<network_group_2>
│       │
│       └───<network_3>
│           │   model.onnx
│           └───test_data_set_0
│               │   input_0.pb
│               │   output_0.pb
│
└───models
    │
    └───<model_group_1>
        │
        └───<full_model_1>
            │   model.onnx
            |   config.yaml (optional for model test suite)
            └───test_data_set_0
                │   input_0.pb
                │   output_0.pb
```
> **_NOTE:_** There can be multiple network groups and each network can have multiple networks to test. Make sure all the ``test names (<network_1>, <network_2>, <network_3>, <full_model_1> etc.)`` should be unique across different directories also.

## 4. Different test suites

The TIDL unit test framework supports two different test suites:

### Operator test suite
The operator test suite is designed to test individual TIDL operators. These tests are located in the `tidl_unit_test_data/operators` directory. Each test focuses on a specific operator or a small combination of operators to verify their functionality.

Key characteristics:
- Tests individual operators or small operator combinations
- Helps isolate and debug specific operator issues
- Useful for validating operator-level optimizations
- Accessed via `test_tidl_unit_operator` in pytest

### Models test suite
The models test suite is designed to test complete neural network models. These tests are located in the `tidl_unit_test_data/models` directory. Each test represents a full model that exercises multiple operators together in a realistic network configuration.

Key characteristics:
- Tests complete neural network models
- Validates end-to-end model functionality
- Useful for testing real-world model performance and accuracy
- Accessed via `test_tidl_unit_full_model` in pytest

Both test suites follow the same directory structure pattern and require the same input/output files as described in the previous section.

## 5. Running the Tests
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
- ``--nmse-threshold=*nmse_threshold*`` - Normalized Mean Squared Error (NMSE) thresehold for inference output. Default: 0.5

Along with arguments, you can also modify ``tidl_unit.yaml`` to set some compilation & inference options

#### Compilation test 

##### Operator test suite
- Run compilation test for a single operator test case: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_operator[<network_1>]
    ```
- Run compilation test for multiple operator test cases: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_operator[<network_1>] test_tidl_unit.py::test_tidl_unit_operator[<network_2>] test_tidl_unit.py::test_tidl_unit_operator[<network_3>]
    ```
- Run compilation test for all operator test cases in tidl_unit_test_data/operators directory: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_operator
    ```

##### Models test suite
- Run compilation test for a single model test case: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_model[<full_model_1>]
    ```
- Run compilation test for multiple model test cases: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_model[<full_model_1>] test_tidl_unit.py::test_tidl_unit_model[<full_model_2>] test_tidl_unit.py::test_tidl_unit_model[<full_model_3>]
    ```
- Run compilation test for all model test cases in tidl_unit_test_data/models directory: 
    ```bash
    pytest test_tidl_unit.py::test_tidl_unit_model
    ```

> **_NOTE:_** After compilation, the compiled model artifacts are present under work_dirs/modelartifacts

#### Inference test 
Append `--run-infer` at the end of pytest commands above to run infer test 

> **_NOTE:_** Inference test picks the compiled model artifacts from work_dirs/modelartifacts directory

### Using shell script
You can also use helper script (which internally invoke pytest) to run tests:

Allowed arguments:
- ``--test_suite=*test_suite*`` - Defines test suite. Allowed values are (operator, model).
- ``--run_compile=*run_compile*`` - Runs model compilation. Allowed values are (0, 1). Default: 1
- ``--run_infer=*run_infer*`` - Runs model inference. Allowed values are (0, 1). Default: 1
- ``--test_file=*file_path*`` - Specify text file containing all tests to run. Default: NULL
- ``--tests=*test_names*`` - Comma separated test names to run. If not given, will run all test based on test_suite. Ex: ``--tests="<network_group_1>,<network_1>,<network_3>"``

> NOTE:
> 1. If 'test_file' is provided, it will take precedence over 'tests'
> 2. Along with the mentioned options you can also provide all the arguments that can be directly provided to python script. These arguments will be passed as is to the script.

#### Running Compilation and Inference test 
- Run compilation test for a single operator test case: 
    ```bash
    ./run_test.sh --test_suite=operator --tests="<network_1>" --run_compile=1 --run_infer=1
    ```
- Run compilation test for multiple operator test cases: 
    ```bash
    ./run_test.sh --test_suite=operator --tests="<network_1>,<network_2>,<network_group_2>" --run_compile=1 --run_infer=1
    ```
- Run compilation test for all test cases in tidl_unit_test_data/operators directory: 
    ```bash
    ./run_test.sh --test_suite=operator --run_compile=1 --run_infer=1
    ```
- Run compilation test for a single model test case: 
    ```bash
    ./run_test.sh --test_suite=model --tests="<full_model_1>" --run_compile=1 --run_infer=1
    ```
- Run compilation test for multiple model test cases: 
    ```bash
    ./run_test.sh --test_suite=model --tests="<full_model_1>,<full_model_2>,<model_group_3>" --run_compile=1 --run_infer=1
    ```
- Run compilation test for all test cases in tidl_unit_test_data/models directory: 
    ```bash
    ./run_test.sh --test_suite=model --run_compile=1 --run_infer=1
    ```
> **_NOTE:_** After compilation, the compiled model artifacts are present under work_dirs/modelartifacts and inference test will use the model artifacts from the same location

## 6. Extra Documentation

Pass/Fail Notes: [pass-fail-notes.md](docs/pass-fail-notes.md)<br>
Code Outline: [code-outline.md](docs/code-outline.md)
