# Usage

## Common Usage

Here are some ways you might want to run these unit tests: 

- Run compilation for all unit tests: `pytest`
- Run inference for all unit tests: `pytest --run-infer`

- Run a single test-suite compilation test: `pytest test_tidl_unit.py::test_tidl_unit_operator`

- Run a single operator compilation test: `pytest test_tidl_unit.py::test_tidl_unit_operator[Convolution_1]`
- Run a single operator inference test: `pytest test_tidl_unit.py::test_tidl_unit_operator[Convolution_1] --run-infer`

Each of the above commands will output a timestamped html report in the `logs` folder


## Important Usage Notes

> **_NOTE:_**  Copy or mount the folder containing all test assets to tidl_unit_test_data/*test_suite* folder

- Tests are automatically retrieved from `tidl_unit_test_data/*test_suite*` under tidl_unit folder. For example if running operator test, the data
should be present in `tidl_unit_test_data/operators`. The file should contain following 

```
tidl_unit_test_data 
│
└───operators
    │
    └───Convolution
    │   │
    │   └───Convolution_1
    │   |   │   model.onnx
    │   |   │
    │   |   └───test_data_set_0
    |   |       │   input_0.pb
    |   |       │   output_0.pb
    |   |
    |   └───Convolution_2
    │       │   model.onnx
    │       │
    │       └───test_data_set_0
    |           │   input_0.pb
    |           │   output_0.pb
    └───Softmax
    │   │
    │   └───Softmax_1
    │   |   │   model.onnx
    │   |   │
    │   |   └───test_data_set_0
    |   |       │   input_0.pb
    |   |       │   output_0.pb
```

- `operator` is the test suite
- `Convolution_1`, `Convolution_2` etc. are all the tests that will run
- Each of the test has `model.onnx` and input/output data

## Extended Usage Notes

- These tests are implemented using pytest and can be run using standard pytest commands. For example, to run all tests using default options, use the command: `pytest`

- pytest.ini defines pytest options which will be applied by default when using the `pytest` command. See that file for a description of the options applied. 

- To disable TIDL offload, include command-line argument `--disable-tidl-offload`

- By default the tests use a host process and a sub-process in order to effectively capture the output of fatal python failures. To run each test in its own process use the command-line argument `--no-subprocess`

- Example of outputting to a timestamped log file:`pytest | tee logs/$(date -d "today" +"%Y%m%d%H%M")_log.txt`

- Known failures are shown in unit_test_known_results.py

