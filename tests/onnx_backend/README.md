# ONNX Backend Tests

[ONNX backend tests](https://github.com/onnx/onnx/blob/main/docs/OnnxBackendTest.md) are a suite of tests written by the ONNX community. It is made up of a large number of single-operator ONNX models with inputs and expected outputs. It not only tests an entire ONNX opset, but also tests each operator under a variety of input parameters. 

## Setup

To use these tests, ensure that you have installed pytest as well as helpful plugins: `pip install pytest pytest-xdist pytest-timeout pytest-html==3.2.0`

## Usage

### Common Usage

Here are some ways you might want to run these backend tests: 

- Run compilation for all backend tests with timestamped log and report: `pytest --html=logs/$(date -d "today" +"%Y%m%d%H%M")_comp.html | tee logs/$(date -d "today" +"%Y%m%d%H%M")_comp.log`
- Run inference for all backend tests with timestamped log and report: `pytest --html=logs/$(date -d "today" +"%Y%m%d%H%M")_infer.html --run-infer | tee logs/$(date -d "today" +"%Y%m%d%H%M")_infer.log`
- Run a single node compilation test: `pytest test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding]`
- Run a single node compilation test without offload to tidl: `pytest test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding] --disable-tidl-offload`
- Run a single node inference test: `pytest test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding] --run-infer`

See notes below for an explanation of all the command-line options seen above, and more context on the tests and expected results

### Usage Notes

- These tests are implemented using pytest and can be run using standard pytest commands. For example, to run all tests using default options, use the command: `pytest`

- pytest.ini defines pytest options which will be applied by default when using the `pytest` command. See that file for a description of the options applied. 

- To disable TIDL offload, include command-line argument `--disable-tidl-offload`

- By default the tests use a host process and a sub-process in order to effectively capture the output of fatal python failures. To run each test in its own process use the command-line argument `--no-subprocess`

- Bug: Compilation and inference cannot be performed in the same script ([JIRA here](https://jira.itg.ti.com/browse/TIDL-3845)). By default, each test only performs compilation. To run inference, add command-line option `--run-infer`. Therefore to run compilation and inference for all tests, run the following commands:

```bash
pytest
pytest --run-infer
```

- Pytest xdist
    - pytest-xdist is a multiprocessing plugin for pytest. In addition to speeding up the execution of test suites, it also helps to recover from fatal python errors like segmentation faults. Usage:

    ```bash
    pytest -n num_cpus # Replace num_cpus with number or auto
    pytest -n num_cpus --max-worker-restart num_max_worker_restart # Replace num_max_worker_restart with number
    ```
    - Pytest-xdist cannot live-output stdout from tests. As a workaround we forward stdout to stderr in conftest.py

- PyTest timeout is a PyTest plugin which allows you to terminate a test after a timeout threshold is reached. This is necessary because certain tests hang indefinitely

- PyTest-html is a plugin which allows you to output a test report to a convenient html file. In the latest version of pytest-html (4.1.1 as of writing this) the report was missing test cases where pytest-xdist workers crashed due to a fatal python error. Therefore we are using an old version 3.2.0. 

- Tests are automatically retrieved from your current onnx python package location. The list of tests can be found in <onnx_python_package_root>/backend/test/data
    - You will find in the above location that there are several folders of tests which correspond to different categories of tests
    - Note that not all categories are implemented 
    - You can run each implemented category individually with the following commands:
    ```bash
    pytest pytest test_onnx_backend.py::test_onnx_backend_node # Node tests
    pytest pytest test_onnx_backend.py::test_onnx_backend_simple # Simple tests
    pytest test_onnx_backend.py::test_onnx_backend_pc # pytorch-converted
    pytest test_onnx_backend.py::test_onnx_backend_no # pytorch-operator
    ```

- Known failures are shown in backend_test_known_results.py

- Helpful PyTest options: 
    - `capture=tee-sys` Enables live output of test stdout/stderr
    - `--pdb` Will invoke pdb on nonfatal failures
        - Does not work with nultiprocessing. To disable multiprocessing, add `-n0`  or remove the `-n auto` argument from pytest.ini  
        - To invoke pdb in arbitrary location, use insert the following line in your code: `import pdb;pdb.set_trace()`
            - This does not require `--pdb` but does require `-n0`


## Pass/Fail Notes

- Test are defined as failing in pytest if they throw an exception. If compilation fails but exits gracefully, it will be defined as a pass by PyTest

- Inference will report a fail if the normalized mean-squared-error (NMSE) is above a threshold
    - If there are multiple outputs in a test, the maximum normalized mean-squared-error is selected
    - The threshold is defined in onnx_backend.yaml under the key "inference_nmse_thresholds". A default threshold is defined and may be overriden on a per-test level




