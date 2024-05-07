# ONNX Backend Tests

[ONNX backend tests](https://github.com/onnx/onnx/blob/main/docs/OnnxBackendTest.md) are a suite of tests written by the ONNX community. It is made up of a large number of single-operator ONNX models with inputs and expected outputs. It not only tests an entire ONNX opset, but also tests each operator under a variety of input parameters. 

## Setup

To use these tests, ensure that you have installed pytest: `pip install pytest pytest-xdist pytest-timeout`

We are also installing helpful plugins pytest-xdist and pytest-timeout

## Usage

### Common Usage

Here are some ways you might want to run these backend tests: 

- Run compilation for all backend tests with pytest-xdist: `pytest -n auto --max-worker-restart 500 -v --capture=tee-sys --timeout=10 | tee compilation_log.txt`
- Run inference for all backend tests with pytest-xdist: `pytest -n auto --max-worker-restart 500 -v  --capture=tee-sys --run-infer --timeout=10 | tee inference_log.txt`
- Run a single node compilation test: `pytest test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding] --capture=tee-sys -v`
- Run a single node compilation test without offload to tidl: `pytest test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding] --capture=tee-sys -v --disable-tidl-offload`
- Run a single node inference test: `pytest test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding] --capture=tee-sys -v --run-infer`

TODO: Create scripts that wrap around these command for a simpler user interface

See notes below for an explanation of all the command-line options seen above, and more context on the tests and expected results

### Usage Notes

- These tests are implemented using pytest and can be run using standard pytest commands. For example, to run all tests using default options, use the command: `pytest`

- To disable TIDL offload, include command-line argument `--disable-tidl-offload`

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

- Known failures are shown at the top of test_onnx_backend.py
    - This list is not comprehensive: Certain tests hang indefinitely and aren't able to be killed with a simple keyboard interrupt

- Helpful PyTest options: 
    - `capture=tee-sys` Enables live output of tests
    - `-v` Adds more pytest info to log including printing the test names
    - `--pdb` Will invoke pdb on nonfatal failures
        - Note: To invoke pdb in arbitrary location, use insert the following line in your code: `import pdb;pdb.set_trace()`


## Pass/Fail Notes

- Test are defined as failing in pytest if they throw an exception. If compilation fails but exits gracefully, it will be defined as a pass by PyTest

- Inference will report a fail if the normalized mean-squared-error (NMSE) is above a threshold. If there are multiple outputs, the maximum normalized mean-squared-error is selected




