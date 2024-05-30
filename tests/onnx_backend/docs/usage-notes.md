# Usage

## Common Usage

Here are some ways you might want to run these backend tests: 

- Run compilation for all backend tests: `pytest`
- Run inference for all backend tests: `pytest --run-infer`
- Run a single node compilation test: `pytest test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding]`
- Run a single node compilation test without offload to tidl: `pytest test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding] --disable-tidl-offload`
- Run a single node inference test: `pytest test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding] --run-infer`

Each of the above commands will output a timestamped html report in the `logs` folder


## Extended Usage Notes

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

- PyTest timeout is a PyTest plugin which allows you to terminate a test after a timeout threshold is reached. This is necessary for certain tests that hang indefinitely (even with timeout handled in)

- PyTest-html is a plugin which allows you to output a test report to a convenient html file. Use the `--html` option to override the log file name. 
    -  In the latest version of pytest-html (4.1.1 as of writing this) the report was missing test cases where pytest-xdist workers crashed due to a fatal python error. Therefore we are using an old version 3.2.0

- Example of outputting to a timestamped log file:`pytest | tee logs/$(date -d "today" +"%Y%m%d%H%M")_log.txt`

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
        - Does not work with nultiprocessing. To disable multiprocessing (xdist and subprocess execution), add `-n0 --no-subprocess` 
        - To invoke pdb in arbitrary location, use insert the following line in your code: `import pdb;pdb.set_trace()`
            - This does not require `--pdb` but does require `-n0 --no-subprocess`

