# ONNX Backend Tests Code Outline

### Entry Point
- pytest looks for tests in files matching patterns: `test_*.py` or `*_test.py`
	- Within those files tests are functions named: `test*`
	- Onnx backend tests are implemented in functions in test_onnx_backend.py named: `test_onnx_backend_*`
		- We have a test function for each ONNX backend test suite: node, simple, pytorch-converted, pytorch-operator

### Parametrization
- Each function`test_onnx_backend_*` is marked with the decorator: `@pytest.mark.parametrize`
	- Each test suite is parametrized by the individual tests within that test suite
	- How we find test cases: Each test case has a folder in the onnx python package at `onnx/backend/test/data` 

- We also mark some of these parameters as *expected fails* by marking the parameter as follows: `pytest.param(test, marks=pytest.mark.xfail)`
	- These expected fails are retrieved from backend_test_known_results.py

### Fixtures
- You will notice each function `test_onnx_backend_*` has a large number of parameters
	- One parameter is test_name which comes from the arguments in `@pytest.mark.parametrize`
	- The other parameters are fixtures: functions used to setup the environment for a test
	- Note that each fixture argument corresponds to a fixture function that returns the data which can be accessed in the function

### Hooks
- Some parameters of `test_onnx_backend_*` are command-line options
	- We add them as command-line options using the pytest *hook*: `pytest_addoption`
		- Then the command-line option value is retrieved by the fixture
	- `pytest_addoption` is implemented in conftest.py, where directory-level hooks and fixtures can be declared

- Note the other hooks implemented in conftest.py to 
	- Configure the html report name and path 
	- Inserts the number of TIDL subgraphs into the report

### pytest.ini
- pytest.ini declares configs for pytest
	- Default command-line options to pass
	- Directories to avoid during test collection

### Backend Dataset

- ONNX backend tests define an input and expected output for each test
	- To retrieve the input and check the expected output we define `edgeai_benchmark.datasets.ONNXBackendDataset`
	- In `__init__`, we populate self.inputs and self.expected_outputs using the protobuf files in the test directory
	- In `__call__` we return the normalized mean-squared-error of the outputs with respect to the expected outputs