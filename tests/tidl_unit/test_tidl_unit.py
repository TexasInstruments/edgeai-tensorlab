import os
from edgeai_benchmark import *
import pytest
import onnx
from .unit_test_known_results import expected_fails
from .unit_test_utils import get_tidl_performance
from .unit_test_utils import remove_dir
from multiprocessing import Process
import glob
import shutil
import numpy as np
import onnxruntime
import yaml

'''
Pytest file for TIDL Unit tests
Note: Pass in --disable-tidl-offload to pytest command in order to disable TIDL offload
Note: Pass in --no-subprocess to disable running the test in a subprocess
Note: Pass in --run-infer to pytest command in order to run inference (default is import which must be done first)
Note: Pass in --exit-on-critical-error to pytest command in order to exit on critical error
Note: Pass in --flow-control to pytest command in order to run with mentioned flow control (default is 1 (REF))
Note: Pass in --temp-buffer-dir to pytest command in order to to redirect temporary buffers (default is /dev/shm)
Note: Pass in --runtime=<runtime> in order to select the compiler runtime to be used. Valid values: (onnxrt, tvmrt)

Available test suites:
- test_tidl_unit_operator: Run tests for individual operators
- test_tidl_unit_model: Run tests for full model
'''

import logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Find tidl unit test root
operator_tests_root = "tidl_unit_test_data/operators/"
model_tests_root = "tidl_unit_test_data/models/"

# Fixtures to pass root dirs to tests
@pytest.fixture
def operator_tests_root_fixture():
    return operator_tests_root

@pytest.fixture
def model_tests_root_fixture():
    return model_tests_root

@pytest.fixture(scope="session")
def tidl_offload(pytestconfig):
    return not pytestconfig.getoption("disable_tidl_offload")

@pytest.fixture(scope="session")
def run_infer(pytestconfig):
    return pytestconfig.getoption("run_infer")

@pytest.fixture(scope="session")
def no_subprocess(pytestconfig):
    return pytestconfig.getoption("no_subprocess")

@pytest.fixture(scope="session")
def exit_on_critical_error(pytestconfig):
    return pytestconfig.getoption("exit_on_critical_error")

@pytest.fixture(scope="session")
def timeout(pytestconfig):
    return pytestconfig.getoption("timeout")

@pytest.fixture(scope="session")
def flow_control(pytestconfig):
    return pytestconfig.getoption("flow_control")

@pytest.fixture(scope="session")
def temp_buffer_dir(pytestconfig):
    return pytestconfig.getoption("temp_buffer_dir")

@pytest.fixture(scope="session")
def temp_nc_dir(pytestconfig):
    return pytestconfig.getoption("temp_nc_dir")

@pytest.fixture(scope="session")
def nmse_threshold(pytestconfig):
    return pytestconfig.getoption("nmse_threshold")

@pytest.fixture(scope="session")
def runtime(pytestconfig):
    return pytestconfig.getoption("runtime")

@pytest.fixture(scope="session")
def work_dir(pytestconfig):
    return pytestconfig.getoption("work_dir")

@pytest.fixture(scope="session")
def test_file(pytestconfig):
    return pytestconfig.getoption("test_file")

def retrieve_tests(root_dir, test_file = None):
    if not os.path.exists(root_dir):
        return [], []

    subdir = os.listdir(root_dir)
    all_tests = []
    all_tests_parent_dir = []
    for i in subdir:
        x = os.path.join(root_dir,i)
        tname = os.listdir(x)
        for j in tname:
            all_tests.append(j)
            all_tests_parent_dir.append(x)

    # Filter tests if a test file is provided
    if test_file:
        test_file = test_file.strip()

    if test_file and test_file != ""  and os.path.exists(test_file):
        with open(test_file, 'r') as f:
            tests_from_file = [line.strip() for line in f.readlines()]
            tests_from_file = [line for line in tests_from_file if line and not line.startswith('#')]

        # Filter all_tests to only include tests from the file
        filtered_tests = []
        filtered_tests_parent_dir = []
        for i, test in enumerate(all_tests):
            if test in tests_from_file:
                filtered_tests.append(test)
                filtered_tests_parent_dir.append(all_tests_parent_dir[i])

        all_tests = filtered_tests
        all_tests_parent_dir = filtered_tests_parent_dir

    tests_with_expected_fails_marked = [pytest.param(test, marks=pytest.mark.xfail) if test in expected_fails else test for test in all_tests]
    return tests_with_expected_fails_marked, all_tests_parent_dir

# Get all tests initially
operator_tests_to_run, operator_tests_parent_dir = retrieve_tests(operator_tests_root)
model_tests_to_run, model_tests_parent_dir = retrieve_tests(model_tests_root)

# Store test-specific parent directories
test_parent_dirs = {}

def pytest_generate_tests(metafunc):
    """
    Dynamically generate tests based on the test_file parameter if provided.
    This is called by pytest before test collection.
    """

    if 'test_name' in metafunc.fixturenames:
        global test_parent_dirs
        test_parent_dirs = {}

        # Get the test_file fixture value
        test_file_path = metafunc.config.getoption("test_file")

        # Determine which test suite to use based on the test function name
        if metafunc.function.__name__ == 'test_tidl_unit_model':
            tests_to_use = model_tests_to_run
            tests_parent_dir_to_use = model_tests_parent_dir
            test_root = model_tests_root
        else: 
            tests_to_use = operator_tests_to_run
            tests_parent_dir_to_use = operator_tests_parent_dir
            test_root = operator_tests_root

        # If a test file is provided, filter the tests
        if test_file_path:
            filtered_tests, filtered_parent_dirs = retrieve_tests(test_root, test_file_path)

            # Update the test_parent_dirs dictionary with the filtered tests
            for i, test in enumerate(filtered_tests):
                if isinstance(test, str):
                    test_parent_dirs[test] = filtered_parent_dirs[i]
                else:  # Handle pytest.param objects
                    test_parent_dirs[test.values[0]] = filtered_parent_dirs[i]

            metafunc.parametrize("test_name", filtered_tests)
        else:
            for i, test in enumerate(tests_to_use):
                if isinstance(test, str):
                    test_parent_dirs[test] = tests_parent_dir_to_use[i]
                else:  # Handle pytest.param objects
                    test_parent_dirs[test.values[0]] = tests_parent_dir_to_use[i]

            metafunc.parametrize("test_name", tests_to_use)

# Test TIDL operator unit test
def test_tidl_unit_operator(no_subprocess : bool, tidl_offload : bool, run_infer : bool, work_dir : str, exit_on_critical_error : bool, flow_control : int, temp_buffer_dir : str, temp_nc_dir : str, nmse_threshold : float, runtime : str, timeout : int, operator_tests_root_fixture : str, test_name : str):
    '''
    Pytest for tidl unit operator tests using the edgeai-benchmark framework
    '''
    # Get the parent directory for this test from the test_parent_dirs dictionary
    if test_name in test_parent_dirs:
        testdir_parent = test_parent_dirs[test_name]
    else:
        # Fallback to the old method if not found in the dictionary
        testdir_parent = operator_tests_root_fixture
        for i in range(len(operator_tests_to_run)):
            if isinstance(operator_tests_to_run[i], str) and operator_tests_to_run[i] == test_name:
                testdir_parent = operator_tests_parent_dir[i]
                break
            elif hasattr(operator_tests_to_run[i], 'values') and operator_tests_to_run[i].values[0] == test_name:
                testdir_parent = operator_tests_parent_dir[i]
                break

    perform_tidl_unit(no_subprocess   = no_subprocess,
                      tidl_offload    = tidl_offload, 
                      run_infer       = run_infer, 
                      work_dir        = work_dir,
                      flow_control    = flow_control,
                      temp_buffer_dir = temp_buffer_dir,
                      temp_nc_dir     = temp_nc_dir,
                      nmse_threshold  = nmse_threshold,
                      timeout         = timeout,
                      test_name       = test_name,
                      test_suite      = "operator",
                      testdir_parent  = testdir_parent,
                      runtime         = runtime)

# Test TIDL full model unit test
def test_tidl_unit_model(no_subprocess : bool, tidl_offload : bool, run_infer : bool, work_dir : str, exit_on_critical_error : bool, flow_control : int, temp_buffer_dir : str, temp_nc_dir : str, nmse_threshold : float, runtime : str, timeout : int, model_tests_root_fixture : str, test_name : str):
    '''
    Pytest for tidl unit full model tests using the edgeai-benchmark framework
    '''
    # Get the parent directory for this test from the test_parent_dirs dictionary
    if test_name in test_parent_dirs:
        testdir_parent = test_parent_dirs[test_name]
    else:
        # Fallback to the old method if not found in the dictionary
        testdir_parent = model_tests_root_fixture
        for i in range(len(model_tests_to_run)):
            if isinstance(model_tests_to_run[i], str) and model_tests_to_run[i] == test_name:
                testdir_parent = model_tests_parent_dir[i]
                break
            elif hasattr(model_tests_to_run[i], 'values') and model_tests_to_run[i].values[0] == test_name:
                testdir_parent = model_tests_parent_dir[i]
                break

    perform_tidl_unit(no_subprocess   = no_subprocess,
                      tidl_offload    = tidl_offload, 
                      run_infer       = run_infer, 
                      work_dir        = work_dir,
                      flow_control    = flow_control,
                      temp_buffer_dir = temp_buffer_dir,
                      temp_nc_dir     = temp_nc_dir,
                      nmse_threshold  = nmse_threshold,
                      timeout         = timeout,
                      test_name       = test_name,
                      test_suite      = "model",
                      testdir_parent  = testdir_parent,
                      runtime         = runtime)

def perform_tidl_unit(no_subprocess : bool, tidl_offload : bool, run_infer : bool, work_dir : str, flow_control : int, temp_buffer_dir : str, temp_nc_dir : str, nmse_threshold : float, runtime : str, timeout : int, testdir_parent : str, test_name : str, test_suite : str):
    '''
    Performs an tidl unit test
    '''
    
    if(no_subprocess):
        perform_tidl_unit_oneprocess(tidl_offload       = tidl_offload, 
                                     run_infer          = run_infer, 
                                     work_dir           = work_dir,
                                     flow_control       = flow_control,
                                     temp_buffer_dir    = temp_buffer_dir,
                                     temp_nc_dir        = temp_nc_dir,
                                     nmse_threshold     = nmse_threshold,
                                     runtime            = runtime,
                                     test_name          = test_name,
                                     test_suite         = test_suite,
                                     testdir_parent     = testdir_parent)
    else:
        perform_tidl_unit_subprocess(tidl_offload    = tidl_offload, 
                                     run_infer       = run_infer,
                                     work_dir        = work_dir,
                                     flow_control    = flow_control,
                                     temp_buffer_dir = temp_buffer_dir,
                                     temp_nc_dir     = temp_nc_dir,
                                     nmse_threshold  = nmse_threshold,
                                     runtime         = runtime,
                                     timeout         = timeout,
                                     test_name       = test_name,
                                     test_suite      = test_suite,
                                     testdir_parent  = testdir_parent)
        


def perform_tidl_unit_subprocess(tidl_offload : bool, run_infer : bool, work_dir : str, flow_control : int, temp_buffer_dir : str, temp_nc_dir : str, nmse_threshold : float, runtime : str, timeout : int, test_name : str, test_suite : str, testdir_parent : str):
    '''
    Perform an tidl unit test using a subprocess (in order to properly capture output for fatal errors)
    Called by perform_tidl_unit
    '''

    kwargs = {"tidl_offload"      : tidl_offload, 
              "run_infer"         : run_infer, 
              "work_dir"          : work_dir,
              "flow_control"      : flow_control,
              "temp_buffer_dir"   : temp_buffer_dir,
              "temp_nc_dir"       : temp_nc_dir,
              "nmse_threshold"    : nmse_threshold,
              "runtime"           : runtime,
              "test_name"         : test_name,
              "test_suite"        : test_suite,
              "testdir_parent"    : testdir_parent}
    
    p = Process(target=perform_tidl_unit_oneprocess, kwargs=kwargs)
    p.start()
    

    # Note: This timeout parameter must be lower than the pytest-timeout parameter 
    #       passed to the pytest command (on the command line or in pytest.ini)
    p.join(timeout=timeout*0.90)
    if p.is_alive():
        p.terminate()
        print("PROCESS TIMED OUT")
        # Cleanup leftover files 
        for f in glob.glob(f"{temp_buffer_dir}/vashm_buff_*"):
            os.remove(f)

    assert p.exitcode == 0, f"Received nonzero exit code: {p.exitcode}"

# Utility function to perform tidl unit test
def perform_tidl_unit_oneprocess(tidl_offload : bool, run_infer : bool, work_dir : str, flow_control : int, temp_buffer_dir : str, temp_nc_dir : str, nmse_threshold : float, runtime : str, test_name : str, test_suite : str, testdir_parent : str):
    '''
    Perform an tidl unit test using without a subprocess wrapper
    Called by perform_tidl_unit_subprocess or directly by perform_tidl_unit if no_subprocess is specified
    '''
    test_dir = os.path.join(testdir_parent, test_name)

    # Check environment is set up correctly
    assert os.path.exists(test_dir), f"test path {test_dir} doesn't exist"

    # Check flow_control
    if flow_control != -1:
        assert flow_control in [0,1,12], f"flow_control {flow_control} not supported, allowed values are 0, 1, or 12"

    # Declare config object
    cur_dir = os.path.dirname(__file__)
    settings = config_settings.ConfigSettings(os.path.join(cur_dir,'tidl_unit.yaml'), tidl_offload=tidl_offload)
    if runtime == "onnxrt":
        session_name = constants.SESSION_NAME_ONNXRT
    elif runtime == "tvmrt":
        session_name = constants.SESSION_NAME_TVMRT
    else:
        raise ValueError("Runtimes currently supported are onnxrt and tvmrt")
    
    # Parse model file and run shape inference
    model_file       = os.path.join(test_dir, "model.onnx")
    onnx.shape_inference.infer_shapes_path(model_file, model_file)

    specific_config_file = os.path.join(test_dir, "config.yaml")
    common_config_file = os.path.join(os.path.join(test_dir, "../"), "config.yaml")

    # Create necessary directory
    if work_dir == "":
        work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    run_dir = os.path.join(work_dir, test_name)
    model_directory = os.path.join(run_dir, 'model')
    artifacts_folder = os.path.join(run_dir, 'artifacts')

    # Declare dataset
    tidl_unit_dataset  = datasets.TIDLUnitDataset(path = test_dir)
    input_data, _ = tidl_unit_dataset[0]

    #Run infer
    if(run_infer):
        logger.debug("Inferring")
        settings.run_import    = False
        settings.run_inference = True

        # Set inference options
        runtime_options  = settings.get_runtime_options(session_name, is_qat=False, debug_level = 0)
        runtime_options["onnxruntime:graph_optimization_level"] = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        runtime_options["onnxruntime:intra_op_num_threads"] = 1

        # For model try parsing the compilation options from config.yaml if present
        if test_suite == "model":
            runtime_options["onnxruntime:graph_optimization_level"] = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
            common_config = None
            if os.path.exists(common_config_file):
                with open(common_config_file, "r") as f:
                    try:
                        common_config = yaml.safe_load(f)
                    except Exception as e:
                        print(f"\n[WARN] Could not load {common_config_file}\n")

            specific_config = None
            if os.path.exists(specific_config_file):
                with open(specific_config_file, "r") as f:
                    try:
                        specific_config = yaml.safe_load(f)
                    except Exception as e:
                        print(f"\n[WARN] Could not load {specific_config_file}\n")

            if common_config:
                if 'infer_options' in common_config:
                    runtime_options.update(common_config['infer_options'])

            if specific_config:
                if 'infer_options' in specific_config:
                    runtime_options.update(specific_config['infer_options'])
                if 'models' in specific_config:
                    for _,val in specific_config['models'].items():
                        if 'disable_onnx_optimizer' in val and val['disable_onnx_optimizer'] == 1:
                            runtime_options["onnxruntime:graph_optimization_level"] = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                        if 'nmse_threshold' in val and nmse_threshold < 0:
                            nmse_threshold = val['nmse_threshold']
                        break

        # Overwrite with provided args 
        runtime_options["tensor_bits"] = settings.tensor_bits
        runtime_options["advanced_options:temp_buffer_dir"] = temp_buffer_dir
        if flow_control != -1:
            runtime_options["advanced_options:flow_ctrl"] = flow_control

        if  nmse_threshold < 0:
            nmse_threshold = 0.5 # Default NMSE Threshold value

        if (tidl_offload == True):
            print("TIDL Offload Enabled\n")
        else:
            print("TIDL Offload Disabled\n")
        print(f"NMSE Threshold:\n{nmse_threshold}\n")

        # Initialize runtime and run
        if runtime == "onnxrt":
            runtime_wrapper = core.ONNXRuntimeWrapper(runtime_options=runtime_options,
                                                      model_file=model_file,
                                                      artifacts_folder=artifacts_folder,
                                                      tidl_tools_path=None,
                                                      tidl_offload=tidl_offload)
        elif runtime == "tvmrt":
            runtime_wrapper = core.TVMRuntimeWrapper(runtime_options=runtime_options,
                                                        model_file=model_file,
                                                        artifacts_folder=artifacts_folder,
                                                        tidl_tools_path=None,
                                                        tidl_offload=tidl_offload)
        else:
            # If not supported
            raise ValueError("Runtimes currently supported are onnxrt and tvmrt")

        results_list = runtime_wrapper.run_inference(input_data)

        assert len(results_list) > 0, " Results not found!!!! "

        logger.debug(results_list)
 
        # Report performance
        stats = get_tidl_performance(runtime_wrapper.interpreter, session_name=runtime)
        print(f"\nPERFORMANCE:\n")
        print(f"\tNum TIDL Subgraphs                    :   {stats['num_subgraphs']}")
        print(f"\tTotal Time (ms)                       :   {stats['total_time']:.2f}")
        print(f"\tCore Time (ms)                        :   {stats['core_time']:.2f}")
        print(f"\tTIDL Subgraphs Processing Time (ms)   :   {stats['subgraph_time']:.2f}")
        print(f"\tDDR Read Bandwidth (MB/s)             :   {stats['read_total']:.2f}")
        print(f"\tDDR Write Bandwidth (MB/s)            :   {stats['write_total']:.2f}")
        print()

        nmse  = tidl_unit_dataset.evaluate(results_list)['nmse']
        mse   = tidl_unit_dataset.evaluate(results_list)['mse']
        delta = tidl_unit_dataset.evaluate(results_list)['delta']

        if any(x is None for x in nmse):
            max_nmse = None
        else:
            max_nmse = max(nmse)

        if any(x is None for x in mse):
            max_mse = None
        else:
            max_mse = max(mse)
        
        if any(x is None for x in delta):
            max_delta = None
        else:
            max_delta = max(delta)

        if max_nmse == None:
            print("MAX_NMSE: None")
        else:
            print("MAX_NMSE: {:.7f}".format(max_nmse))

        if max_mse == None:
            print("MAX_MSE: None")
        else:
            print("MAX_MSE: {:.7f}".format(max_mse))

        if max_delta == None:
            print("MAX_DELTA: None")
        else:
            print("MAX_DELTA: {:.7f}".format(max_delta))

        # max_nmse can be none if output has zero variance - check max_mse in this case
        if max_nmse == None and max_mse == None:
            del runtime_wrapper
            pytest.fail(f" Could not calculate NMSE")
        elif max_nmse != None:
            if max_nmse > nmse_threshold:
                del runtime_wrapper
                pytest.fail(f" max_nmse of {max_nmse} is higher than threshold {nmse_threshold}")
        elif max_mse != None:
            if max_mse > nmse_threshold:
                del runtime_wrapper
                pytest.fail(f" max_mse of {max_mse} is higher than threshold {nmse_threshold}")


    #Otherwise run import
    else:
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(model_directory, exist_ok=True)
        shutil.copy2(model_file, os.path.join(model_directory,"model.onnx"))
        os.makedirs(artifacts_folder, exist_ok=True)

        tidl_tools_path = os.environ.get('TIDL_TOOLS_PATH')
        assert tidl_tools_path is not None, "TIDL_TOOLS_PATH not set"
        assert os.path.exists(tidl_tools_path)

        logger.debug("Importing")
        runtime_options  = settings.get_runtime_options(session_name, is_qat=False, debug_level = 0)

        # Set compile options
        runtime_options["onnxruntime:intra_op_num_threads"] = 1
        runtime_options["onnxruntime:graph_optimization_level"] = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        runtime_options["advanced_options:quantization_scale_type"] = constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN

        # For model try parsing the compilation options from config.yaml if present
        if test_suite == "model":
            runtime_options["onnxruntime:graph_optimization_level"] = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
            common_config = None
            if os.path.exists(common_config_file):
                with open(common_config_file, "r") as f:
                    try:
                        common_config = yaml.safe_load(f)
                    except Exception as e:
                        print(f"\n[WARN] Could not load {common_config_file}\n")

            specific_config = None
            if os.path.exists(specific_config_file):
                with open(specific_config_file, "r") as f:
                    try:
                        specific_config = yaml.safe_load(f)
                    except Exception as e:
                        print(f"\n[WARN] Could not load {specific_config_file}\n")

            if common_config:
                if 'compile_options' in common_config:
                    runtime_options.update(common_config['compile_options'])

            if specific_config:
                if 'compile_options' in specific_config:
                    runtime_options.update(specific_config['compile_options'])
                if 'models' in specific_config:
                    for _,val in specific_config['models'].items():
                        if 'disable_onnx_optimizer' in val and val['disable_onnx_optimizer'] == 1:
                            runtime_options["onnxruntime:graph_optimization_level"] = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                        break

        # Overwrite with provided args 
        runtime_options["tensor_bits"] = settings.tensor_bits
        runtime_options["advanced_options:temp_buffer_dir"] = temp_buffer_dir
        runtime_options["advanced_options:nc_temp_info_dir"] = temp_nc_dir

        if (tidl_offload == True):
            print("TIDL Offload Enabled\n")
            print(f"Model Compilation Options:\n{runtime_options}\n")
        else:
            print("TIDL Offload Disabled\n")
        
        # Initialize runtime and run
        if runtime == "onnxrt":
            runtime_wrapper = core.ONNXRuntimeWrapper(runtime_options=runtime_options,
                                                        model_file=model_file,
                                                        artifacts_folder=artifacts_folder,
                                                        tidl_tools_path=tidl_tools_path,
                                                        tidl_offload=tidl_offload)
            results_list = runtime_wrapper.run_import(input_data)
            remove_dir(os.path.join(artifacts_folder, "tempDir"))
            assert len(results_list) > 0, " Results not found!!!! "
        elif runtime == "tvmrt":
            runtime_wrapper = core.TVMRuntimeWrapper(runtime_options=runtime_options,
                                                        model_file=model_file,
                                                        artifacts_folder=artifacts_folder,
                                                        tidl_tools_path=tidl_tools_path,
                                                        tidl_offload=tidl_offload)
            status = runtime_wrapper.run_import(input_data)
            remove_dir(os.path.join(artifacts_folder, "tempDir"))
            assert status >= 0, "TIDL Import Failed"
            assert status > 0, "TIDL Tools missing"
        else:
            # If not supported
            raise ValueError("Runtimes currently supported are onnxrt and tvmrt")
