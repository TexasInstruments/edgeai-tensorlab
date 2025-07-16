import os
from edgeai_benchmark import *
import pytest
import onnx
from .unit_test_known_results import expected_fails
from .unit_test_utils import get_tidl_performance
from multiprocessing import Process
import glob
import shutil
import numpy as np
import onnxruntime

'''
Pytest file for TIDL Unit tests
Note: Pass in --disable-tidl-offload to pytest command in order to disable TIDL offload
Note: Pass in --no-subprocess to disable running the test in a subprocess
Note: Pass in --run-infer to pytest command in order to run inference (default is import which must be done first)
Note: Pass in --exit-on-critical-error to pytest command in order to exit on critical error
Note: Pass in --flow-control to pytest command in order to run with mentioned flow control (default is 1 (REF))
Note: Pass in --temp-buffer-dir to pytest command in order to to redirect temporary buffers (default is /dev/shm)
Note: Pass in --runtime=<runtime> in order to select the compiler runtime to be used. Valid values: (onnxrt, tvmrt)
'''

import logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Find tidl unit test root
operator_tests_root = "tidl_unit_test_data/operators/"

# Fixtures to pass root dirs to tests
@pytest.fixture
def operator_tests_root_fixture():
    return operator_tests_root

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
def runtime(pytestconfig):
    return pytestconfig.getoption("runtime")

def retrieve_tests_operator(root_dir):
    subdir = os.listdir(root_dir)
    all_tests = []
    all_tests_parent_dir = []
    for i in subdir:
        x = os.path.join(root_dir,i)
        tname = os.listdir(x)
        for j in tname:
            all_tests.append(j)
            all_tests_parent_dir.append(x)
    tests_with_expected_fails_marked = [pytest.param(test, marks=pytest.mark.xfail) if test in expected_fails else test for test in all_tests]
    return tests_with_expected_fails_marked, all_tests_parent_dir

operator_tests_to_run, operator_tests_parent_dir = retrieve_tests_operator(operator_tests_root)

# Test TIDL operator unit test
@pytest.mark.parametrize(("test_name"), operator_tests_to_run)
def test_tidl_unit_operator(no_subprocess : bool, tidl_offload : bool, run_infer : bool, exit_on_critical_error : bool, flow_control : int, temp_buffer_dir : str, runtime : str, timeout : int, operator_tests_root_fixture : str, test_name : str):
    '''
    Pytest for tidl unit operator tests using the edgeai-benchmark framework
    '''
    testdir_parent = operator_tests_root_fixture
    for i in range(len(operator_tests_to_run)):
        if operator_tests_to_run[i] == test_name:
            testdir_parent = operator_tests_parent_dir[i]
            break

    perform_tidl_unit(no_subprocess   = no_subprocess,
                      tidl_offload    = tidl_offload, 
                      run_infer       = run_infer, 
                      flow_control    = flow_control,
                      temp_buffer_dir = temp_buffer_dir,
                      timeout         = timeout,
                      test_name       = test_name,
                      test_suite      = "operator",
                      testdir_parent  = testdir_parent,
                      runtime         = runtime)

def perform_tidl_unit(no_subprocess : bool, tidl_offload : bool, run_infer : bool, flow_control : int, temp_buffer_dir : str, runtime : str, timeout : int, testdir_parent : str, test_name : str, test_suite : str):
    '''
    Performs an tidl unit test
    '''
    
    if(no_subprocess):
        perform_tidl_unit_oneprocess(tidl_offload       = tidl_offload, 
                                     run_infer          = run_infer, 
                                     flow_control       = flow_control,
                                     temp_buffer_dir    = temp_buffer_dir,
                                     runtime            = runtime,
                                     test_name          = test_name,
                                     test_suite         = test_suite,
                                     testdir_parent     = testdir_parent)
    else:
        perform_tidl_unit_subprocess(tidl_offload    = tidl_offload, 
                                     run_infer       = run_infer,
                                     flow_control    = flow_control,
                                     temp_buffer_dir = temp_buffer_dir,
                                     runtime         = runtime,
                                     timeout         = timeout,
                                     test_name       = test_name,
                                     test_suite      = test_suite,
                                     testdir_parent  = testdir_parent)
        


def perform_tidl_unit_subprocess(tidl_offload : bool, run_infer : bool, flow_control : int, temp_buffer_dir : str, runtime : str, timeout : int, test_name : str, test_suite : str, testdir_parent : str):
    '''
    Perform an tidl unit test using a subprocess (in order to properly capture output for fatal errors)
    Called by perform_tidl_unit
    '''

    kwargs = {"tidl_offload"      : tidl_offload, 
              "run_infer"         : run_infer, 
              "flow_control"      : flow_control,
              "temp_buffer_dir"   : temp_buffer_dir,
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
def perform_tidl_unit_oneprocess(tidl_offload : bool, run_infer : bool,  flow_control : int, temp_buffer_dir : str, runtime : str, test_name : str, test_suite : str, testdir_parent : str):
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
        session_name = constants.SESSION_NAME_TVMDLR
    else:
        raise ValueError("Runtimes currently supported are onnxrt and tvmrt")
    model_file       = os.path.join(test_dir, "model.onnx")
    onnx.shape_inference.infer_shapes_path(model_file, model_file)

    # Create necessary directory
    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    run_dir = os.path.join(work_dir, test_name)
    model_directory = os.path.join(run_dir, 'model')
    artifacts_folder = os.path.join(run_dir, 'artifacts')

    # Declare dataset
    tidl_unit_dataset  = datasets.TIDLUnitDataset(path = test_dir)

    #Run infer
    if(run_infer):
        logger.debug("Inferring")
        settings.run_import    = False
        settings.run_inference = True
        runtime_options  = settings.get_runtime_options(session_name, is_qat=False, debug_level = 0)
        runtime_options["advanced_options:quantization_scale_type"] = constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN
        runtime_options["onnxruntime:graph_optimization_level"] = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        runtime_options["advanced_options:temp_buffer_dir"] = temp_buffer_dir
        if flow_control != -1:
            runtime_options["advanced_options:flow_ctrl"] = flow_control

        if runtime == "onnxrt":
            runtime_wrapper = core.ONNXRuntimeWrapper(runtime_options=runtime_options,
                                                      model_file=model_file,
                                                      artifacts_folder=artifacts_folder,
                                                      tidl_tools_path=None,
                                                      tidl_offload=tidl_offload)
        elif runtime == "tvmrt":
            runtime_wrapper = core.TVMDLRRuntimeWrapper(runtime_options=runtime_options,
                                                        model_file=model_file,
                                                        artifacts_folder=artifacts_folder,
                                                        tidl_tools_path=None,
                                                        tidl_offload=tidl_offload)
        else:
            # If not supported
            raise ValueError("Runtimes currently supported are onnxrt and tvmdlr")

        results_list = runtime_wrapper.run_inference(tidl_unit_dataset[0])

        assert len(results_list) > 0, " Results not found!!!! "

        logger.debug(results_list)

        threshold = settings.inference_nmse_thresholds.get(test_name) or settings.inference_nmse_thresholds.get("default")
 
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
        
        max_nmse = tidl_unit_dataset([results_list])['max_nmse']
        print(f"MAX_NMSE: {max_nmse}")
        if max_nmse == None:
            del runtime_wrapper
            pytest.fail(f" Could not calculate NMSE")
        elif(max_nmse > threshold):
            del runtime_wrapper
            pytest.fail(f" max_nmse of {max_nmse} is higher than threshold {threshold}")


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
        runtime_options["advanced_options:quantization_scale_type"] = constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN
        runtime_options["onnxruntime:graph_optimization_level"] = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        runtime_options["advanced_options:temp_buffer_dir"] = temp_buffer_dir

        if runtime == "onnxrt":
            runtime_wrapper = core.ONNXRuntimeWrapper(runtime_options=runtime_options,
                                                        model_file=model_file,
                                                        artifacts_folder=artifacts_folder,
                                                        tidl_tools_path=tidl_tools_path,
                                                        tidl_offload=tidl_offload)
            results_list = runtime_wrapper.run_import(tidl_unit_dataset[0])
            assert len(results_list) > 0, " Results not found!!!! "
        elif runtime == "tvmrt":
            runtime_wrapper = core.TVMDLRRuntimeWrapper(runtime_options=runtime_options,
                                                        model_file=model_file,
                                                        artifacts_folder=artifacts_folder,
                                                        tidl_tools_path=tidl_tools_path,
                                                        tidl_offload=tidl_offload)
            status = runtime_wrapper.run_import(tidl_unit_dataset[0])
            assert status >= 0, "TIDL Import Failed"
            assert status > 0, "TIDL Tools missing"
        else:
            # If not supported
            raise ValueError("Runtimes currently supported are onnxrt and tvmdlr")
