import os
from edgeai_benchmark import *
import pytest
import onnx
from .backend_test_known_results import expected_fails
from multiprocessing import Process
import glob


'''
Pytest file for ONNX Backend tests
Note: Pass in --disable-tidl-offload to pytest command in order to disable TIDL offload
Note: Pass in --no-subprocess to disable running the test in a subprocess
Note: Pass in --run-infer to pytest command in order to run inference (default is import which must be done first)
'''

import logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# TODO: Maybe integrate within ONNX's formal backend test framework using Backend class
# TODO: Add onnx backend full model tests

# Use onnx import to find backend node test root
onnx_root         = os.path.dirname(onnx.__file__)
data_root         = os.path.join(onnx_root, "backend/test/data")
node_tests_root   = os.path.join(data_root, "node")
pc_tests_root     = os.path.join(data_root, "pytorch-converted")
po_tests_root     = os.path.join(data_root, "pytorch-operator")
simple_tests_root = os.path.join(data_root, "simple")

# Fixtures to pass root dirs to tests
@pytest.fixture
def node_tests_root_fixture():
    return node_tests_root

# pytorch-converted
@pytest.fixture
def pc_tests_root_fixture():
    return pc_tests_root

# pytorch-operator
@pytest.fixture
def po_tests_root_fixture():
    return po_tests_root

@pytest.fixture
def simple_tests_root_fixture():
    return simple_tests_root

@pytest.fixture(scope="session")
def tidl_offload(pytestconfig):
    return not pytestconfig.getoption("disable_tidl_offload")

@pytest.fixture(scope="session")
def run_infer(pytestconfig):
    return pytestconfig.getoption("run_infer")

@pytest.fixture(scope="session")
def no_subprocess(pytestconfig):
    return pytestconfig.getoption("no_subprocess")

def retrieve_tests(root_dir):
    all_tests = os.listdir(root_dir)
    tests_with_expected_fails_marked = [pytest.param(test, marks=pytest.mark.xfail) if test in expected_fails else test for test in all_tests]
    return tests_with_expected_fails_marked

node_tests_to_run   = retrieve_tests(node_tests_root)
simple_tests_to_run = retrieve_tests(simple_tests_root)
pc_tests_to_run     = retrieve_tests(pc_tests_root)
po_tests_to_run     = retrieve_tests(po_tests_root)


# Test onnx node test
@pytest.mark.parametrize("test_name", node_tests_to_run)
def test_onnx_backend_node(no_subprocess : bool, tidl_offload : bool, run_infer : bool, node_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    '''

    perform_onnx_backend(no_subprocess=no_subprocess,
                        tidl_offload    = tidl_offload, 
                        run_infer       = run_infer, 
                        test_name       = test_name,
                        testdir_parent  = node_tests_root_fixture)
    


# Test onnx simple test
@pytest.mark.parametrize("test_name", simple_tests_to_run)
def test_onnx_backend_simple(no_subprocess : bool, tidl_offload : bool, run_infer : bool, simple_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    '''

    perform_onnx_backend(no_subprocess  = no_subprocess,
                         tidl_offload    = tidl_offload, 
                         run_infer       = run_infer, 
                         test_name       = test_name,
                         testdir_parent  = simple_tests_root_fixture)
    

# Test onnx pytorch-converted test
@pytest.mark.parametrize("test_name", pc_tests_to_run)
def test_onnx_backend_pc(no_subprocess : bool, tidl_offload : bool, run_infer : bool, pc_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    '''

    perform_onnx_backend(no_subprocess  = no_subprocess,
                         tidl_offload    = tidl_offload, 
                         run_infer       = run_infer, 
                         test_name       = test_name,
                         testdir_parent  = pc_tests_root_fixture)
    

# Test onnx pytorch-operator test
@pytest.mark.parametrize("test_name", po_tests_to_run)
def test_onnx_backend_po(no_subprocess : bool, tidl_offload : bool, run_infer : bool, po_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    '''

    perform_onnx_backend(no_subprocess  = no_subprocess,
                         tidl_offload    = tidl_offload, 
                         run_infer       = run_infer, 
                         test_name       = test_name,
                         testdir_parent  = po_tests_root_fixture)

def perform_onnx_backend(no_subprocess : bool, tidl_offload : bool, run_infer : bool, testdir_parent : str, test_name : str):
    '''
    Performs an onnx backend test
    '''
    
    if(no_subprocess):
        perform_onnx_backend_oneprocess(tidl_offload    = tidl_offload, 
                             run_infer       = run_infer, 
                             test_name       = test_name,
                             testdir_parent  = testdir_parent)
    else:
        perform_onnx_backend_subprocess(tidl_offload    = tidl_offload, 
                             run_infer       = run_infer, 
                             test_name       = test_name,
                             testdir_parent  = testdir_parent)
        


def perform_onnx_backend_subprocess(tidl_offload : bool, run_infer : bool, test_name : str, testdir_parent : str):
    '''
    Perform an onnx backend test using a subprocess (in order to properly capture output for fatal errors)
    Called by perform_onnx_backend
    '''
    
    kwargs = {"tidl_offload"   : tidl_offload, 
              "run_infer"      : run_infer, 
              "test_name"      : test_name,
              "testdir_parent" : testdir_parent}
    
    p = Process(target=perform_onnx_backend_oneprocess, kwargs=kwargs)
    p.start()
    

    # Note: This timeout parameter must be lower than the pytest-timeout parameter 
    #       passed to the pytest command (on the command line or in pytest.ini)
    p.join(timeout=60)
    if p.is_alive():
        p.terminate()

        # Cleanup leftover files 
        for f in glob.glob("/dev/shm/vashm_buff_*"):
            os.remove(f)

    assert p.exitcode == 0, f"Received nonzero exit code: {p.exitcode}"

# Utility function to perform onnx backend test
def perform_onnx_backend_oneprocess(tidl_offload : bool, run_infer : bool, test_name : str, testdir_parent : str):
    '''
    Perform an onnx backend test using without a subprocess wrapper
    Called by perform_onnx_backend_subprocess or directly by perform_onnx_backend if no_subprocess is specified
    '''
    test_dir = os.path.join(testdir_parent, test_name)

    # Check environment is set up correctly
    assert os.path.exists(test_dir), f"test path {test_dir} doesn't exist"
    assert os.environ.get('TIDL_RT_AVX_REF') is not None, "Make sure to source run_set_env.sh"
    assert os.path.exists(os.environ['TIDL_TOOLS_PATH'])
    assert os.path.exists(os.environ['ARM64_GCC_PATH'])

    # Declare config object
    cur_dir = os.path.dirname(__file__)
    settings = config_settings.ConfigSettings(os.path.join(cur_dir,'onnx_backend.yaml'), tidl_offload=tidl_offload)

    # Declare ONNX Session
    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    session_name     = constants.SESSION_NAME_ONNXRT
    session_type     = settings.get_session_type(session_name)
    runtime_options  = settings.get_runtime_options(session_name, quantization_scale_type=constants.QUANTScaleType.QUANT_SCALE_TYPE_P2, is_qat=False, debug_level = 3)
    onnx_session_cfg = sessions.get_nomeanscale_session_cfg(settings, work_dir=work_dir)
    session          = session_type(**onnx_session_cfg, runtime_options=runtime_options, model_path=os.path.join(test_dir, "model.onnx"))

    # Declare dataset
    ob_dataset  = datasets.ONNXBackendDataset(path = test_dir)

    # Declare pipeline configs to 
    pipeline_configs = {
        os.path.basename(test_dir): dict(
            dataset_category = datasets.DATASET_CATEGORY_IMAGENET,
            calibration_dataset=ob_dataset,
            input_dataset=ob_dataset,
            preprocess=preprocess.PreProcessTransforms(settings).get_transform_none(),
            session=session,
            postprocess=postprocess.PostProcessTransforms(settings).get_transform_none(),
        ),
    }
  
    
    # Run infer
    if(run_infer):
        logger.debug("Inferring")
        settings.run_import    = False
        settings.run_inference = True
        results_list = interfaces.run_accuracy(settings, work_dir, pipeline_configs)
        
        assert len(results_list) > 0, " Results not found!!!! "
        assert results_list[0].get("error") is None or len(results_list[0].get("error")) == 0, " Internal OSRT/TIDL Error:\n {} ".format(results_list[0]["error"])

        logger.debug(results_list[0]['result'])
        
        threshold = settings.inference_nmse_thresholds.get(test_name) or settings.inference_nmse_thresholds.get("default")
        if(results_list[0]['result']['max_nmse'] > threshold):
            pytest.fail(f" max_nmse of {results_list[0]['result']['max_nmse']} is higher than threshold {threshold}")
    
    # Otherwise run import
    else:
        logger.debug("Importing")
        results_list = interfaces.run_accuracy(settings, work_dir, pipeline_configs)
        assert len(results_list) > 0, " Results not found!!!! "
        assert results_list[0].get("error") is None, " Internal OSRT/TIDL Error:\n {} ".format(results_list[0]["error"])

