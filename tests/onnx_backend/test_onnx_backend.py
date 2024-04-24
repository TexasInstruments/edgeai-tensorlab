import os
from edgeai_benchmark import *
import pytest
import onnx
from .backend_test_known_results import cpu_failing_node_tests, compilation_failing_node_tests, inference_failing_node_tests

'''
Pytest file for ONNX Backend tests
Note: Pass in --disable-tidl-offload to pytest command in order to disable TIDL offload
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
    return pytestconfig.getoption("disable_tidl_offload")

@pytest.fixture(scope="session")
def run_infer(pytestconfig):
    return pytestconfig.getoption("run_infer")


node_tests_to_run   = os.listdir(node_tests_root)
simple_tests_to_run = os.listdir(simple_tests_root)
pc_tests_to_run     = os.listdir(pc_tests_root)
po_tests_to_run     = os.listdir(po_tests_root)


# Test onnx node test
@pytest.mark.timeout(method="thread") # Needed to properly terminate test that are hanging
@pytest.mark.parametrize("test_name", node_tests_to_run)
def test_onnx_backend_node(tidl_offload : bool, run_infer : bool, node_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    Note command-line options --disable-tidl-offload (disable offload to TIDL) and --run-infer (default is import, this enables inference after import)
    Example of running a single test: test_onnx_backend.py::test_onnx_backend_node[test_conv_with_strides_no_padding] --disable-tidl-offload
    '''

    # Skip tests that fail with CPU mode
    if(test_name in cpu_failing_node_tests):
        pytest.skip()

    test_dir = os.path.join(node_tests_root_fixture, test_name)
    perform_onnx_backend(tidl_offload = tidl_offload, 
                      run_infer    = run_infer, 
                      test_dir     = test_dir)


# Test onnx simple test
@pytest.mark.parametrize("test_name", simple_tests_to_run)
def test_onnx_backend_simple(tidl_offload : bool, run_infer : bool, simple_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    Note command-line options --disable-tidl-offload (disable offload to TIDL) and --run-infer (default is import, this enables inference after import)
    Example of running a single test: test_onnx_backend.py::test_onnx_backend_simple[test_expand_shape_model1] --disable-tidl-offload
    '''
    test_dir = os.path.join(simple_tests_root_fixture, test_name)
    perform_onnx_backend(tidl_offload = tidl_offload, 
                      run_infer    = run_infer, 
                      test_dir     = test_dir)
    

# Test onnx pytorch-converted test
@pytest.mark.parametrize("test_name", pc_tests_to_run)
def test_onnx_backend_pc(tidl_offload : bool, run_infer : bool, pc_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    Note command-line options --disable-tidl-offload (disable offload to TIDL) and --run-infer (default is import, this enables inference after import)
    Example of running a single test: test_onnx_backend.py::test_onnx_backend_pc[test_AvgPool1d] --disable-tidl-offload
    '''
    test_dir = os.path.join(pc_tests_root_fixture, test_name)
    perform_onnx_backend(tidl_offload = tidl_offload, 
                      run_infer    = run_infer, 
                      test_dir     = test_dir)

# Test onnx pytorch-operator test
@pytest.mark.parametrize("test_name", po_tests_to_run)
def test_onnx_backend_po(tidl_offload : bool, run_infer : bool, po_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    Note command-line options --disable-tidl-offload (disable offload to TIDL) and --run-infer (default is import, this enables inference after import)
    Example of running a single test: test_onnx_backend.py::test_onnx_backend_po[test_operator_add_broadcast] --disable-tidl-offload
    '''
    test_dir = os.path.join(po_tests_root_fixture, test_name)
    perform_onnx_backend(tidl_offload = tidl_offload, 
                      run_infer    = run_infer, 
                      test_dir     = test_dir)


# Utility function to perform onnx backend test
def perform_onnx_backend(tidl_offload : bool, run_infer : bool, test_dir : str):
  
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
        assert results_list[0].get("error") is None, " Internal OSRT/TIDL Error:\n {} ".format(results_list[0]["error"])

        logger.debug(results_list[0]['result'])
        
        test_name = os.path.basename(test_dir)
        threshold = settings.inference_nmse_thresholds.get(test_name) or settings.inference_nmse_thresholds.get("default")
        if(results_list[0]['result']['max_nmse'] > threshold):
            pytest.fail(f" max_nmse of {results_list[0]['result']['max_nmse']} is higher than threshold {threshold}")
    
    # Otherwise run import
    else:
        logger.debug("Importing")
        results_list = interfaces.run_accuracy(settings, work_dir, pipeline_configs)
        assert len(results_list) > 0, " Results not found!!!! "
        assert results_list[0].get("error") is None, " Internal OSRT/TIDL Error:\n {} ".format(results_list[0]["error"])





