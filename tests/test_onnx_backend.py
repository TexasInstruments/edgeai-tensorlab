import os
import tempfile
import argparse
import cv2
from edgeai_benchmark import *
import pytest
import onnx
'''
Pytest file for ONNX Backend tests
Note: Pass in --disable-tidl-offload to pytest command in order to disable TIDL offload
Note: Pass in --run-infer to pytest command in order to run inference (default is import which must be done first)
'''

import logging

logger = logging.getLogger(__name__)

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


# Fail when TIDL offload is disabled 
# Interestingly not the same nodes as 
cpu_failing_testcases = ['test_sequence_insert_at_back', \
'test_sequence_map_add_1_sequence_1_tensor', \
'test_identity_opt', \
'test_optional_has_element_optional_input', \
'test_optional_has_element_tensor_input', \
'test_if_seq', \
'test_identity_sequence', \
'test_sequence_map_add_1_sequence_1_tensor_expanded', \
'test_sequence_map_add_2_sequences', \
'test_loop13_seq', \
'test_loop16_seq_none', \
'test_sequence_map_identity_1_sequence_expanded', \
'test_sequence_insert_at_front', \
'test_sequence_map_identity_2_sequences_expanded', \
'test_optional_get_element_optional_sequence', \
'test_optional_get_element_optional_tensor', \
'test_sequence_map_identity_1_sequence_1_tensor', \
'test_optional_get_element_sequence', \
'test_sequence_map_identity_1_sequence_1_tensor_expanded', \
'test_sequence_map_extract_shapes', \
'test_sequence_map_identity_1_sequence', \
'test_sequence_map_add_2_sequences_expanded', \
'test_optional_has_element_empty_optional_input', \
'test_sequence_map_extract_shapes_expanded', \
'test_if_opt', \
'test_sequence_map_identity_2_sequences']

# Noncomprehensive list of failing nodes
# Note: many other nodes fail in the same way when called amongst other tests, suggesting a race condition
fatal_python_error = ['test_argmax_default_axis_example', \
'test_argmax_default_axis_example_select_last_index', \
'test_argmax_default_axis_random', \
'test_argmax_default_axis_random_select_last_index', \
'test_argmax_negative_axis_keepdims_example', \
'test_argmax_negative_axis_keepdims_example_select_last_index', \
'test_argmax_negative_axis_keepdims_random', \
'test_argmax_negative_axis_keepdims_random_select_last_index', \
'test_batchnorm_epsilon',\
'test_batchnorm_epsilon_training_mode',\
'test_bitshift_right_uint64', \
'test_bitshift_right_uint8', \
'test_bitwise_or_i16_4d', \
'test_bitwise_or_i32_2d', \
'test_bitwise_xor_i16_3d', \
'test_bitwise_xor_i32_2d', \
'test_blackmanwindow_expanded', \
'test_blackmanwindow_symmetric_expanded', \
'test_cast_STRING_to_FLOAT', \
'test_ceil', \
'test_ceil_example', \
'test_celu_expanded', \
'test_clip_default_int8_inbounds', \
'test_concat_2d_axis_0', \
'test_concat_2d_axis_1', \
'test_concat_2d_axis_negative_1', \
'test_concat_2d_axis_negative_2', \
'test_concat_3d_axis_0', \
'test_concat_3d_axis_negative_1', \
'test_concat_3d_axis_negative_2', \
'test_concat_3d_axis_negative_3', \
'test_constantofshape_float_ones', \
'test_constantofshape_int_zeros', \
'test_constant_pad_axes', \
'test_cosh', \
'test_depthtospace_crd_mode_example', \
'test_depthtospace_example', \
'test_det_2d', \
'test_dft_axis', \
'test_dft_inverse', \
'test_div_uint8', \
'test_dropout_default', \
'test_dropout_default_mask', \
'test_dropout_default_old', \
'test_dropout_random_old', \
'test_dynamicquantizelinear', \
'test_dynamicquantizelinear_expanded', \
'test_dynamicquantizelinear_max_adjusted', \
'test_dynamicquantizelinear_max_adjusted_expanded', \
'test_dynamicquantizelinear_min_adjusted_expanded', \
'test_edge_pad', \
'test_einsum_sum', \
'test_einsum_transpose', \
'test_elu', \
'test_elu_example', \
'test_equal', \
'test_eyelike_without_dtype', \
'test_floor_example', \
'test_globalaveragepool_precomputed', \
'test_globalmaxpool_precomputed', \
'test_greater_equal', \
'test_greater_equal_expanded', \
'test_group_normalization_epsilon', \
'test_group_normalization_example', \
'test_hammingwindow_expanded', \
'test_hammingwindow_symmetric_expanded', \
'test_hannwindow', \
'test_hannwindow_expanded', \
'test_hannwindow_symmetric_expanded', \
'test_hardmax_axis_0', \
'test_reduce_min_default_axes_keepdims_example', \
'test_reduce_min_default_axes_keepdims_random', \
'test_reduce_prod_default_axes_keepdims_example', \
'test_reflect_pad', \
'test_relu', \
'test_resize_downsample_scales_cubic', \
'test_resize_downsample_scales_cubic_align_corners', \
'test_resize_downsample_scales_cubic_A_n0p5_exclude_outside', \
'test_resize_downsample_scales_cubic_antialias', \
'test_resize_downsample_scales_linear', \
'test_resize_downsample_scales_linear_align_corners', \
'test_resize_downsample_scales_linear_antialias', \
'test_resize_downsample_scales_nearest', \
'test_resize_downsample_sizes_cubic', \
'test_resize_downsample_sizes_cubic_antialias', \
'test_resize_downsample_sizes_linear_antialias', \
'test_resize_downsample_sizes_linear_pytorch_half_pixel', \
'test_resize_downsample_sizes_nearest', \
'test_resize_downsample_sizes_nearest_not_larger', \
'test_resize_downsample_sizes_nearest_not_smaller', \
'test_resize_tf_crop_and_resize', \
'test_resize_tf_crop_and_resize_axes_2_3', \
'test_resize_tf_crop_and_resize_axes_3_2', \
'test_resize_upsample_scales_cubic', \
'test_resize_upsample_scales_cubic_align_corners', \
'test_resize_upsample_scales_cubic_A_n0p5_exclude_outside', \
'test_resize_upsample_scales_cubic_asymmetric', \
'test_resize_upsample_scales_linear', \
'test_resize_upsample_scales_linear_align_corners', \
'test_resize_upsample_scales_nearest', \
'test_resize_upsample_scales_nearest_axes_2_3', \
'test_resize_upsample_scales_nearest_axes_3_2', \
'test_resize_upsample_sizes_cubic', \
'test_resize_upsample_sizes_nearest', \
'test_resize_upsample_sizes_nearest_axes_2_3', \
'test_resize_upsample_sizes_nearest_axes_3_2', \
'test_resize_upsample_sizes_nearest_ceil_half_pixel', \
'test_resize_upsample_sizes_nearest_floor_align_corners', \
'test_resize_upsample_sizes_nearest_not_larger', \
'test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric', \
'test_round', \
'test_scatter_elements_with_duplicate_indices', \
'test_scatter_elements_with_reduction_max', \
'test_scatter_elements_with_reduction_min', \
'test_scatternd', \
'test_scatternd_add', \
'test_scatternd_max', \
'test_scatternd_min', \
'test_scatternd_multiply', \
'test_sce_mean', \
'test_sce_mean_3d', \
'test_sce_mean_3d_expanded', \
'test_sce_mean_3d_log_prob', \
'test_sce_mean_3d_log_prob_expanded', \
'test_sce_mean_expanded', \
'test_sce_mean_log_prob', \
'test_sce_mean_log_prob_expanded', \
'test_sce_mean_no_weight_ii', \
'test_sce_mean_no_weight_ii_3d', \
'test_sce_mean_no_weight_ii_3d_expanded', \
'test_sce_mean_no_weight_ii_3d_log_prob', \
'test_sce_mean_no_weight_ii_3d_log_prob_expanded', \
'test_sce_mean_no_weight_ii_4d', \
'test_sce_mean_no_weight_ii_4d_expanded', \
'test_sce_mean_no_weight_ii_4d_log_prob', \
'test_sce_mean_no_weight_ii_4d_log_prob_expanded', \
'test_sce_mean_no_weight_ii_expanded', \
'test_sce_mean_no_weight_ii_log_prob', \
'test_sce_mean_no_weight_ii_log_prob_expanded', \
'test_sce_mean_weight', \
'test_sce_mean_weight_expanded', \
'test_sce_mean_weight_ii', \
'test_sce_mean_weight_ii_3d', \
'test_sce_mean_weight_ii_3d_expanded', \
'test_sce_mean_weight_ii_3d_log_prob', \
'test_sce_mean_weight_ii_3d_log_prob_expanded', \
'test_sce_mean_weight_ii_4d', \
'test_sce_mean_weight_ii_4d_expanded', \
'test_sce_mean_weight_ii_4d_log_prob', \
'test_sce_mean_weight_ii_4d_log_prob_expanded', \
'test_sce_mean_weight_ii_expanded', \
'test_sce_mean_weight_ii_log_prob', \
'test_sce_mean_weight_ii_log_prob_expanded', \
'test_sce_mean_weight_log_prob', \
'test_sce_mean_weight_log_prob_expanded', \
'test_sce_NCd1d2d3d4d5_mean_weight', \
'test_sce_NCd1d2d3d4d5_mean_weight_expanded', \
'test_sce_NCd1d2d3d4d5_mean_weight_log_prob', \
'test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded', \
'test_sce_NCd1d2d3d4d5_none_no_weight', \
'test_sce_NCd1d2d3d4d5_none_no_weight_expanded', \
'test_sce_NCd1d2d3d4d5_none_no_weight_log_prob', \
'test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded', \
'test_sce_NCd1d2d3_none_no_weight_negative_ii', \
'test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded', \
'test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob', \
'test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded', \
'test_sce_NCd1d2d3_sum_weight_high_ii', \
'test_sce_NCd1d2d3_sum_weight_high_ii_expanded', \
'test_sce_NCd1d2d3_sum_weight_high_ii_log_prob', \
'test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded', \
'test_sce_NCd1_mean_weight_negative_ii', \
'test_sce_NCd1_mean_weight_negative_ii_expanded', \
'test_sce_NCd1_mean_weight_negative_ii_log_prob', \
'test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded', \
'test_sce_none', \
'test_sce_none_expanded', \
'test_sce_none_log_prob', \
'test_sce_none_log_prob_expanded', \
'test_sce_none_weights', \
'test_sce_none_weights_expanded', \
'test_sce_none_weights_log_prob', \
'test_sce_none_weights_log_prob_expanded', \
'test_sce_sum', \
'test_sce_sum_expanded', \
'test_sce_sum_log_prob', \
'test_sce_sum_log_prob_expanded', \
'test_selu', \
'test_selu_default', \
'test_selu_example', \
'test_shape', \
'test_shape_clip_end', \
'test_shape_clip_start', \
'test_shape_end_negative_1', \
'test_shape_example', \
'test_shape_start_1_end_2', \
'test_shape_start_1_end_negative_1', \
'test_shape_start_negative_1', \
'test_sigmoid', \
'test_sin', \
'test_sin_example', \
'test_sinh', \
'test_size_example', \
'test_slice', \
'test_slice_end_out_of_bounds', \
'test_slice_neg', \
'test_slice_negative_axes', \
'test_slice_neg_steps', \
'test_slice_start_out_of_bounds', \
'test_softmax_axis_0', \
'test_softmax_axis_0_expanded', \
'test_softmax_axis_1', \
'test_softmax_axis_1_expanded', \
'test_softmax_axis_2_expanded', \
'test_softmax_default_axis_expanded_ver18', \
'test_softmax_example', \
'test_softmax_example_expanded_ver18', \
'test_softmax_large_number_expanded', \
'test_softmax_negative_axis', \
'test_softmax_negative_axis_expanded_ver18', \
'test_softplus', \
'test_softplus_example', \
'test_softplus_example_expanded', \
'test_softplus_expanded', \
'test_softsign', \
'test_spacetodepth_example', \
'test_split_equal_parts_1d_opset13', \
'test_split_equal_parts_default_axis_opset13', \
'test_split_variable_parts_1d_opset13', \
'test_split_variable_parts_2d_opset13', \
'test_split_variable_parts_default_axis_opset13', \
'test_split_zero_size_splits_opset13', \
'test_sqrt', \
'test_sqrt_example', \
'test_strnormalizer_export_monday_casesensintive_lower', \
'test_strnormalizer_export_monday_casesensintive_upper', \
'test_strnormalizer_export_monday_empty_output', \
'test_strnormalizer_export_monday_insensintive_upper_twodim', \
'test_strnormalizer_nostopwords_nochangecase', \
'test_sub_uint8', \
'test_sum_example', \
'test_sum_one_input', \
'test_tan', \
'test_tanh', \
'test_tanh_example', \
'test_tfidfvectorizer_tf_batch_onlybigrams_skip0', \
'test_tfidfvectorizer_tf_only_bigrams_skip0', \
'test_tfidfvectorizer_tf_uniandbigrams_skip5', \
'test_thresholdedrelu', \
'test_thresholdedrelu_default', \
'test_thresholdedrelu_example', \
'test_transpose_all_permutations_0', \
'test_transpose_all_permutations_2', \
'test_transpose_all_permutations_3', \
'test_transpose_all_permutations_4', \
'test_transpose_all_permutations_5', \
'test_transpose_default', \
'test_tril', \
'test_tril_square', \
'test_triu_square', \
'test_unique_not_sorted_without_axis', \
'test_unique_sorted_with_axis', \
'test_unique_sorted_with_axis_3d', \
'test_unique_sorted_with_negative_axis', \
'test_unique_sorted_without_axis', \
'test_xor4d']

other_fails = [
# max_nmse too high
'test_add_bcast',\
'test_averagepool_2d_precomputed_same_upper',\
'test_col2im_pads',\

# output shape mismatch
'test_argmax_keepdims_example_',\
'test_argmax_keepdims_example_select_last_index_',\
'test_argmax_keepdims_random_',\
'test_argmax_keepdims_random_select_last_index_',\
'test_argmax_no_keepdims_example_',\
'test_argmax_no_keepdims_example_select_last_index_',\
'test_argmax_no_keepdims_random_',\
'test_argmax_no_keepdims_random_select_last_index_',\
'test_averagepool_2d_ceil_',\
'test_averagepool_2d_precomputed_pads_',\
'test_averagepool_2d_precomputed_pads_count_include_pad_',\
'test_bernoulli_seed_',\
'test_bernoulli_seed_expanded_',\
'test_clip_default_inbounds_expanded_',\

# crashes without message
'test_cast_FLOAT16_to_DOUBLE', \

# hangs
'test_batchnorm_example',\
'test_batchnorm_example_training_mode',\
'test_quantizelinear_axis',\
'test_dequantizelinear',\
'test_reshape_reordered_last_dims',\
'test_reshape_zero_and_negative_dim',\
'test_reshape_zero_dim'
'test_gemm_transposeB',\
'test_gemm_default_zero_bias',\
'test_dequantizelinear_axis',\
'test_prelu_example', \

# Calling ialg.algAlloc failed with status = -1120
'test_div', \
'test_div_example', \
'test_div_uint8', \
]


known_failing_test_cases = cpu_failing_testcases + other_fails + fatal_python_error 

node_tests_to_run = os.listdir(node_tests_root)
# Uncomment below line to filter out known failures
# node_tests_to_run = [node_test for node_test in os.listdir(node_tests_root) if node_test not in known_failing_test_cases]

simple_tests_to_run = os.listdir(simple_tests_root)
pc_tests_to_run     = os.listdir(pc_tests_root)
po_tests_to_run     = os.listdir(po_tests_root)


# Test onnx node test
@pytest.mark.parametrize("test_name", node_tests_to_run)
def test_onnx_backend_node(tidl_offload : bool, run_infer : bool, node_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    Note command-line options --disable-offload (disable offload to TIDL) and --run-infer (default is import, this enables inference after import)
    Example of running a single test: pytest test_onnx_backend_node[test_conv_with_strides_no_padding] --disable-offload
    '''
    test_dir = os.path.join(node_tests_root_fixture, test_name)
    test_onnx_backend(tidl_offload = tidl_offload, 
                      run_infer    = run_infer, 
                      test_dir     = test_dir)


# Test onnx simple test
@pytest.mark.parametrize("test_name", simple_tests_to_run)
def test_onnx_backend_simple(tidl_offload : bool, run_infer : bool, simple_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    Note command-line options --disable-offload (disable offload to TIDL) and --run-infer (default is import, this enables inference after import)
    Example of running a single test: pytest test_onnx_backend_simple[test_expand_shape_model1] --disable-offload
    '''
    test_dir = os.path.join(simple_tests_root_fixture, test_name)
    test_onnx_backend(tidl_offload = tidl_offload, 
                      run_infer    = run_infer, 
                      test_dir     = test_dir)
    

# Test onnx pytorch-converted test
@pytest.mark.parametrize("test_name", pc_tests_to_run)
def test_onnx_backend_pc(tidl_offload : bool, run_infer : bool, pc_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    Note command-line options --disable-offload (disable offload to TIDL) and --run-infer (default is import, this enables inference after import)
    Example of running a single test: pytest test_onnx_backend_pc[test_AvgPool1d] --disable-offload
    '''
    test_dir = os.path.join(pc_tests_root_fixture, test_name)
    test_onnx_backend(tidl_offload = tidl_offload, 
                      run_infer    = run_infer, 
                      test_dir     = test_dir)

# Test onnx pytorch-operator test
@pytest.mark.parametrize("test_name", po_tests_to_run)
def test_onnx_backend_po(tidl_offload : bool, run_infer : bool, po_tests_root_fixture : str, test_name : str):
    '''
    Pytest for onnx backend node tests using the edgeai-benchmark framework
    Note command-line options --disable-offload (disable offload to TIDL) and --run-infer (default is import, this enables inference after import)
    Example of running a single test: pytest test_onnx_backend_po[test_operator_add_broadcast] --disable-offload
    '''
    test_dir = os.path.join(po_tests_root_fixture, test_name)
    test_onnx_backend(tidl_offload = tidl_offload, 
                      run_infer    = run_infer, 
                      test_dir     = test_dir)


# Utility function to perform onnx backend test
def test_onnx_backend(tidl_offload : bool, run_infer : bool, test_dir : str):
  
    # Check environment is set up correctly
    assert os.path.exists(test_dir), f"test path {test_dir} doesn't exist"
    assert os.environ.get('TIDL_RT_AVX_REF') is not None, "Make sure to source run_set_env.sh"
    assert os.path.exists(os.environ['TIDL_TOOLS_PATH'])
    assert os.path.exists(os.environ['ARM64_GCC_PATH'])

    # Declare config object
    settings = config_settings.ConfigSettings('./onnx_backend.yaml', target_device = "TDA4VM", tidl_offload=tidl_offload)

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
        logger.debug(results_list[0]['result'])
        
        # TODO: Choose better threshold. 0.5 chosen for now to reveal worst offenders
        assert results_list[0]['result']['max_nmse']<0.5, f" max_nmse of {results_list[0]['result']['max_nmse']} is too high"
    
    # Otherwise run import
    else:
        logger.debug("Importing")
        interfaces.run_accuracy(settings, work_dir, pipeline_configs)





