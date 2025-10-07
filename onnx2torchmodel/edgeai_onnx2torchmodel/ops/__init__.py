from .single_ipop_layers import add_node_2_torch_graph_1ip_1op
from .multi_ip_single_op_layers import add_node_2_torch_graph_multi_ip_1op
from .affine_grid import add_affine_grid_2_torch_graph
from .custom import add_custom_operator, add_custom_node_2_torch_graph, custom_add_2_torch_graph
from .argop import add_argop_2_torch_graph
from .attention import add_attention_2_torch_graph
from .pool import add_avg_pool_2_torch_graph, add_max_pool_2_torch_graph, add_global_avg_pool_2_torch_graph, add_global_max_pool_2_torch_graph, add_global_lp_pool_2_torch_graph, add_lp_pool_2_torch_graph, add_max_roi_pool_2_torch_graph, add_max_unpool_2_torch_graph
from .normalization import add_batchnorm_2_torch_graph, add_instance_norm_2_torch_graph, add_layer_norm_2_torch_graph, add_group_norm_2_torch_graph, add_lp_norm_2_torch_graph, add_mean_variance_norm_2_torch_graph
from .bernouli import add_bernouli_2_torch_graph
from .bitshift import add_bitshift_2_torch_graph
from .blackman_window import add_blackman_window_2_torch_graph
from .cast import add_cast_2_torch_graph, add_cast_like_2_torch_graph
from .activation_func import add_celu_2_torch_graph, add_elu_2_torch_graph, add_hardsigmoid_2_torch_graph, add_hardswish_2_torch_graph, add_hardmax_2_torch_graph, add_log_softmax_2_torch_graph, add_leakyrelu_2_torch_graph
from .center_crop_pad import add_center_crop_pad_2_torch_graph
from .clip import add_clip_2_torch_graph
from .col2im import add_col2im_2_torch_graph
from .concat import add_concat_2_torch_graph, add_concat_from_sequence_2_torch_graph, add_expand_2_torch_graph
from .compress import add_compress_2_torch_graph
from .constant import add_constant_2_torch_graph, add_constant_of_shape_2_torch_graph, add_eye_like_2_torch_graph
from .conv import add_conv_2_torch_graph, add_conv_integer_2_torch_graph, add_conv_transpose_2_torch_graph, add_deform_conv_2_torch_graph
from .cumsum import add_cumsum_2_torch_graph
from .dft import add_dft_2_torch_graph
from .quantization import add_dequantize_linear_2_torch_graph, add_dynamic_quantize_linear_2_torch_graph
from .dropout import add_dropout_2_torch_graph
from .gather import add_gather_2_torch_graph, add_gather_elements_2_torch_graph, add_gather_nd_2_torch_graph
from .grid_sample import add_grid_sample_2_torch_graph
from .reduce_ops import add_reduce_max_2_torch_graph
from .reshape import add_reshape_2_torch_graph, add_flatten_2_torch_graph
from .resize import add_resize_2_torch_graph
from .slice import add_slice_2_torch_graph
from .softmax import add_softmax_2_torch_graph
from .squeeze import add_squeeze_2_torch_graph
from .topk import add_topk_2_torch_graph
from .transpose import add_transpose_2_torch_graph
from .unsqueeze import add_unsqueeze_2_torch_graph
from .gemm import add_gemm_2_torch_graph
from .einsum import add_einsum_2_torch_graph
from .gru import add_gru_2_torch_graph
from .window import add_hann_window_2_torch_graph, add_hamming_window_2_torch_graph
from .conditional_flow import add_if_2_torch_graph, add_loop_2_torch_graph
from .image_decoder import add_image_decoder_2_torch_graph
from .bool_ops import add_is_inf_2_torch_graph
from .lrn import add_lrn_2_torch_graph
from .lstm import add_lstm_2_torch_graph
from .matmul_int import add_matmul_int_2_torch_graph
from .stats_ops import add_stat_op_2_torch_graph
from . import utils

basic_ops_2_func_dict = {
    'Abs': add_node_2_torch_graph_1ip_1op,
    'Acos': add_node_2_torch_graph_1ip_1op,
    'Acosh': add_node_2_torch_graph_1ip_1op,
    'Add': add_node_2_torch_graph_multi_ip_1op,
    'AffineGrid': add_affine_grid_2_torch_graph,
    'And': add_node_2_torch_graph_multi_ip_1op,
    'ArgMax': add_argop_2_torch_graph,
    'ArgMin': add_argop_2_torch_graph,
    'Asin': add_node_2_torch_graph_1ip_1op,
    'Asinh': add_node_2_torch_graph_1ip_1op,
    'Atan': add_node_2_torch_graph_1ip_1op,
    'Atanh': add_node_2_torch_graph_1ip_1op,
    'Attention': add_attention_2_torch_graph,
    'AveragePool': add_avg_pool_2_torch_graph,
    'BatchNormalization': add_batchnorm_2_torch_graph,
    'Bernoulli': add_bernouli_2_torch_graph,
    'BitShift': add_bitshift_2_torch_graph,
    'BitwiseAnd': add_node_2_torch_graph_multi_ip_1op,
    'BitwiseNot': add_node_2_torch_graph_1ip_1op,
    'BitwiseOr': add_node_2_torch_graph_multi_ip_1op,
    'BitwiseXor': add_node_2_torch_graph_multi_ip_1op,
    'BlackmanWindow': add_blackman_window_2_torch_graph,
    'Cast': add_cast_2_torch_graph,
    'CastLike': add_cast_like_2_torch_graph,
    'Ceil': add_node_2_torch_graph_1ip_1op,
    'Celu': add_celu_2_torch_graph,
    'CenterCropPad': add_center_crop_pad_2_torch_graph,
    'Clip': add_clip_2_torch_graph,
    'Col2Im': add_col2im_2_torch_graph,
    'Compress': add_compress_2_torch_graph,
    'Concat': add_concat_2_torch_graph,
    'ConcatFromSequence': add_concat_from_sequence_2_torch_graph,
    'Constant': add_constant_2_torch_graph,
    'ConstantOfShape': add_constant_of_shape_2_torch_graph,
    'Conv': add_conv_2_torch_graph,
    'ConvInteger': add_conv_integer_2_torch_graph,
    'ConvTranspose': add_conv_transpose_2_torch_graph,
    'Cos': add_node_2_torch_graph_1ip_1op,
    'Cosh': add_node_2_torch_graph_1ip_1op,
    'CumSum': add_cumsum_2_torch_graph,
    'DFT': add_dft_2_torch_graph,
    'DeformConv': add_deform_conv_2_torch_graph,
    'DequantizeLinear': add_dequantize_linear_2_torch_graph,
    'Det': add_node_2_torch_graph_1ip_1op,
    'Div': add_node_2_torch_graph_multi_ip_1op,
    'Dropout': add_dropout_2_torch_graph,
    'DynamicQuantizeLinear': add_dynamic_quantize_linear_2_torch_graph,
    'Einsum': add_einsum_2_torch_graph,
    'Elu': add_elu_2_torch_graph,
    'Equal': add_node_2_torch_graph_multi_ip_1op,
    'Erf': add_node_2_torch_graph_1ip_1op,
    'Exp': add_node_2_torch_graph_1ip_1op,
    'Expand': add_expand_2_torch_graph,
    'EyeLike': add_eye_like_2_torch_graph,
    'Flatten': add_flatten_2_torch_graph,
    'Floor': add_node_2_torch_graph_1ip_1op,
    'GRU': add_gru_2_torch_graph,
    'Gather': add_gather_2_torch_graph,
    'GatherElements': add_gather_elements_2_torch_graph,
    'GatherND': add_gather_nd_2_torch_graph,
    'Gemm': add_gemm_2_torch_graph,
    'GlobalAveragePool': add_global_avg_pool_2_torch_graph,
    'GlobalLpPool': add_global_lp_pool_2_torch_graph,
    'GlobalMaxPool': add_global_max_pool_2_torch_graph,
    'Greater': add_node_2_torch_graph_multi_ip_1op,
    'GreaterOrEqual': add_node_2_torch_graph_multi_ip_1op,
    'GridSample': add_grid_sample_2_torch_graph,
    'GroupNormalization': add_group_norm_2_torch_graph,
    'HammingWindow': add_hamming_window_2_torch_graph,
    'HannWindow': add_hann_window_2_torch_graph,
    'HardSigmoid': add_hardsigmoid_2_torch_graph,
    'HardSwish': add_hardswish_2_torch_graph,
    'Hardmax': add_hardmax_2_torch_graph,
    'If': add_if_2_torch_graph,
    'ImageDecoder': add_image_decoder_2_torch_graph,
    'InstanceNormalization': add_instance_norm_2_torch_graph,
    'IsInf': add_is_inf_2_torch_graph,
    'IsNaN': add_node_2_torch_graph_1ip_1op,
    'LRN': add_lrn_2_torch_graph,
    'LSTM': add_lstm_2_torch_graph,
    'LayerNormalization': add_layer_norm_2_torch_graph,
    'LeakyRelu': add_leakyrelu_2_torch_graph,
    'Less': add_node_2_torch_graph_multi_ip_1op,
    'LessOrEqual': add_node_2_torch_graph_multi_ip_1op,
    'Log': add_node_2_torch_graph_1ip_1op,
    'LogSoftmax': add_log_softmax_2_torch_graph,
    'Loop': add_loop_2_torch_graph,
    'LpNormalization': add_lp_norm_2_torch_graph,
    'LpPool': add_lp_pool_2_torch_graph,
    'MatMul': add_node_2_torch_graph_multi_ip_1op,
    'MatMulInteger': add_matmul_int_2_torch_graph,
    'Max': add_stat_op_2_torch_graph,
    'MaxPool': add_max_pool_2_torch_graph,
    'MaxRoiPool': add_max_roi_pool_2_torch_graph,
    'MaxUnpool': add_max_unpool_2_torch_graph,
    'Mean': add_stat_op_2_torch_graph,
    'MeanVarianceNormalization':add_mean_variance_norm_2_torch_graph,
    'Min': add_stat_op_2_torch_graph,
    'Mul': add_node_2_torch_graph_multi_ip_1op,
    'Not': add_node_2_torch_graph_1ip_1op,
    'Or': add_node_2_torch_graph_multi_ip_1op,
    'ReduceMax': add_reduce_max_2_torch_graph,
    'Relu': add_node_2_torch_graph_1ip_1op,
    'Reshape': add_reshape_2_torch_graph,
    'Resize': add_resize_2_torch_graph,
    'Sigmoid': add_node_2_torch_graph_1ip_1op,
    'Sin': add_node_2_torch_graph_1ip_1op,
    'Sinh': add_node_2_torch_graph_1ip_1op,
    'Slice': add_slice_2_torch_graph,
    'Softmax': add_softmax_2_torch_graph,
    'Squeeze': add_squeeze_2_torch_graph,
    'Sub': add_node_2_torch_graph_multi_ip_1op,
    'Tan': add_node_2_torch_graph_1ip_1op,
    'Tanh': add_node_2_torch_graph_1ip_1op,
    'TopK': add_topk_2_torch_graph,
    'Transpose': add_transpose_2_torch_graph,
    'Unsqueeze': add_unsqueeze_2_torch_graph,
    'Xor': add_node_2_torch_graph_multi_ip_1op,
}

__all__ = ['basic_ops_2_func_dict', 'add_custom_operator', 'add_custom_node_2_torch_graph', 'custom_add_2_torch_graph', 'utils']
