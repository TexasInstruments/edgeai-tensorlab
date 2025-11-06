# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from .custom import add_custom_operator, add_custom_node_2_torch_graph, custom_add_2_torch_graph
from .single_ipop_layers import *
from .multi_ip_single_op_layers import *
from .activation_func import *
from .argop import *
from .attention import *
from .pool import *
from .normalization import *
from .bitshift import *
from .cast import *
from .concat import*
from .compress import *
from .constant import *
from .conv import *
from .cumsum import *
from .quantization import *
from .dropout import *
from .gather import *
from .grid_sample import *
from .reduce_ops import *
from .reshape import *
from .resize import *
from .slice import *
from .squeeze import *
from .topk import *
from .transpose import *
from .unsqueeze import *
from .gemm import *
from .einsum import *
from .window import *
from .conditional_flow import *
from .bool_ops import *
from .matmul import *
from .stats_ops import *
from .multinomial import *
from .loss import *
from .misc import *
from .string_ops import *
from .nms import *
from .pad import *
from .rnn import *
from .random import *
from .stft import *
from .scatter import *
from .space_to_depth import *
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
    'MatMul': add_matmul_2_torch_graph,
    'MatMulInteger': add_matmul_int_2_torch_graph,
    'Max': add_stat_op_2_torch_graph,
    'MaxPool': add_max_pool_2_torch_graph,
    'MaxRoiPool': add_max_roi_pool_2_torch_graph,
    'MaxUnpool': add_max_unpool_2_torch_graph,
    'Mean': add_stat_op_2_torch_graph,
    'MeanVarianceNormalization':add_mean_variance_norm_2_torch_graph,
    'MelWeightMatrix': add_mel_weight_matrix_2_torch_graph,
    'Min': add_stat_op_2_torch_graph,
    'Mish': add_mish_2_torch_graph,
    'Mod': add_mod_2_torch_graph,
    'Mul': add_node_2_torch_graph_multi_ip_1op,
    'Multinomial': add_multinomial_2_torch_graph,
    'Neg': add_node_2_torch_graph_1ip_1op,
    'NegativeLogLikelihoodLoss': add_negative_log_likelihood_loss_2_torch_graph,
    'NonMaxSuppression': add_non_max_suppression_2_torch_graph,
    'NonZero': add_non_zero_2_torch_graph,
    'Not': add_node_2_torch_graph_1ip_1op,
    'OneHot': add_one_hot_2_torch_graph,
    'Optional': add_optional_2_torch_graph,
    'OptionalGetElement': add_optional_get_element_2_torch_graph,
    'OptionalHasElement': add_optional_has_element_2_torch_graph,
    'Or': add_node_2_torch_graph_multi_ip_1op,
    'PRelu': add_prelu_2_torch_graph,
    'Pad': add_pad_2_torch_graph,
    'Pow': add_node_2_torch_graph_multi_ip_1op,
    'QLinearConv': add_q_linear_conv_2_torch_graph,
    'QLinearMatMul': add_q_linear_matmul_2_torch_graph,
    'QuantizeLinear': add_quantize_linear_2_torch_graph,
    'RNN': add_rnn_2_torch_graph,
    'RandomNormal': add_random_normal_2_torch_graph,
    'RandomNormalLike': add_random_normal_like_2_torch_graph,
    'RandomUniform': add_random_uniform_2_torch_graph,
    'RandomUniformLike': add_random_uniform_like_2_torch_graph,
    'Range': add_range_2_torch_graph,
    'Reciprocal': add_node_2_torch_graph_1ip_1op,
    'ReduceL1': add_reduce_l1_2_torch_graph,
    'ReduceL2': add_reduce_l2_2_torch_graph,
    'ReduceLogSum': add_reduce_log_sum_2_torch_graph,
    'ReduceLogSumExp': add_reduce_log_sum_exp_2_torch_graph,
    'ReduceMax': add_reduce_max_2_torch_graph,
    'ReduceMean': add_reduce_mean_2_torch_graph,
    'ReduceMin': add_reduce_min_2_torch_graph,
    'ReduceProd': add_reduce_prod_2_torch_graph,
    'ReduceSum': add_reduce_sum_2_torch_graph,
    'ReduceSumSquare': add_reduce_sum_square_2_torch_graph,
    'RegexFullMatch': add_regex_full_match_2_torch_graph,
    'Relu': add_relu_2_torch_graph,
    'Reshape': add_reshape_2_torch_graph,
    'Resize': add_resize_2_torch_graph,
    'ReverseSequence': add_reverse_sequence_2_torch_graph,
    'RoiAlign': add_roi_align_2_torch_graph,
    'RotaryEmbedding': add_rotary_embedding_2_torch_graph,
    'Round': add_node_2_torch_graph_1ip_1op,
    'STFT': add_stft_2_torch_graph,
    'Scan': add_scan_2_torch_graph,
    'Scatter': add_scatter_2_torch_graph,
    'ScatterElements': add_scatter_elements_2_torch_graph,
    'ScatterND': add_scatter_nd_2_torch_graph,
    'Selu': add_selu_2_torch_graph,
    'SequenceAt': add_sequence_at_2_torch_graph,
    'SequenceConstruct': add_sequence_construct_2_torch_graph,
    'SequenceEmpty': add_sequence_empty_2_torch_graph,
    'SequenceErase': add_sequence_erase_2_torch_graph,
    'SequenceInsert': add_sequence_insert_2_torch_graph,
    'SequenceLength': add_sequence_length_2_torch_graph,
    'SequenceMap': add_sequence_map_2_torch_graph,
    'Shape': add_shape_2_torch_graph,
    'Shrink': add_shrink_2_torch_graph,
    'Sigmoid': add_sigmoid_2_torch_graph,
    'Sign': add_node_2_torch_graph_1ip_1op,
    'Sin': add_node_2_torch_graph_1ip_1op,
    'Sinh': add_node_2_torch_graph_1ip_1op,
    'Slice': add_slice_2_torch_graph,
    'Softmax': add_softmax_2_torch_graph,
    'SoftmaxCrossEntropyLoss':add_softmax_cross_entropy_loss_2_torch_graph,
    'Softplus':add_softplus_2_torch_graph,
    'Softsign': add_softsign_2_torch_graph,
    'SpaceToDepth': add_space2depth_2_torch_graph,
    'Split': add_split_2_torch_graph,
    'SplitToSequence': add_split_to_sequence_2_torch_graph,
    'Sqrt': add_node_2_torch_graph_1ip_1op,
    'Squeeze': add_squeeze_2_torch_graph,
    'StringCocat': add_string_concat_2_torch_graph,
    'StringNormalizer': add_string_normalizer_2_torch_graph,
    'StringSplit': add_string_split_2_torch_graph,
    'Sub': add_node_2_torch_graph_multi_ip_1op,
    'Sum': add_sum_2_torch_graph,
    'Swish': add_swish_2_torch_graph,
    'Tan': add_node_2_torch_graph_1ip_1op,
    'Tanh': add_node_2_torch_graph_1ip_1op,
    'TensorScatterUpdate': add_tensor_scatter_2_torch_graph,
    'TfLdfVectorizer':add_tfldf_vectorizer_2_torch_graph,
    'ThresholdedRelu': add_thresholded_relu_2_torch_graph,
    'Tile': add_tile_2_torch_graph,
    'TopK': add_topk_2_torch_graph,
    'Transpose': add_transpose_2_torch_graph,
    'Trilu': add_trilu_2_torch_graph,
    'Unique': add_unique_2_torch_graph,
    'Unsqueeze': add_unsqueeze_2_torch_graph,
    'Upsample': add_upsample_2_torch_graph,
    'Where': add_where_2_torch_graph,
    'Xor': add_node_2_torch_graph_multi_ip_1op,
}

__all__ = ['basic_ops_2_func_dict', 'add_custom_operator', 'add_custom_node_2_torch_graph', 'custom_add_2_torch_graph', 'utils']
