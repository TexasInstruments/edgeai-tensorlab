from .single_ipop_layers import add_node_2_torch_graph_1ip_1op
from .multi_ip_single_op_layers import add_node_2_torch_graph_multi_ip_1op
from .affine_grid import add_affine_grid_2_torch_graph
from .custom import add_custom_operator, add_custom_node_2_torch_graph, custom_add_2_torch_graph
from .argop import add_argop_2_torch_graph
from .attention import add_attention_2_torch_graph
from .pool import add_avg_pool_2_torch_graph, add_max_pool_2_torch_graph
from .normalization import add_batchnorm_2_torch_graph, add_instance_norm_2_torch_graph, add_layer_norm_2_torch_graph
from .bernouli import add_bernouli_2_torch_graph
from .bitshift import add_bitshift_2_torch_graph
from .blackman_window import add_blackman_window_2_torch_graph
from .cast import add_cast_2_torch_graph, add_cast_like_2_torch_graph
from .activation_func import add_celu_2_torch_graph
from .center_crop_pad import add_center_crop_pad_2_torch_graph
from .clip import add_clip_2_torch_graph
from .col2im import add_col2im_2_torch_graph
from .concat import add_concat_2_torch_graph, add_concat_from_sequence_2_torch_graph
from .compress import add_compress_2_torch_graph
from .constant import add_constant_2_torch_graph, add_constant_of_shape_2_torch_graph
from .conv import add_conv_2_torch_graph, add_conv_integer_2_torch_graph, add_conv_transpose_2_torch_graph, add_deform_conv_2_torch_graph
from .cumsum import add_cumsum_2_torch_graph
from .dft import add_dft_2_torch_graph
from .quantization import add_dequantize_linear_2_torch_graph
from .dropout import add_dropout_2_torch_graph
from .gather import add_gather_2_torch_graph
from .grid_sample import add_grid_sample_2_torch_graph
from .reduce_ops import add_reduce_max_2_torch_graph
from .reshape import add_reshape_2_torch_graph
from .resize import add_resize_2_torch_graph
from .slice import add_slice_2_torch_graph
from .softmax import add_softmax_2_torch_graph
from .squeeze import add_squeeze_2_torch_graph
from .topk import add_topk_2_torch_graph
from .transpose import add_transpose_2_torch_graph
from .unsqueeze import add_unsqueeze_2_torch_graph
from . import utils

basic_ops_2_func_dict = {
    'Add': add_node_2_torch_graph_multi_ip_1op,
    'Mul': add_node_2_torch_graph_multi_ip_1op,
    'Sub': add_node_2_torch_graph_multi_ip_1op,
    'Div': add_node_2_torch_graph_multi_ip_1op,
    'And': add_node_2_torch_graph_multi_ip_1op,
    'Abs': add_node_2_torch_graph_1ip_1op,
    'Acos': add_node_2_torch_graph_1ip_1op,
    'Acosh': add_node_2_torch_graph_1ip_1op,
    'Asin': add_node_2_torch_graph_1ip_1op,
    'Asinh': add_node_2_torch_graph_1ip_1op,
    'Atan': add_node_2_torch_graph_1ip_1op,
    'Atanh': add_node_2_torch_graph_1ip_1op,
    'ArgMax': add_argop_2_torch_graph,
    'ArgMin': add_argop_2_torch_graph,
    'AffineGrid': add_affine_grid_2_torch_graph,
    'Attention': add_attention_2_torch_graph,
    'AveragePool': add_avg_pool_2_torch_graph,
    'BatchNormalization': add_batchnorm_2_torch_graph,
    'Bernoulli': add_bernouli_2_torch_graph,
    'BitShift': add_bitshift_2_torch_graph,
    'BitwiseNot': add_node_2_torch_graph_1ip_1op,
    'BitwiseAnd': add_node_2_torch_graph_multi_ip_1op,
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
    'Concat': add_concat_2_torch_graph,
    'ConcatFromSequence': add_concat_from_sequence_2_torch_graph,
    'Compress': add_compress_2_torch_graph,
    'Constant': add_constant_2_torch_graph,
    'ConstantOfShape': add_constant_of_shape_2_torch_graph,
    'Conv' : add_conv_2_torch_graph,
    'ConvInteger' : add_conv_integer_2_torch_graph,
    'ConvTranspose' : add_conv_transpose_2_torch_graph,
    'Cos': add_node_2_torch_graph_1ip_1op,
    'Cosh': add_node_2_torch_graph_1ip_1op,
    'CumSum': add_cumsum_2_torch_graph,
    'DFT': add_dft_2_torch_graph,
    'DeformConv': add_deform_conv_2_torch_graph,
    'DequantizeLinear': add_dequantize_linear_2_torch_graph,
    'Det': add_node_2_torch_graph_1ip_1op,
    'Dropout': add_dropout_2_torch_graph,
    'Erf': add_node_2_torch_graph_1ip_1op,
    'Gather': add_gather_2_torch_graph,
    'GridSample': add_grid_sample_2_torch_graph,
    'InstanceNormalization': add_instance_norm_2_torch_graph,
    'LayerNormalization': add_layer_norm_2_torch_graph,
    'Log': add_node_2_torch_graph_1ip_1op,
    'MatMul': add_node_2_torch_graph_multi_ip_1op,
    'MaxPool': add_max_pool_2_torch_graph,
    'Relu': add_node_2_torch_graph_1ip_1op,
    'ReduceMax': add_reduce_max_2_torch_graph,
    'Reshape': add_reshape_2_torch_graph,
    'Resize': add_resize_2_torch_graph,
    'Sigmoid': add_node_2_torch_graph_1ip_1op,
    'Slice': add_slice_2_torch_graph,
    'Softmax': add_softmax_2_torch_graph,
    'Squeeze': add_squeeze_2_torch_graph,
    'Sin': add_node_2_torch_graph_1ip_1op,
    'Sinh': add_node_2_torch_graph_1ip_1op,
    'Tan': add_node_2_torch_graph_1ip_1op,
    'Tanh': add_node_2_torch_graph_1ip_1op,
    'TopK': add_topk_2_torch_graph,
    'Transpose': add_transpose_2_torch_graph,
    'Unsqueeze': add_unsqueeze_2_torch_graph,
}   

__all__ = ['basic_ops_2_func_dict', 'add_custom_operator', 'add_custom_node_2_torch_graph', 'custom_add_2_torch_graph', 'utils']
