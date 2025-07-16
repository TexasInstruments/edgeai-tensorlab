# from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
# from torch.ao.quantization.backend_config.native import get_native_backend_config
# from torch.ao.quantization.backend_config import (
#     BackendConfig,
#     BackendPatternConfig,
#     DTypeConfig,
#     ObservationType,
# )
# from torch.ao.nn import intrinsic as nni
# import torch.ao.nn.intrinsic.qat as nniqat
# import torch.ao.nn.qat as nnqat
# import torch.ao.nn.quantized.reference as nnqr
# import torch.nn.functional as F

# def fuse_conv2d_relu(is_qat, conv, relu):
#     """Return a fused ConvReLU2d from individual conv and relu modules."""
#     return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)

# weighted_int8_dtype_config = DTypeConfig(
#     input_dtype=torch.quint8,
#     output_dtype=torch.quint8,
#     weight_dtype=torch.qint8,
#     bias_dtype=torch.float)

# # For fusing Conv2d + ReLU6 into ConvReLU62d
# conv_relu_config = BackendPatternConfig((torch.nn.Conv2d, torch.nn.ReLU6)) \
#     .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
#     .add_dtype_config(weighted_int8_dtype_config) \
#     .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d) \
#     .set_fuser_method(fuse_conv2d_relu)

# # For quantizing ConvReLU2d
# fused_conv_relu_config = BackendPatternConfig(torch.ao.nn.intrinsic.ConvReLU2d) \
#     .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
#     .add_dtype_config(weighted_int8_dtype_config) \
#     .set_root_module(torch.nn.Conv2d) \
#     .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d) \
#     .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)
    
# backend_config = get_native_backend_config() \
#     .set_backend_pattern_configs([conv_relu_config]) \
#     .set_backend_pattern_configs([fused_conv_relu_config])

# class ConvBnAct2d(nni._FusedModule):
#     r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
#     During quantization this will be replaced with the corresponding fused module."""

#     def __init__(self, conv, bn, relu):
#         super().__init__(conv, bn, relu)
        
# class ConvAct2d(nni._FusedModule):
#     r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
#     During quantization this will be replaced with the corresponding fused module."""

#     def __init__(self, conv, relu):
#         super().__init__(conv, relu)

# nni.ConvBnAct2d = ConvBnAct2d
# nni.ConvAct2d = ConvAct2d
        
# class QAT_ConvBnAct2d(nni.qat.modules.conv_fused.ConvBn2d):
#     # base class defines _FLOAT_MODULE as "ConvBn2d"
#     _FLOAT_MODULE = ConvBnAct2d  # type: ignore[assignment]
#     _FLOAT_CONV_MODULE = nn.Conv2d
#     _FLOAT_BN_MODULE = nn.BatchNorm2d
#     _FLOAT_RELU_MODULE = nn.ReLU6  # type: ignore[assignment]
#     # module class after fusing bn into conv
#     _FUSED_FLOAT_MODULE = ConvAct2d

#     def __init__(self,
#                 # Conv2d args
#                 in_channels, out_channels, kernel_size, stride=1,
#                 padding=0, dilation=1, groups=1,
#                 bias=None,
#                 padding_mode='zeros',
#                 # BatchNorm2d args
#                 # num_features: out_channels
#                 eps=1e-05, momentum=0.1,
#                 # affine: True
#                 # track_running_stats: True
#                 # Args for this module
#                 freeze_bn=False,
#                 qconfig=None):
#         super().__init__(in_channels, out_channels, kernel_size, stride,
#                         padding, dilation, groups, bias,
#                         padding_mode, eps, momentum,
#                         freeze_bn,
#                         qconfig)

#     def forward(self, input):
#         return F.relu6(nni.qat.modules.conv_fused.ConvBn2d._forward(self, input))

#     @classmethod
#     def from_float(cls, mod, use_precomputed_fake_quant=False):
#         return super().from_float(mod, use_precomputed_fake_quant)
    
# nni.ConvBnReLU2d = copy.deepcopy(ConvBnAct2d)
# nni.ConvReLU2d = copy.deepcopy(ConvAct2d)

# nni.ConvBnReLU2d.__module__ = str(nni)
# nni.ConvReLU2d.__module__ = str(nni)


# def fuse_conv2d_bn_act(is_qat, conv, bn, relu):
#     """Return a fused from individual conv, bn and relu modules."""
#     assert conv.training == bn.training == relu.training, \
#         "Conv and BN both must be in the same mode (train or eval)."
#     if is_qat:
#         assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
#         assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
#         assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
#         # return ConvBnReLU2d(conv, bn, relu)
#         return nni.ConvBnAct2d(conv, bn, relu)
#     else:
#         fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
#         # return ConvReLU2d(fused_conv, relu)
#         return nni.ConvAct2d(fused_conv, relu)
        

# # For fusing Conv2d + ReLU6 into ConvReLU62d
# # setting to nni.ConvBnReLU2d is slightly wrong as the relu6 will be  converted to relu due to this
# conv_bn_relu_config = BackendPatternConfig((torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU6)) \
#     .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
#     .add_dtype_config(weighted_int8_dtype_config) \
#     .set_fused_module(nni.ConvBnAct2d) \
#     .set_fuser_method(fuse_conv2d_bn_act)

# # For quantizing ConvReLU2d
# fused_conv_bn_relu_config = BackendPatternConfig(nni.ConvBnAct2d) \
#     .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
#     .add_dtype_config(weighted_int8_dtype_config) \
#     .set_root_module(torch.nn.Conv2d) \
#     .set_qat_module(nniqat.ConvBnAct2d) \
#     .set_reference_quantized_module(nnqr.Conv2d)

# backend_config = get_native_backend_config() \
#     .set_backend_pattern_configs([conv_bn_relu_config]) \
#     .set_backend_pattern_configs([fused_conv_bn_relu_config])

