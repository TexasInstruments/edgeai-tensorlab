#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import torch
from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
    MinMaxObserver,
    PlaceholderObserver,
)
from .observer_types import (
    AdaptiveWeightObserver,
    AdaptivePerChannelWeightObserver,
    AdaptiveActivationObserver,
    AdaptiveOutlierRemovalActivationObserver
)
from .fake_quantize_types import AdaptiveWeightFakeQuantize, AdaptiveActivationFakeQuantize
from torch.ao.quantization.quantizer.quantizer import (
    Quantizer,
    QuantizationAnnotation,
    SharedQuantizationSpec,
    QuantizationSpec
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from typing import Callable, Dict, List, Optional, Set, Any
import enum
import warnings
from ..... import xnn
    

class QConfigMethod(enum.Enum):
    DISABLED = 0
    QAT = 1

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class QConfigType():
    DISABLED = 0
    DEFAULT = "DEFAULT"                         # default behavior is same as that of WC8_AT8
    WC8_AT8 = "WC8_AT8"                         # per-channel quantization for weights, per-tensor quantization for activations
    
    MSA_WC8_AT8 = "MSA_WC8_AT8"                 # WC8_AT8 + no range shrink (mostly for attention networks with important peaks)

    WT8SYMP2_AT8SYMP2 = "WT8SYMP2_AT8SYMP2"     # per-tensor symmetric power-power-of-2 quantization for both weights and activations
    WC8SYMP2_AT8SYMP2 = "WC8SYMP2_AT8SYMP2"     # per-channel symmetric power-power-of-2 quantization for weights, per-tensor symmetric for activations
    WC8SYMP2_AT8SYMP2R4 = "WC8SYMP2_AT8SYMP2R4" # same as above with fixed activation range

    WC4_AT8 = "WC4_AT8"                         # 4-bits per-channel quantization for weights, 8-bit per-tensor quantization for activations
    WC4M4_AT8 = "WC4M4_AT8"                     # same as above with a maximum weight range

    @classmethod
    def choices(cls):
        return [value for value in dir(cls) if not value.startswith('__') and value != 'choices']


class QConfigMode(enum.Enum):
    DEFAULT = 0
    FREEZE_DEPTHWISE_LAYERS = 1
    FREEZE_UNSTABLE_LAYERS = 2

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


def get_repr_string_from_dict(input_dict):
    repr_string = []
    for k, v in input_dict.items():
        repr_string += [f'{k}_{v}'.replace('.', '_')]
    #
    return '__'.join(repr_string)

########################################################################
# def get_act_quantization_config(activation_qconfig):
#     observer_name = 'CustomAdaptiveActivationObserver' + get_repr_string_from_dict(activation_qconfig)
#     activation_bitwidth = activation_qconfig.get('bitwidth', 8)
    
#     act_quantization_spec = QuantizationSpec(
#         dtype=activation_qconfig.get('dtype', torch.int8),
#         quant_min=activation_qconfig.get('quant_min', -(2 ** (activation_bitwidth-1))),
#         quant_max=activation_qconfig.get('quant_max', (2 ** (activation_bitwidth-1)) - 1),
#         qscheme=activation_qconfig.get('qscheme', torch.per_tensor_affine),
#         is_dynamic=activation_qconfig.get('is_dynamic', False),
#         observer_or_fake_quant_ctr=activation_qconfig.get('observer_or_fake_quant_ctr',\
#             AdaptiveActivationObserver).with_args(
#                 quant_min=activation_qconfig.get('quant_min', -(2 ** (activation_bitwidth-1))),
#                 quant_max=activation_qconfig.get('quant_max', (2 ** (activation_bitwidth-1)) - 1),
#                 dtype=activation_qconfig.get('dtype', torch.int8),
#                 qscheme=activation_qconfig.get('qscheme', torch.per_tensor_affine),
#                 power2_scale=activation_qconfig.get('power2_scale', False),
#                 range_max=activation_qconfig.get('range_max', None),
#                 fixed_range=activation_qconfig.get('fixed_range', False),
#                 # class_name=activation_qconfig.get('observer_name', observer_name),
#                 range_shrink_percentile=activation_qconfig.get('range_shrink_percentile', 0.01)
#             )
#     )
#     return act_quantization_spec


# def get_weight_activation_config(weight_qconfig):
#     observer_name = 'CustomAdaptiveWeightObserver' + '__' + get_repr_string_from_dict(weight_qconfig)
#     weight_bitwidth = weight_qconfig.get('bitwidth', 8)
#     weight_qscheme = weight_qconfig.get('qscheme', torch.per_channel_symmetric)
#     WeightObserverBaseToUse = AdaptivePerChannelWeightObserver \
#         if weight_qscheme == torch.per_channel_symmetric else AdaptiveWeightObserver
        
#     weight_quantization_spec = QuantizationSpec(
#         dtype=weight_qconfig.get('dtype', torch.int8),
#         quant_min=weight_qconfig.get('quant_min', -(2 ** (weight_bitwidth-1))),  
#         quant_max=weight_qconfig.get('quant_max', (2 ** (weight_bitwidth-1)) - 1),
#         qscheme=weight_qscheme,
#         ch_axis=weight_qconfig.get('ch_axis', 0),
#         is_dynamic=weight_qconfig.get('is_dynamic', False),
#         observer_or_fake_quant_ctr=WeightObserverBaseToUse.with_args(
#             quant_min=weight_qconfig.get('quant_min', -(2 ** (weight_bitwidth-1))),  
#             quant_max=weight_qconfig.get('quant_max', (2 ** (weight_bitwidth-1)) - 1),
#             dtype=weight_qconfig.get('dtype', torch.int8),
#             power2_scale=weight_qconfig.get('power2_scale', False),
#             range_max=weight_qconfig.get('range_max', None),
#             fixed_range=weight_qconfig.get('fixed_range', False),
#             # class_name=weight_qconfig.get('observer_name', observer_name)
#         )                                 
#     )
#     return weight_quantization_spec


def get_act_quantization_config(activation_qconfig):
    observer_name = 'CustomAdaptiveActivationObserver' + get_repr_string_from_dict(activation_qconfig)
    activation_bitwidth = activation_qconfig.get('bitwidth', 8)
    
    activation_observer = xnn.utils.partialclass(activation_qconfig.get('observer_or_fake_quant_ctr', AdaptiveActivationObserver),
                                             quant_min=activation_qconfig.get('quant_min', 0),
                                             quant_max=activation_qconfig.get('quant_max', (2 ** activation_bitwidth) - 1),
                                             dtype=activation_qconfig.get('dtype', torch.uint8),
                                             qscheme=activation_qconfig.get('qscheme', torch.per_tensor_affine),
                                             power2_scale=activation_qconfig.get('power2_scale', False),
                                             range_max=activation_qconfig.get('range_max', None),
                                             fixed_range=activation_qconfig.get('fixed_range', False),
                                             class_name=activation_qconfig.get('observer_name', observer_name),
                                             range_shrink_percentile=activation_qconfig.get('range_shrink_percentile', 0.01))
    
    # activation_observer = activation_qconfig.get('observer_or_fake_quant_ctr',\
    #         AdaptiveActivationObserver).with_args(
    #             quant_min=activation_qconfig.get('quant_min', 0),
    #             quant_max=activation_qconfig.get('quant_max', (2 ** (activation_bitwidth)) - 1),
    #             dtype=activation_qconfig.get('dtype', torch.uint8),
    #             qscheme=activation_qconfig.get('qscheme', torch.per_tensor_affine),
    #             power2_scale=activation_qconfig.get('power2_scale', False),
    #             range_max=activation_qconfig.get('range_max', None),
    #             fixed_range=activation_qconfig.get('fixed_range', False),
    #             # class_name=activation_qconfig.get('observer_name', observer_name),
    #             range_shrink_percentile=activation_qconfig.get('range_shrink_percentile', 0.01)
    #         )
            
    # fake_quantized_activation_observer = AdaptiveActivationFakeQuantize.with_args(observer=AdaptiveActivationObserver, **activation_qconfig)
    # fake_quantized_activation_observer = AdaptiveActivationFakeQuantize.with_args(observer=activation_observer)
    fake_quantized_activation_observer = activation_observer
    
    act_quantization_spec = QuantizationSpec(
        dtype=activation_qconfig.get('dtype', torch.uint8),
        quant_min=activation_qconfig.get('quant_min', 0),
        quant_max=activation_qconfig.get('quant_max', (2 ** (activation_bitwidth)) - 1),
        qscheme=activation_qconfig.get('qscheme', torch.per_tensor_affine),
        is_dynamic=activation_qconfig.get('is_dynamic', False),
        observer_or_fake_quant_ctr=fake_quantized_activation_observer
    )
    return act_quantization_spec


def get_weight_activation_config(weight_qconfig):
    # observer_name = 'CustomAdaptiveWeightObserver' + '__' + get_repr_string_from_dict(weight_qconfig)
    weight_bitwidth = weight_qconfig.get('bitwidth', 8)
    weight_qscheme = weight_qconfig.get('qscheme', torch.per_channel_symmetric)
    WeightObserverBaseToUse = AdaptivePerChannelWeightObserver \
        if weight_qscheme == torch.per_channel_symmetric else AdaptiveWeightObserver
    
    
    weight_observer = xnn.utils.partialclass(WeightObserverBaseToUse,
                                             quant_min=weight_qconfig.get('quant_min', 0),
                                             quant_max=weight_qconfig.get('quant_max', (2 ** weight_bitwidth) - 1),
                                             dtype=weight_qconfig.get('dtype', torch.uint8),
                                             qscheme=weight_qconfig.get('qscheme', torch.per_tensor_symmetric),
                                             power2_scale=weight_qconfig.get('power2_scale', False),
                                             range_max=weight_qconfig.get('range_max', None),
                                             fixed_range=weight_qconfig.get('fixed_range', False))
                                            #  class_name=weight_qconfig.get('observer_name', observer_name),
                                            #  range_shrink_percentile=weight_qconfig.get('range_shrink_percentile', 0.01))
    
        
    # weight_observer = WeightObserverBaseToUse.with_args(
    #         # quant_min=weight_qconfig.get('quant_min',-127),  
    #         # quant_max=weight_qconfig.get('quant_max', 127),
    #         # dtype=weight_qconfig.get('dtype', torch.int8),
    #         quant_min=weight_qconfig.get('quant_min',0),  
    #         quant_max=weight_qconfig.get('quant_max', (2 ** (weight_bitwidth)) - 1),
    #         dtype=weight_qconfig.get('dtype', torch.uint8),
    #         power2_scale=weight_qconfig.get('power2_scale', False),
    #         range_max=weight_qconfig.get('range_max', None),
    #         fixed_range=weight_qconfig.get('fixed_range', False),
    #         # class_name=weight_qconfig.get('observer_name', observer_name)
    #     )
    
    # fake_quantized_weight_observer = AdaptiveWeightFakeQuantize.with_args(observer=WeightObserverBaseToUse, **weight_qconfig)
    # fake_quantized_weight_observer = AdaptiveWeightFakeQuantize.with_args(observer=weight_observer)
    fake_quantized_weight_observer = weight_observer
        
    weight_quantization_spec = QuantizationSpec(
        dtype=weight_qconfig.get('dtype', torch.int8),
        quant_min=weight_qconfig.get('quant_min', -127),  
        quant_max=weight_qconfig.get('quant_max', 127),
        # dtype=weight_qconfig.get('dtype', torch.uint8),
        # quant_min=weight_qconfig.get('quant_min', 0),  
        # quant_max=weight_qconfig.get('quant_max', (2 ** (weight_bitwidth)) - 1),
        qscheme=weight_qscheme,
        ch_axis=weight_qconfig.get('ch_axis', 0),
        is_dynamic=weight_qconfig.get('is_dynamic', False),
        observer_or_fake_quant_ctr=fake_quantized_weight_observer                         
    )
    return weight_quantization_spec


def get_quantization_config(qconfig_dict, is_qat=False):
    act_quantization_spec = get_act_quantization_config(qconfig_dict.get('activation', dict()))
    weight_quantization_spec = get_weight_activation_config(qconfig_dict.get('weight', dict()))

    bias_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = PlaceholderObserver
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.float,
        observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr
    )
    
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_qat
    )
    return quantization_config


def adjust_mixed_precision_qconfig(model, is_qat, qconfig_type):
    warnings.warn("Adjust Mixed Precision is not implemented yet!")
    return model


####################################################################
_QCONFIG_TYPE_TO_DICT = dict()

# per-channel
_QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8] = get_quantization_config(dict(
    weight=dict(qscheme=torch.per_channel_symmetric),
    activation=dict(qscheme=torch.per_tensor_affine)))

# per-channel transformers
_QCONFIG_TYPE_TO_DICT[QConfigType.MSA_WC8_AT8] = get_quantization_config(dict(
    weight=dict(qscheme=torch.per_channel_symmetric),
    activation=dict(qscheme=torch.per_tensor_affine, range_shrink_percentile=0)))

# symmetric power-of-2
_QCONFIG_TYPE_TO_DICT[QConfigType.WT8SYMP2_AT8SYMP2] = get_quantization_config(dict(
    weight=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True),
    activation=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True)))

# per-channel symmetric power-of-2
_QCONFIG_TYPE_TO_DICT[QConfigType.WC8SYMP2_AT8SYMP2] = get_quantization_config(dict(
    weight=dict(qscheme=torch.per_channel_symmetric, power2_scale=True),
    activation=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True)))

# per-channel symmetric power-of-2, fixed activation range
_QCONFIG_TYPE_TO_DICT[QConfigType.WC8SYMP2_AT8SYMP2R4] = get_quantization_config(dict(
    weight=dict(qscheme=torch.per_channel_symmetric, power2_scale=True),
    activation=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True, rage_max=4, fixed_range=True)))

# 4 bit weight
_QCONFIG_TYPE_TO_DICT[QConfigType.WC4_AT8] = get_quantization_config(dict(
    weight=dict(bitwidth=4, qscheme=torch.per_channel_symmetric),
    activation=dict(qscheme=torch.per_tensor_affine)))

# 4 bit weight, restricted range
_QCONFIG_TYPE_TO_DICT[QConfigType.WC4M4_AT8] = get_quantization_config(dict(
    weight=dict(bitwidth=4, qscheme=torch.per_channel_symmetric, range_max=4),
    activation=dict(qscheme=torch.per_tensor_affine)))

###########
# _QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = _QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8]
_QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = _QCONFIG_TYPE_TO_DICT[QConfigType.MSA_WC8_AT8]

####################################################################


def get_qconfig(qconfig_type=None):
    if isinstance(qconfig_type, QuantizationConfig):
        return qconfig_type
    elif isinstance(qconfig_type, str) and qconfig_type in _QCONFIG_TYPE_TO_DICT:
        qconfig_obj = _QCONFIG_TYPE_TO_DICT[qconfig_type]
    elif isinstance(qconfig_type, dict):
        # custom qconfig_type parameters are given in a dict
        qconfig_obj = get_quantization_config(qconfig_type)
    else:
        raise RuntimeError("Unknown qconfig_type: " + str(qconfig_type))
    #
    return qconfig_obj