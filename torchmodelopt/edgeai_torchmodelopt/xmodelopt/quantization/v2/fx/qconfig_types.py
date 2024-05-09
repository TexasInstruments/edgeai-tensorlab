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

import enum
import warnings
import torch
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qat_qconfig, get_default_qconfig_mapping
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, \
    FakeQuantize, FusedMovingAvgObsFakeQuantize
import torch.ao.quantization
import torch.ao.quantization.quantize_fx
import torch.nn as nn

from .... import xnn

from . import observer_types
from . import fake_quanitze_types

try:
    # this is not part of torch 2.0.1 release - so provide an alternate implementation for now
    from torch.ao.quantization.qconfig_mapping import \
        get_default_qat_qconfig_mapping, _get_default_qconfig_mapping_with_default_qconfig
    warnings.warn("could not find _get_default_qconfig_mapping_with_default_qconfig in torch.ao.quantization.qconfig_mapping")
except:
    from torch.ao.quantization.qconfig_mapping import _FIXED_QPARAMS_OP_TO_OBSERVER
    def _get_default_qconfig_mapping_with_default_qconfig(
        is_qat: bool,
        backend: str,
        default_qconfig: QConfig,
    ) -> QConfigMapping:
        """
        Return a QConfigMapping that uses the provided qconfig as the default QConfig.
        """
        if is_qat:
            qconfig_mapping = get_default_qat_qconfig_mapping(backend)
        else:
            qconfig_mapping = get_default_qconfig_mapping(backend)
        qconfig_mapping.set_global(default_qconfig)
        for pattern in qconfig_mapping.object_type_qconfigs.keys():
            if pattern not in _FIXED_QPARAMS_OP_TO_OBSERVER:
                qconfig_mapping.set_object_type(pattern, default_qconfig)
        return qconfig_mapping


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


####################################################################
def get_weight_observer_from_dict(weight_qconfig):
    observer_name = 'CustomAdaptiveWeightObserver' + '__' + get_repr_string_from_dict(weight_qconfig)
    weight_bitwidth = weight_qconfig.get('bitwidth', 8)
    weight_qscheme = weight_qconfig.get('qscheme', torch.per_channel_symmetric)
    # qconfig
    WeightObserverBaseToUse = observer_types.AdaptivePerChannelWeightObserver \
        if weight_qscheme == torch.per_channel_symmetric else observer_types.AdaptiveWeightObserver
    weight_observer = xnn.utils.partialclass(WeightObserverBaseToUse,
                                             quant_min=weight_qconfig.get('quant_min', -(2 ** (weight_bitwidth-1))),
                                             quant_max=weight_qconfig.get('quant_max', (2 ** (weight_bitwidth-1)) - 1),
                                             dtype=weight_qconfig.get('dtype', torch.qint8),
                                             power2_scale=weight_qconfig.get('power2_scale', False),
                                             range_max=weight_qconfig.get('range_max', None),
                                             fixed_range=weight_qconfig.get('fixed_range', False),
                                             class_name=weight_qconfig.get('observer_name', observer_name))
    return weight_observer


def get_activation_observer_from_dict(activation_qconfig):
    observer_name = 'CustomAdaptiveActivationObserver' + get_repr_string_from_dict(activation_qconfig)
    activation_bitwidth = activation_qconfig.get('bitwidth', 8)
    activation_observer = xnn.utils.partialclass(observer_types.AdaptiveActivationObserver,
                                             quant_min=activation_qconfig.get('quant_min', 0),
                                             quant_max=activation_qconfig.get('quant_max', (2 ** activation_bitwidth) - 1),
                                             dtype=activation_qconfig.get('dtype', torch.quint8),
                                             qscheme=activation_qconfig.get('qscheme', torch.per_tensor_affine),
                                             power2_scale=activation_qconfig.get('power2_scale', False),
                                             range_max=activation_qconfig.get('range_max', None),
                                             fixed_range=activation_qconfig.get('fixed_range', False),
                                             class_name=activation_qconfig.get('observer_name', observer_name),
                                             range_shrink_percentile=activation_qconfig.get('range_shrink_percentile', 0.01))
    return activation_observer


def get_qconfig_from_dict(qconfig_dict):
    # custom qconfig_type parameters are given in a dict
    weight_observer = get_weight_observer_from_dict(qconfig_dict['weight']) if isinstance(qconfig_dict['weight'], dict) else qconfig_dict['weight']
    activation_observer = get_activation_observer_from_dict(qconfig_dict['activation']) if isinstance(qconfig_dict['activation'], dict) else qconfig_dict['activation']
    qconfig_obj = QConfig(weight=fake_quanitze_types.AdaptiveWeightFakeQuantize.with_args(observer=weight_observer),
                          activation=fake_quanitze_types.AdaptiveActivationFakeQuantize.with_args(observer=activation_observer))
    return qconfig_obj


####################################################################
_QCONFIG_TYPE_TO_DICT = dict()

# per-channel
_QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8] = get_qconfig_from_dict(dict(
    weight=dict(qscheme=torch.per_channel_symmetric),
    activation=dict(qscheme=torch.per_tensor_affine)))

# per-channel transformers
_QCONFIG_TYPE_TO_DICT[QConfigType.MSA_WC8_AT8] = get_qconfig_from_dict(dict(
    weight=dict(qscheme=torch.per_channel_symmetric),
    activation=dict(qscheme=torch.per_tensor_affine, range_shrink_percentile=0)))

# symmetric power-of-2
_QCONFIG_TYPE_TO_DICT[QConfigType.WT8SYMP2_AT8SYMP2] = get_qconfig_from_dict(dict(
    weight=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True),
    activation=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True)))

# per-channel symmetric power-of-2
_QCONFIG_TYPE_TO_DICT[QConfigType.WC8SYMP2_AT8SYMP2] = get_qconfig_from_dict(dict(
    weight=dict(qscheme=torch.per_channel_symmetric, power2_scale=True),
    activation=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True)))

# per-channel symmetric power-of-2, fixed activation range
_QCONFIG_TYPE_TO_DICT[QConfigType.WC8SYMP2_AT8SYMP2R4] = get_qconfig_from_dict(dict(
    weight=dict(qscheme=torch.per_channel_symmetric, power2_scale=True),
    activation=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True, rage_max=4, fixed_range=True)))

# 4 bit weight
_QCONFIG_TYPE_TO_DICT[QConfigType.WC4_AT8] = get_qconfig_from_dict(dict(
    weight=dict(bitwidth=4, qscheme=torch.per_channel_symmetric),
    activation=dict(qscheme=torch.per_tensor_affine)))

# 4 bit weight, restricted range
_QCONFIG_TYPE_TO_DICT[QConfigType.WC4M4_AT8] = get_qconfig_from_dict(dict(
    weight=dict(bitwidth=4, qscheme=torch.per_channel_symmetric, range_max=4),
    activation=dict(qscheme=torch.per_tensor_affine)))

###########
# _QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = _QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8]
_QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = _QCONFIG_TYPE_TO_DICT[QConfigType.MSA_WC8_AT8]

# Note: get_default_qat_qconfig from pytorch uses fused_moving_avg_obs_fake_quant and that cannot be exported to onnx
#_QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = get_default_qat_qconfig()

####################################################################


def get_qconfig(is_qat, backend='qnnpack', qconfig_type=None):
    if isinstance(qconfig_type, QConfig):
        return qconfig_type
    elif isinstance(qconfig_type, str) and qconfig_type in _QCONFIG_TYPE_TO_DICT:
        qconfig_obj = _QCONFIG_TYPE_TO_DICT[qconfig_type]
    elif isinstance(qconfig_type, dict):
        # custom qconfig_type parameters are given in a dict
        qconfig_obj = get_qconfig_from_dict(is_qat=is_qat, backend=backend, qconfig_type=qconfig_type)
    else:
        raise RuntimeError("Unknown qconfig_type: " + str(qconfig_type))
    #
    return qconfig_obj


def get_qconfig_mapping(is_qat, backend, qconfig_type=None):
    qconfig_type_base = qconfig_type[0] if isinstance(qconfig_type, (list,tuple)) else qconfig_type
    qconfig_type = get_qconfig(is_qat, backend, qconfig_type_base)
    qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig_type)
    # apply specific qconfigs to specific types if needed
    qconfig_reuse = torch.ao.quantization.default_reuse_input_qconfig
    qconfig_map.set_object_type(torch.nn.Dropout, qconfig_reuse)
    return qconfig_map


def _apply_qconfig(pmodule, cmodule, cname, qconfig_aux, current_device):
    if isinstance(cmodule, fake_quanitze_types.ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES):
        setattr(pmodule, cname, qconfig_aux.weight().to(current_device))
    #
    elif isinstance(cmodule, fake_quanitze_types.ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES):
        setattr(pmodule, cname, qconfig_aux.activation().to(current_device))
    #


def adjust_mixed_precision_qconfig(model, is_qat, backend, qconfig_type):
    if not isinstance(qconfig_type, (list,tuple)) or len(qconfig_type) == 1:
        return model

    current_device = next(model.parameters()).device
    qconfig_type_aux = qconfig_type[-1]
    qconfig_aux = get_qconfig(is_qat, backend, qconfig_type_aux)

    input_fake_quant_module = None
    input_conv_module = None
    output_linear_module = None
    depthwise_conv_module = None
    for pname, pmodule in list(model.named_modules()):
        if not input_fake_quant_module and isinstance(pmodule, fake_quanitze_types.AdaptiveActivationFakeQuantize):
            # input activation_module
            input_fake_quant_module = pmodule
        if not input_conv_module and isinstance(pmodule, torch.nn.Conv2d) and pmodule.in_channels < 8:
            # first conv module
            input_conv_module = pmodule
        if isinstance(pmodule, torch.nn.Linear):
            # last linear module
            output_linear_module = pmodule
        if isinstance(pmodule, torch.nn.Conv2d) and pmodule.groups == pmodule.in_channels:
            depthwise_conv_module = pmodule
        #
    #
    for pname, pmodule in list(model.named_modules()):
        for cname, cmodule in list(pmodule.named_children()):
            if (pmodule is input_conv_module) or (pmodule is output_linear_module):
                _apply_qconfig(pmodule, cmodule, cname, qconfig_aux, current_device)
            # elif pmodule is depthwise_conv_module:
            #     _apply_qconfig(pmodule, cmodule, cname, qconfig_aux, current_device)
            elif cmodule is input_fake_quant_module:
                _apply_qconfig(pmodule, cmodule, cname, qconfig_aux, current_device)
            #
        #
    #
    return model


def is_softmax_present(node, find_level):
    if "softmax" in str(node.args[0].target):
        return True
    elif find_level>0:
        return is_softmax_present(node.args[0], find_level-1)
    else:
        return False


def adjust_matmul_inputs_qconfig(model):
    # setting symmetric quantization scheme for the inputs of the matmul
    for node in model.graph.nodes:
        for n_id in node.users:
            if n_id.target=='output':
                continue
            if n_id.target==torch.matmul:
                if is_softmax_present(node, 4):
                    pass
                else:
                    f = getattr(model, str(node))
                    f.activation_post_process.symmetric = True
                    setattr(model, str(node), f)  
                #
            #
        #
    #
    return model

    
def is_mlp_fc2_layer(all_modules, node, find_level, found_gelu=False):
    if find_level<0:
        return False
    elif node.target in all_modules:
        if isinstance(all_modules[node.target], nn.Linear):
            if found_gelu: 
                return True
            else: 
                # found linear before the gelu layer
                return False
            #
        #
        elif isinstance(all_modules[node.target], nn.GELU):
            found_gelu = True
        #
    #
    return is_mlp_fc2_layer(all_modules, node.args[0], find_level-1, found_gelu)
    

def adjust_fc_outlier_supression(model):
    # changing the observer of the second fc layer in each mlp to outlier removal observer
    all_modules = dict(model.named_modules())
    for node in model.graph.nodes:
        if (node.target in all_modules) and isinstance(all_modules[node.target], nn.Linear):
            if is_mlp_fc2_layer(all_modules, node.args[0], 6):
                new_activation_observer = xnn.utils.partialclass(observer_types.AdaptiveOutlierRemovalActivationObserver,
                    quant_min=all_modules[node.next.target].activation_post_process.quant_min,
                    quant_max=all_modules[node.next.target].activation_post_process.quant_max,
                    dtype=all_modules[node.next.target].activation_post_process.dtype,
                    qscheme=all_modules[node.next.target].activation_post_process.qscheme,
                    power2_scale=all_modules[node.next.target].activation_post_process.power2_scale,
                    range_max=all_modules[node.next.target].activation_post_process.range_max,
                    fixed_range=all_modules[node.next.target].activation_post_process.fixed_range,
                    class_name='OutlierRemoval' + all_modules[node.next.target].activation_post_process._get_name(),
                    range_shrink_percentile=all_modules[node.next.target].activation_post_process.range_shrink_percentile)   
                orig_fake_quantize = getattr(model, str(node.next))
                new_fake_quantize = fake_quanitze_types.AdaptiveActivationFakeQuantize.with_args(observer=new_activation_observer)
                setattr(model, str(node.next), new_fake_quantize().to(orig_fake_quantize.zero_point.device))
            #
        #
    #
    return model