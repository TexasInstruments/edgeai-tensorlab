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
import enum
import warnings
import torch.ao.quantization
from torch.ao.quantization import QConfig
import torch.ao.quantization.quantize_fx
from torch.ao.quantization.qconfig_mapping import _get_default_qconfig_mapping_with_default_qconfig

from .... import xnn

from . import observer_types
from . import fake_quantize_types
    

class QConfigMethod(enum.Enum):
    DISABLED = 0
    QAT = 1

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class QConfigFormat:
    FLOAT_MODEL = "FLOAT_MODEL"    # original float model format
    FAKEQ_MODEL = "FAKEQ_MODEL"    # trained FakeQ model before conversion
    QDQ_MODEL = "QDQ_MODEL"        # converted QDQ model
    INT_MODEL = "INT_MODEL"        # integer model
    _NUM_FORMATS_ = 4

    @classmethod
    def choices(cls):
        return [value for value in dir(cls) if not value.startswith('__') and value != 'choices']


class QConfigType():
    DISABLED = 0
    DEFAULT = "DEFAULT"                         # default behavior is same as that of WC8_AT8
    WC8_AT8 = "WC8_AT8"                         # per-channel quantization for weights, per-tensor quantization for activations
    
    MSA_WC8_AT8 = "MSA_WC8_AT8"                 # WC8_AT8 + no range shrink (mostly for attention networks with important peaks)

    WT8SYMP2_AT8SYMP2 = "WT8SYMP2_AT8SYMP2"     # per-tensor symmetric power-of-2 quantization for both weights and activations
    WC8SYMP2_AT8SYMP2 = "WC8SYMP2_AT8SYMP2"     # per-channel symmetric power-of-2 quantization for weights, per-tensor symmetric power-of-2 for activations
    WC8SYMP2_AT8SYMP2R4 = "WC8SYMP2_AT8SYMP2R4" # same as above with fixed activation range
    
    WC8P2_AT8P2 = "WC8P2_AT8P2"                 # per-channel symmetric power-of-2 quantization for weights, per-tensor affine power-of-2 for activations
    MSA_WC8P2_AT8P2 = "MSA_WC8P2_AT8P2"         # WC8P2_AT8P2 + no range shrink (mostly for attention networks with important peaks)

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

def get_weight_quantization_config(weight_qconfig, is_qat=True):
    observer_name = 'CustomAdaptiveWeightObserver' + '__' + get_repr_string_from_dict(weight_qconfig)
    weight_bitwidth = weight_qconfig.get('bitwidth', 8)
    weight_qscheme = weight_qconfig.get('qscheme', torch.per_channel_symmetric)
	
    WeightObserverBaseToUse = observer_types.AdaptivePerChannelWeightObserver \
        if weight_qscheme == torch.per_channel_symmetric else observer_types.AdaptiveWeightObserver

    weight_observer = xnn.utils.partialclass(WeightObserverBaseToUse,
                                             quant_min=weight_qconfig.get('quant_min', -(2 ** (weight_bitwidth-1))),
                                             quant_max=weight_qconfig.get('quant_max', (2 ** (weight_bitwidth-1)) - 1),
                                             dtype=weight_qconfig.get('dtype', torch.qint8),
                                             qscheme=weight_qconfig.get('qscheme', torch.per_tensor_symmetric),
                                             power2_scale=weight_qconfig.get('power2_scale', False),
                                             range_max=weight_qconfig.get('range_max', None),
                                             fixed_range=weight_qconfig.get('fixed_range', False),
                                             class_name=weight_qconfig.get('observer_name', observer_name))
    return weight_observer


def get_act_quantization_config(activation_qconfig, is_qat=True, fast_mode=False):
    observer_name = 'CustomAdaptiveActivationObserver' + get_repr_string_from_dict(activation_qconfig)
    activation_bitwidth = activation_qconfig.get('bitwidth', 8)

    AdaptiveActivationObserverToUse = observer_types.AdaptiveActivationObserverFast if fast_mode else observer_types.AdaptiveActivationObserver
    
    activation_observer = xnn.utils.partialclass(AdaptiveActivationObserverToUse,
                                             quant_min=activation_qconfig.get('quant_min', 0),
                                             quant_max=activation_qconfig.get('quant_max', (2 ** activation_bitwidth) - 1),
                                             dtype=activation_qconfig.get('dtype', torch.quint8),
                                             qscheme=activation_qconfig.get('qscheme', torch.per_tensor_affine),
                                             power2_scale=activation_qconfig.get('power2_scale', False),
                                             range_max=activation_qconfig.get('range_max', None),
                                             fixed_range=activation_qconfig.get('fixed_range', False),
                                             class_name=activation_qconfig.get('observer_name', observer_name),
                                             range_shrink_percentile=activation_qconfig.get('range_shrink_percentile', 0.01))
               
    fake_quantized_activation_observer = fake_quantize_types.AdaptiveActivationFakeQuantize.with_args(observer=activation_observer) if is_qat else activation_observer
    
    return activation_observer


def get_quantization_config(qconfig_dict, is_qat=True, fast_mode=False):
    # custom qconfig_type parameters are given in a dict
    weight_observer = get_weight_quantization_config(qconfig_dict.get('weight', dict()), is_qat=is_qat)
    activation_observer = get_act_quantization_config(qconfig_dict.get('activation', dict()), is_qat=is_qat, fast_mode=fast_mode)

    qconfig_obj = QConfig(weight=fake_quantize_types.AdaptiveWeightFakeQuantize.with_args(observer=weight_observer),
                          activation=fake_quantize_types.AdaptiveActivationFakeQuantize.with_args(observer=activation_observer))
    return qconfig_obj


####################################################################
def get_quantization_config_default(qconfig_type, is_qat=True, fast_mode=False):
    _QCONFIG_TYPE_TO_DICT = dict()

    # per-channel
    _QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8] = get_quantization_config(dict(
        weight=dict(qscheme=torch.per_channel_symmetric),
        activation=dict(qscheme=torch.per_tensor_affine)), is_qat=is_qat, fast_mode=fast_mode)

    # per-channel transformers
    _QCONFIG_TYPE_TO_DICT[QConfigType.MSA_WC8_AT8] = get_quantization_config(dict(
        weight=dict(qscheme=torch.per_channel_symmetric),
        activation=dict(qscheme=torch.per_tensor_affine, range_shrink_percentile=0)), is_qat=is_qat, fast_mode=fast_mode)

    # symmetric power-of-2
    _QCONFIG_TYPE_TO_DICT[QConfigType.WT8SYMP2_AT8SYMP2] = get_quantization_config(dict(
        weight=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True),
        activation=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True)), is_qat=is_qat, fast_mode=fast_mode)

    # per-channel symmetric power-of-2
    _QCONFIG_TYPE_TO_DICT[QConfigType.WC8SYMP2_AT8SYMP2] = get_quantization_config(dict(
        weight=dict(qscheme=torch.per_channel_symmetric, power2_scale=True),
        activation=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True)), is_qat=is_qat, fast_mode=fast_mode)
    
    # per-channel power-of-2
    _QCONFIG_TYPE_TO_DICT[QConfigType.WC8P2_AT8P2] = get_quantization_config(dict(
        weight=dict(qscheme=torch.per_channel_symmetric, power2_scale=True),
        activation=dict(qscheme=torch.per_tensor_affine, power2_scale=True)), is_qat=is_qat, fast_mode=fast_mode)
    
    # per-channel power-of-2 transformers
    _QCONFIG_TYPE_TO_DICT[QConfigType.MSA_WC8P2_AT8P2] = get_quantization_config(dict(
        weight=dict(qscheme=torch.per_channel_symmetric, power2_scale=True),
        activation=dict(qscheme=torch.per_tensor_affine, power2_scale=True, range_shrink_percentile=0)), is_qat=is_qat, fast_mode=fast_mode)

    # per-channel symmetric power-of-2, fixed activation range
    _QCONFIG_TYPE_TO_DICT[QConfigType.WC8SYMP2_AT8SYMP2R4] = get_quantization_config(dict(
        weight=dict(qscheme=torch.per_channel_symmetric, power2_scale=True),
        activation=dict(qscheme=torch.per_tensor_symmetric, power2_scale=True, rage_max=4, fixed_range=True)), is_qat=is_qat, fast_mode=fast_mode)

    # 4 bit weight
    _QCONFIG_TYPE_TO_DICT[QConfigType.WC4_AT8] = get_quantization_config(dict(
        weight=dict(bitwidth=4, qscheme=torch.per_channel_symmetric),
        activation=dict(qscheme=torch.per_tensor_affine)), is_qat=is_qat, fast_mode=fast_mode)

    # 4 bit weight, restricted range
    _QCONFIG_TYPE_TO_DICT[QConfigType.WC4M4_AT8] = get_quantization_config(dict(
        weight=dict(bitwidth=4, qscheme=torch.per_channel_symmetric, range_max=4),
        activation=dict(qscheme=torch.per_tensor_affine)), is_qat=is_qat, fast_mode=fast_mode)

    ###########
    # _QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = _QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8]
    _QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = _QCONFIG_TYPE_TO_DICT[QConfigType.MSA_WC8_AT8]

    return _QCONFIG_TYPE_TO_DICT[qconfig_type]

####################################################################


def get_qconfig(qconfig_type=None, is_qat=True, fast_mode=False, backend='qnnpack'):
    if isinstance(qconfig_type, QConfig):
        return qconfig_type
    elif isinstance(qconfig_type, str):
        qconfig_obj = get_quantization_config_default(qconfig_type, is_qat=is_qat, fast_mode=fast_mode)
    elif isinstance(qconfig_type, dict):
        # custom qconfig_type parameters are given in a dict
        qconfig_obj = get_quantization_config(qconfig_type, is_qat=is_qat, fast_mode=fast_mode)
    else:
        raise RuntimeError("Unknown qconfig_type: " + str(qconfig_type))
    #
    return qconfig_obj


def get_qconfig_mapping(is_qat, backend, qconfig_type=None):
    qconfig_type_base = qconfig_type[0] if isinstance(qconfig_type, (list,tuple)) else qconfig_type
    qconfig_type = get_qconfig(qconfig_type=qconfig_type_base, is_qat=is_qat, backend=backend)
    qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig_type)
    # apply specific qconfigs to specific types if needed
    qconfig_reuse = torch.ao.quantization.default_reuse_input_qconfig
    qconfig_map.set_object_type(torch.nn.Dropout, qconfig_reuse)
    return qconfig_map


def _apply_qconfig(pmodule, cmodule, cname, qconfig_aux, current_device):
    if isinstance(cmodule, fake_quantize_types.ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES):
        setattr(pmodule, cname, qconfig_aux.weight().to(current_device))
    #
    elif isinstance(cmodule, fake_quantize_types.ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES):
        setattr(pmodule, cname, qconfig_aux.activation().to(current_device))
    #


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
