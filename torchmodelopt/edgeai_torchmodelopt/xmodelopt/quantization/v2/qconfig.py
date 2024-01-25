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
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qat_qconfig
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, \
    FakeQuantize, FusedMovingAvgObsFakeQuantize

from .... import xnn

from . import observer
from . import fake_quanitze

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
    WT8SYMP2_AT8SYMP2 = "WT8SYMP2_AT8SYMP2"     # per-tensor symmetric power-power-of-2 quantization for both weights and activations
    WC8_AT8 = "WC8_AT8"                         # per-channel quantization for weights, per-tensor quantization for activations
    WC4_AT8 = "WC4_AT8"                         # 4-bits per-channel quantization for weights, 8-bit per-tensor quantization for activations
    WC4M4_AT8 = "WC4M4_AT8"                     # restricted range weights

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class QConfigMode(enum.Enum):
    DEFAULT = 0
    FREEZE_DEPTHWISE_LAYERS = 1
    FREEZE_UNSTABLE_LAYERS = 2

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


####################################################################
_QCONFIG_TYPE_TO_DICT = dict()

####################################################################
# per-channel
_QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

# symmetric power-of-2
_QCONFIG_TYPE_TO_DICT[QConfigType.WT8SYMP2_AT8SYMP2] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePower2WeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveSymPower2ActivationObserver))

# 4 bit modes
_QCONFIG_TYPE_TO_DICT[QConfigType.WC4_AT8] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePerChannelBit4WeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

# restricted range
_QCONFIG_TYPE_TO_DICT[QConfigType.WC4M4_AT8] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePerChannelBit4MaxRange4WeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

###########
_QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = _QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8]

# Note: get_default_qat_qconfig from pytorch uses fused_moving_avg_obs_fake_quant and that cannot be exported to onnx
#_QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = get_default_qat_qconfig()

####################################################################


def get_qconfig(is_qat, backend, qconfig_type=None):
    if isinstance(qconfig_type, QConfig):
        return qconfig_type
    elif isinstance(qconfig_type, str) and qconfig_type in _QCONFIG_TYPE_TO_DICT:
        qconfig_obj = _QCONFIG_TYPE_TO_DICT[qconfig_type]
    elif isinstance(qconfig_type, dict):
        # custom qconfig_type parameters are given in a dict
        # weight
        weight_observer_name = qconfig_type.get('weight_observer_name', 'CustomAdaptiveWeightObserver')
        weight_bitwidth = qconfig_type.get('weight_bitwidth', 8)
        weight_quant_min = qconfig_type.get('weight_quant_min', -(2 ** (weight_bitwidth-1)))
        weight_quant_max = qconfig_type.get('weight_quant_max', (2 ** (weight_bitwidth-1)) - 1)
        weight_per_channel = qconfig_type.get('weight_per_channel', True)
        weight_power2 = qconfig_type.get('weight_power2', False)
        weight_range_max = qconfig_type.get('weight_range_max', None)
        weight_fixed_range = qconfig_type.get('weight_fixed_range', False)
        # activation
        activation_observer_name = qconfig_type.get('activation_observer_name', 'CustomAdaptiveActivationObserver')
        activation_bitwidth = qconfig_type.get('activation_bitwidth', 8)
        activation_quant_min = qconfig_type.get('activation_quant_min', 0)
        activation_quant_max = qconfig_type.get('activation_quant_max', (2 ** activation_bitwidth) - 1)
        activation_range_max = qconfig_type.get('activation_range_max', None)
        activation_fixed_range = qconfig_type.get('activation_fixed_range', False)
        activation_symmetric = qconfig_type.get('activation_symmetric', False)
        activation_power2 = qconfig_type.get('activation_power2', False)
        # qconfig
        WeightObserverBaseToUse = observer.AdaptivePerChannelWeightObserver if weight_per_channel else observer.AdaptiveWeightObserver
        weight_observer = xnn.utils.partialclass(WeightObserverBaseToUse, quant_min=weight_quant_min, quant_max=weight_quant_max, power2=weight_power2, range_max=weight_range_max, fixed_range=weight_fixed_range, class_name=weight_observer_name)
        activation_observer = xnn.utils.partialclass(observer.AdaptiveActivationObserver, quant_min=activation_quant_min, quant_max=activation_quant_max, symmetric=activation_symmetric, power2=activation_power2, range_max=activation_range_max, fixed_range=activation_fixed_range, class_name=activation_observer_name)
        qconfig_obj = QConfig(weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=weight_observer), activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=activation_observer))
        return qconfig_obj
    else:
        raise RuntimeError("Unknown qconfig_type: " + str(qconfig_type))
    #
    return qconfig_obj


def get_qconfig_mapping(is_qat, backend, qconfig_type=None):
    qconfig_type_base = qconfig_type[0] if isinstance(qconfig_type, (list,tuple)) else qconfig_type
    qconfig = get_qconfig(is_qat, backend, qconfig_type_base)
    qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig)
    # apply specific qconfigs to specific types if needed
    qconfig_reuse = torch.ao.quantization.default_reuse_input_qconfig
    qconfig_map.set_object_type(torch.nn.Dropout, qconfig_reuse)
    return qconfig_map


def _apply_qconfig(pmodule, cmodule, cname, qconfig_aux, current_device):
    if isinstance(cmodule, fake_quanitze.ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES):
        setattr(pmodule, cname, qconfig_aux.weight().to(current_device))
    #
    elif isinstance(cmodule, fake_quanitze.ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES):
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
        if not input_fake_quant_module and isinstance(pmodule, fake_quanitze.AdaptiveActivationFakeQuantize):
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
