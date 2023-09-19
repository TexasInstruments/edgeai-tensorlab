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


class QConfigType(enum.Enum):
    DISABLED = 0
    DEFAULT = "DEFAULT"

    WT8_AT8 = "WT8_AT8"
    WC8_AT8 = "WC8_AT8"
    WT8SP2_AT8SP2 = "WT8SP2_AT8SP2"
    WC8SP2_AT8SP2 = "WC8SP2_AT8SP2"

    WC4_AT8 = "WC4_AT8"
    WC4R4_AT8 = "WC4R4_AT8"

    WC4_AT4 = "WC4_AT4"
    WC4R4_AT4R4 = "WC4R4_AT4R4"

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

_QCONFIG_TYPE_TO_DICT[QConfigType.WT8_AT8] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptiveWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.WT8SP2_AT8SP2] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePower2WeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptivePower2ActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.WC8SP2_AT8SP2] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePower2PerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptivePower2ActivationObserver))

###########
_QCONFIG_TYPE_TO_DICT[QConfigType.WC4_AT8] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptiveLowBITPerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.WC4R4_AT8] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptiveRangeRestricted4LowBITPerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

###########
_QCONFIG_TYPE_TO_DICT[QConfigType.WC4_AT4] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptiveLowBITPerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveLowBITActivationObserver))


_QCONFIG_TYPE_TO_DICT[QConfigType.WC4R4_AT4R4] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptiveRangeRestricted4LowBITPerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveRangeRestricted4LowBITActivationObserver))

###########
# get_default_qat_qconfig from pytorch uses fused_moving_avg_obs_fake_quant and that cannot be exported to onnx
#_QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = get_default_qat_qconfig()
_QCONFIG_TYPE_TO_DICT[QConfigType.DEFAULT] = _QCONFIG_TYPE_TO_DICT[QConfigType.WC8_AT8]

####################################################################


def get_qconfig(is_qat, backend, qconfig_type=None):
    qconfig_type = QConfigType(qconfig_type)
    if qconfig_type not in _QCONFIG_TYPE_TO_DICT:
        raise RuntimeError("Unknown qconfig_type: " + str(qconfig_type))
    #
    qconfig = _QCONFIG_TYPE_TO_DICT[qconfig_type]
    return qconfig


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
