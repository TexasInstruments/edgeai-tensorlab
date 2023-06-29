#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
from torch.ao.quantization import default_fused_per_channel_wt_fake_quant, default_fused_act_fake_quant
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


class QConfigType(enum.Enum):
    # default is same as QCONFIG_TYPE_W8T_A8T
    QCONFIG_TYPE_DEFAULT = "DEFAULT"
    QCONFIG_TYPE_W8T_A8T = "W8T_A8T"
    QCONFIG_TYPE_W8C_A8T = "W8C_A8T"
    QCONFIG_TYPE_W8T_A8T_SYM_P2 = "W8T_A8T_SYM_P2"
    QCONFIG_TYPE_W8C_A8T_SYM_P2 = "W8C_A8T_SYM_P2"
    QCONFIG_TYPE_W4C_A8T = "W4C_A8T"
    QCONFIG_TYPE_W4C_A4T = "W4C_A4T"
    QCONFIG_TYPE_W4C_A4T_FR = "W4C_A4T_FR"
    QCONFIG_TYPE_W4C_A4T_FR_NOQ = "W4C_A4T_FR_NOQ"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


####################################################################
_QCONFIG_TYPE_TO_DICT = dict()

_QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_W8T_A8T] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptiveWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_W8C_A8T] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_W8T_A8T_SYM_P2] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePower2WeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptivePower2ActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_W8C_A8T_SYM_P2] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePower2PerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptivePower2ActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_W4C_A8T] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_W4C_A4T] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_W4C_A4T_FR] = QConfig(
    weight=fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptiveFixedRangeLowBITPerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveFixedRangeLowBITActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_W4C_A4T_FR_NOQ] = QConfig(
    weight=fake_quanitze.AdaptiveWeightNoQuantize.with_args(observer=observer.AdaptiveFixedRangeLowBITPerChannelWeightObserver),
    activation=fake_quanitze.AdaptiveWeightNoQuantize.with_args(observer=observer.AdaptiveFixedRangeLowBITActivationObserver))

_QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_DEFAULT] = _QCONFIG_TYPE_TO_DICT[QConfigType.QCONFIG_TYPE_W8T_A8T_SYM_P2]
####################################################################


def get_qconfig(is_qat, backend, qconfig_type=None):
    if qconfig_type not in _QCONFIG_TYPE_TO_DICT:
        raise RuntimeError("Unknown qconfig_type: " + str(qconfig_type))
    #
    qconfig = _QCONFIG_TYPE_TO_DICT[qconfig_type]
    return qconfig


def get_qconfig_mapping(is_qat, backend, qconfig_type=None):
    qconfig = get_qconfig(is_qat, backend, qconfig_type)
    qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig)
    return qconfig_map
