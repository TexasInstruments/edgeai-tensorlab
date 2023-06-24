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
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qat_qconfig_mapping
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, \
    FakeQuantize, FusedMovingAvgObsFakeQuantize

from . import observer
from . import fake_quanitze

try:
    # this is not part of torch 2.0.1 release - so provide an alternate implementation for now
    from torch.ao.quantization.qconfig_mapping import _get_default_qconfig_mapping_with_default_qconfig
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
    # default is same as QCONFIG_TYPE_8BIT_PER_TENSOR_WEIGHT
    QCONFIG_TYPE_DEFAULT = "DEFAULT"
    QCONFIG_TYPE_8BIT_PER_TENSOR_WEIGHT = "8WT8AT"
    QCONFIG_TYPE_8BIT_PER_CHAN_WEIGHT = "8WC8AT"
    QCONFIG_TYPE_8BIT_PER_TENSOR_WEIGHT_SYMM_P2 = "8WT8AT_SYM_P2"
    QCONFIG_TYPE_8BIT_PER_CHAN_WEIGHT_SYMM_P2 = "8WC8AT_SYM_P2"
    QCONFIG_TYPE_4BIT_PER_CHAN_WEIGHT = "4WC4AT"
    QCONFIG_TYPE_4W_8A_PER_CHAN_WEIGHT = "4WC8AT"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


def get_default_qconfig_mapping(is_qat, backend, qconfig_type=None):
    # it is possible to use a non qat qconfig such as torch.ao.quantization.get_default_qconfig_mapping(backend)
    # however qat qconfig which does fake quantization may be better even for PTQ cases.
    # torch.ao.quantization.get_default_qat_qconfig_mapping(backend)
    if qconfig_type in (QConfigType.QCONFIG_TYPE_8BIT_PER_TENSOR_WEIGHT, QConfigType.QCONFIG_TYPE_DEFAULT):
        qconfig_map = get_default_qat_qconfig_mapping(backend)
    elif qconfig_type in (QConfigType.QCONFIG_TYPE_8BIT_PER_CHAN_WEIGHT,):
        # FusedMovingAvgObsFakeQuantize will not use calculate_qparams() during forward (only during convert)
        # it directly calls torch.fused_moving_avg_obs_fake_quant() which implements everything inside it
        # so use FakeQuantize here as we need to override calculate_qparams()
        activation_fake_quant = \
            fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver,
                                                                   quant_min=0,
                                                                   quant_max=255,
                                                                   reduce_range=False)
        weight_fake_quant = \
            fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePerChannelWeightObserver,
                                                               quant_min=-128,
                                                               quant_max=127,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric)
        qconfig = QConfig(activation=activation_fake_quant,
                          weight=weight_fake_quant)
        qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig)
    elif qconfig_type in (QConfigType.QCONFIG_TYPE_4BIT_PER_CHAN_WEIGHT,):
        # FusedMovingAvgObsFakeQuantize will not use calculate_qparams() during forward (only during convert)
        # it directly calls torch.fused_moving_avg_obs_fake_quant() which implements everything inside it
        # so use FakeQuantize here as we need to override calculate_qparams()
        activation_fake_quant = \
            fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveLowBITActivationObserver,
                                                                   quant_min=0,
                                                                   quant_max=15,
                                                                   reduce_range=False)
        weight_fake_quant = \
            fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptiveLowBITPerChannelWeightObserver,
                                                               quant_min=-8,
                                                               quant_max=7,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric)
        qconfig = QConfig(activation=activation_fake_quant,
                          weight=weight_fake_quant)
        qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig)
    elif qconfig_type in (QConfigType.QCONFIG_TYPE_4W_8A_PER_CHAN_WEIGHT,):
        # FusedMovingAvgObsFakeQuantize will not use calculate_qparams() during forward (only during convert)
        # it directly calls torch.fused_moving_avg_obs_fake_quant() which implements everything inside it
        # so use FakeQuantize here as we need to override calculate_qparams()
        activation_fake_quant = \
            fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptiveActivationObserver,
                                                                   quant_min=0,
                                                                   quant_max=255,
                                                                   reduce_range=False)
        weight_fake_quant = \
            fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptiveLowBITPerChannelWeightObserver,
                                                               quant_min=-8,
                                                               quant_max=7,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric)
        qconfig = QConfig(activation=activation_fake_quant,
                          weight=weight_fake_quant)
        qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig)
    elif qconfig_type in (QConfigType.QCONFIG_TYPE_8BIT_PER_TENSOR_WEIGHT_SYMM_P2,):
        # FusedMovingAvgObsFakeQuantize will not use calculate_qparams() during forward (only during convert)
        # it directly calls torch.fused_moving_avg_obs_fake_quant() which implements everything inside it
        # so use FakeQuantize here as we need to override calculate_qparams()
        activation_fake_quant = \
            fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptivePower2ActivationObserver,
                                                                   quant_min=0,
                                                                   quant_max=255,
                                                                   dtype=torch.quint8,
                                                                   qscheme=torch.per_tensor_symmetric,
                                                                   reduce_range=False)
        weight_fake_quant = \
            fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePower2WeightObserver,
                                                               quant_min=-128,
                                                               quant_max=127,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_tensor_symmetric)
        qconfig = QConfig(activation=activation_fake_quant,
                          weight=weight_fake_quant)
        qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig)
    elif qconfig_type in (QConfigType.QCONFIG_TYPE_8BIT_PER_CHAN_WEIGHT_SYMM_P2,):
        # FusedMovingAvgObsFakeQuantize will not use calculate_qparams() during forward (only during convert)
        # it directly calls torch.fused_moving_avg_obs_fake_quant() which implements everything inside it
        # so use FakeQuantize here as we need to override calculate_qparams()
        activation_fake_quant = \
            fake_quanitze.AdaptiveActivationFakeQuantize.with_args(observer=observer.AdaptivePower2ActivationObserver,
                                                                   quant_min=0,
                                                                   quant_max=255,
                                                                   dtype=torch.quint8,
                                                                   qscheme=torch.per_tensor_symmetric,
                                                                   reduce_range=False)
        weight_fake_quant = \
            fake_quanitze.AdaptiveWeightFakeQuantize.with_args(observer=observer.AdaptivePower2PerChannelWeightObserver,
                                                               quant_min=-128,
                                                               quant_max=127,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric)
        qconfig = QConfig(activation=activation_fake_quant,
                          weight=weight_fake_quant)
        qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig)
    else:
        raise RuntimeError("Unknown qconfig_type: " + str(qconfig_type))
    return qconfig_map
