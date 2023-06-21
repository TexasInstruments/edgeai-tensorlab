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
from torch.ao.quantization import default_fused_per_channel_wt_fake_quant, default_embedding_fake_quant_4bit
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qat_qconfig_mapping

try:
    # this is not part of torch 2.0.1 release
    from torch.ao.quantization.qconfig_mapping import _get_default_qconfig_mapping_with_default_qconfig
    warnings.warn("could not find _get_default_qconfig_mapping_with_default_qconfig in torch.ao.quantization.qconfig_mapping")
except:
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
    QCONFIG_TYPE_DEFAULT = 0
    QCONFIG_TYPE_PER_CHANNEL_WEIGHTS = 1
    QCONFIG_TYPE_SYMMETRIC_POWER2 = 2
    QCONFIG_TYPE_4BIT = 4

# the default qconfig
def get_default_qconfig_mapping(is_qat, backend, qconfig_type=None):
    # it is possible to use a non qat qconfig such as torch.ao.quantization.get_default_qconfig_mapping(backend)
    # however qat qconfig which does fake quantization may be better even for PTQ cases.
    # torch.ao.quantization.get_default_qat_qconfig_mapping(backend)
    if qconfig_type in (None, qconfig_type == QConfigType.QCONFIG_TYPE_DEFAULT):
        qconfig_map = get_default_qat_qconfig_mapping(backend)
    elif qconfig_type == QConfigType.QCONFIG_TYPE_PER_CHANNEL_WEIGHTS:
        qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                             quant_min=0,
                                                                             quant_max=255,
                                                                             reduce_range=False),
                          weight=default_fused_per_channel_wt_fake_quant)
        qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig)
    elif qconfig_type == QConfigType.QCONFIG_TYPE_4BIT:
        activation_fake_quant = \
            FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                     quant_min=0,
                                                     quant_max=15,
                                                     reduce_range=False)
        per_channel_weight_fake_quant_4bit = \
            FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                      quant_min=-16,
                                                      quant_max=15,
                                                      dtype=torch.qint8,
                                                      qscheme=torch.per_channel_symmetric)
        qconfig = QConfig(activation=activation_fake_quant,
                          weight=per_channel_weight_fake_quant_4bit)
        qconfig_map = _get_default_qconfig_mapping_with_default_qconfig(is_qat, backend, qconfig)
    else:
        RuntimeError("Unknown qconfig_type: " + str(qconfig_type))
    return qconfig_map
