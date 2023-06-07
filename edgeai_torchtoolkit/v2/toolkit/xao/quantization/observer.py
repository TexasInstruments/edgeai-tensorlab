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

import random
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.ao.quantization as quantization
from torch.ao.quantization.utils import check_min_max_valid
import edgeai_torchtoolkit.v1.toolkit.xnn as xnn
from . import qsettings


# modified from:  UniformQuantizationObserverBase:_calculate_qparams()
# https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/observer.py#L294
# key modifications: (1) option for making scale as a power-of-2 value
#                    (2) use full range for unsigned tensor in symmetric mode
@torch.jit.export
def _calculate_qparams_accurate(
    min_val: torch.Tensor, max_val: torch.Tensor,
    quant_min: int, quant_max: int,
    qscheme: Any, dtype: Any, eps: Any,
    has_customized_qrange: bool,
    power2: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates the quantization parameters, given min and max
    value tensors. Works for both per tensor and per channel cases

    Args:
        min_val: Minimum values per channel
        max_val: Maximum values per channel

    Returns:
        scales: Scales tensor of shape (#channels,)
        zero_points: Zero points tensor of shape (#channels,)
    """
    if not check_min_max_valid(min_val, max_val):
        return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

    quant_min, quant_max = quant_min, quant_max
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    device = min_val_neg.device
    scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    unsigned_tensor = float(min_val) >= 0
    signed_tensor = (not unsigned_tensor)

    if qscheme == torch.per_tensor_symmetric or qscheme == torch.per_channel_symmetric:
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        if qsettings.USE_FULL_RANGE_FOR_SYMMETRIC and unsigned_tensor:
            scale = max_val_pos / float(quant_max - quant_min)
        else:
            # this is the original method used in torch observers
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
        #
        scale = torch.max(scale, eps)
        if power2:
            scale = xnn.layers.functional.ceil2_g(scale)
        #
        if signed_tensor and dtype == torch.quint8:
            if has_customized_qrange:
                # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                zero_point = zero_point.new_full(zero_point.size(), (quant_min + quant_max + 1)//2) #128
            else:
                zero_point = zero_point.new_full(zero_point.size(), (quant_max + 1)//2) #128
            #
        elif unsigned_tensor and dtype == torch.qint8:
            zero_point = zero_point.new_full(zero_point.size(), quant_min) #-128
        #
    elif qscheme == torch.per_channel_affine_float_qparams:
        scale = (max_val - min_val) / float(quant_max - quant_min)
        scale = torch.where(scale > eps, scale, torch.ones_like(scale))
        if power2:
            scale = xnn.layers.functional.ceil2_g(scale)
        #
        # We use the quantize function
        # xq = Round(Xf * inv_scale + zero_point),
        # setting zero_point to (-1 * min *inv_scale) we get
        # Xq = Round((Xf - min) * inv_scale)
        zero_point = -1 * min_val / scale
    else:
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, eps)
        if power2:
            scale = xnn.layers.functional.ceil2_g(scale)
        #
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

    # For scalar values, cast them to Tensors of size 1 to keep the shape
    # consistent with default values in FakeQuantize.
    if len(scale.shape) == 0:
        # TODO: switch to scale.item() after adding JIT support
        scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
    if len(zero_point.shape) == 0:
        # TODO: switch to zero_point.item() after adding JIT support
        zero_point = torch.tensor(
            [int(zero_point)], dtype=zero_point.dtype, device=device
        )
        if qscheme == torch.per_channel_affine_float_qparams:
            zero_point = torch.tensor(
                [float(zero_point)], dtype=zero_point.dtype, device=device
            )

    return scale, zero_point


class MovingAverageMinMaxObserverPower2(quantization.MovingAverageMinMaxObserver):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        if qsettings.COMPUTE_ACCURATE_QPARAMS:
            scale, zero_point = _calculate_qparams_accurate(min_val=min_val, max_val=max_val,
                                                  quant_min=self.quant_min, quant_max=self.quant_max,
                                                  qscheme=self.qscheme, dtype=self.dtype, eps=self.eps,
                                                  has_customized_qrange=self.has_customized_qrange, power2=True)
        else:
            scale, zero_point = super()._calculate_qparams(min_val, max_val)
            scale = xnn.layers.functional.ceil2_g(scale)
        #
        return scale, zero_point


class HistogramObserverPower2(quantization.HistogramObserver):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def forward(self, x_orig):
    #     # still slow, even if we do this
    #     fast_mode = True #False
    #     fast_stride = 2
    #     fast_stride2 = fast_stride * 2
    #     if fast_mode and len(x_orig.size()) == 4 and (x_orig.size(2) > fast_stride2) and (x_orig.size(3) > fast_stride2):
    #         r_start = random.randint(0, fast_stride - 1)
    #         c_start = random.randint(0, fast_stride - 1)
    #         x_new = x_orig[..., r_start::fast_stride, c_start::fast_stride]
    #         x_new = super().forward(x_new)
    #     else:
    #         x_new = super().forward(x_orig)
    #     #
    #     return x_new

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        if qsettings.COMPUTE_ACCURATE_QPARAMS:
            scale, zero_point = _calculate_qparams_accurate(min_val=min_val, max_val=max_val,
                                                  quant_min=self.quant_min, quant_max=self.quant_max,
                                                  qscheme=self.qscheme, dtype=self.dtype, eps=self.eps,
                                                  has_customized_qrange=self.has_customized_qrange, power2=True)
        else:
            scale, zero_point = super()._calculate_qparams(min_val, max_val)
            scale = xnn.layers.functional.ceil2_g(scale)
        #
        return scale, zero_point
