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

import torch
import torch.ao.quantization as quantization
from ... import xnn
from . import qsettings


class MovingAverageMinMaxObserverPower2(quantization.MovingAverageMinMaxObserver):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.jit.export
    def calculate_qparams(self):
        # for unsigned tensor, the implementation in base class does not use the full range
        if qsettings.USE_FULL_RANGE_FOR_SYMMETRIC and \
            self.qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric) and float(self.min_val) >= 0:
            scale, zero_point = self._calculate_qparams_unsigned()
        else:
            scale, zero_point = super().calculate_qparams()
        #
        scale = xnn.layers.functional.ceil2_g(scale)
        return scale, zero_point

    def _calculate_qparams_unsigned(self):
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(self.min_val, torch.zeros_like(self.min_val))
        max_val_pos = torch.max(self.max_val, torch.zeros_like(self.max_val))
        max_val_pos = torch.max(-min_val_neg, max_val_pos)

        device = min_val_neg.device
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
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
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor(
                    [float(zero_point)], dtype=zero_point.dtype, device=device
                )

        return scale, zero_point


class HistogramObserverPower2(quantization.HistogramObserver):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.jit.export
    def calculate_qparams(self):
        # for unsigned tensor, the implementation in base class does not use the full range
        if qsettings.USE_FULL_RANGE_FOR_SYMMETRIC and \
            self.qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric) and float(self.min_val) >= 0:
            scale, zero_point = self._calculate_qparams_unsigned()
        else:
            scale, zero_point = super().calculate_qparams()
        #
        scale = xnn.layers.functional.ceil2_g(scale)
        return scale, zero_point

    def _calculate_qparams_unsigned(self):
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(self.min_val, torch.zeros_like(self.min_val))
        max_val_pos = torch.max(self.max_val, torch.zeros_like(self.max_val))
        max_val_pos = torch.max(-min_val_neg, max_val_pos)

        device = min_val_neg.device
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
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
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor(
                    [float(zero_point)], dtype=zero_point.dtype, device=device
                )

        return scale, zero_point
