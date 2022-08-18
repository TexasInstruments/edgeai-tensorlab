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


class MovingAverageMinMaxObserverPower2(quantization.MovingAverageMinMaxObserver):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        x_orig = super().forward(x_orig)
        self.min_val = xnn.layers.functional.ceil2_g(self.min_val)
        self.max_val = xnn.layers.functional.ceil2_g(self.max_val)
        return x_orig

    # @torch.jit.export
    # def calculate_qparams(self):
    #     scale, zero_point = super().calculate_qparams()
    #     scale = xnn.layers.functional.ceil2_g(scale)
    #     return scale, zero_point


class MovingAveragePerChannelMinMaxObserverPower2(quantization.MovingAveragePerChannelMinMaxObserver):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        x_orig = super().forward(x_orig)
        self.min_val = xnn.layers.functional.ceil2_g(self.min_val)
        self.max_val = xnn.layers.functional.ceil2_g(self.max_val)
        return x_orig

    # @torch.jit.export
    # def calculate_qparams(self):
    #     scale, zero_point = super().calculate_qparams()
    #     scale = xnn.layers.functional.ceil2_g(scale)
    #     return scale, zero_point
