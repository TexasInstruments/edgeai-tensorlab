# Copyright (c) 2018-2021, Texas Instruments
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

# edgeai_torchmodelopt is installed by the setup of https://github.com/TexasInstruments/edgeai-modeloptimization#torchmodelopt
import edgeai_torchmodelopt


class XMMDetQuantCalibrateModule(edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantCalibrateModule):
    def __init__(self, model,  dummy_input, *args, forward_analyze_method='forward_dummy', **kwargs):
        super().__init__(model, dummy_input, *args, forward_analyze_method=forward_analyze_method, **kwargs)

    def forward(self, img, *args, **kwargs):
        return super().forward(img, *args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.module.train_step(*args, **kwargs)

    def val_step(self, *args, **kwargs):
        return self.module.val_step(*args, **kwargs)


class XMMDetQuantTrainModule(edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantTrainModule):
    def __init__(self, model,  dummy_input, *args, forward_analyze_method='forward_dummy', **kwargs):
        super().__init__(model, dummy_input, *args,bitwidth_weights=8, bitwidth_activations=8, forward_analyze_method=forward_analyze_method, **kwargs)

    def forward(self, *args, return_loss=True, **kwargs):
        return super().forward(return_loss, *args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.module.train_step(*args, **kwargs)

    def val_step(self, *args, **kwargs):
        return self.module.val_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.module.test_step(*args, **kwargs)


class XMMDetQuantTestModule(edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantTestModule):
    def __init__(self, model,  dummy_input, *args, forward_analyze_method='forward_dummy', **kwargs):
        super().__init__(model, dummy_input, *args, forward_analyze_method=forward_analyze_method, **kwargs)

    def forward(self, img, *args, **kwargs):
        return super().forward(img, *args, **kwargs)

    def val_step(self, *args, **kwargs):
        return self.module.val_step(*args, **kwargs)


def is_mmdet_quant_module(model):
    return isinstance(model, (XMMDetQuantCalibrateModule, XMMDetQuantTrainModule, XMMDetQuantTestModule))