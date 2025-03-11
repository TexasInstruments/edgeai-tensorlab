
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

import functools

from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from edgeai_torchmodelopt import xmodelopt


all = ['QATFxHook', 'PTQFxHook']

__all__ = all


class QuantBaseHook(Hook):
    pass


@HOOKS.register_module()
class QATFxHook(QuantBaseHook):
    def __init__(self, is_qat=True):
        self.is_qat = is_qat

    def _quant_init(self, runner) -> None:
        cfg = runner.cfg
        if not cfg.quantization:
            return

        # wrap the model
        if cfg.quantization == xmodelopt.quantization.QuantizationVersion.QUANTIZATION_V1:
            is_wrapped = False
            if is_model_wrapper(runner.model):
                runner.model = runner.model.module
                is_wrapped = True
            #
            test_loader = runner.build_dataloader(runner._test_dataloader)
            example_input = next(iter(test_loader))
            quant_wrapper = xmodelopt.quantization.v1.QuantTrainModule if self.is_qat else \
                xmodelopt.quantization.v1.QuantCalibrateModule
            runner.model = runner.model.quant_init(quant_wrapper, dummy_input=example_input,
                                                   total_epochs=runner.max_epochs)
            if is_wrapped:
                runner.model = runner.wrap_model(runner.cfg.get('model_wrapper_cfg'), runner.model)
            #
        elif cfg.quantization == xmodelopt.quantization.QuantizationVersion.QUANTIZATION_V2:
            is_wrapped = False
            if is_model_wrapper(runner.model):
                runner.model = runner.model.module
                is_wrapped = True
            #
            if hasattr(runner.model, 'quant_init'):
                print('wrapping the model to prepare for quantization')
                # quant_wrapper = xnn.utils.partialclass(xmodelopt.quantization.v2.QATFxModule, self.is_qat)
                quant_wrapper = functools.partial(xmodelopt.quantization.v2.quant_fx_func.init, is_qat=self.is_qat)
                runner.model = runner.model.quant_init(quant_wrapper, total_epochs=runner.max_epochs)
            else:
                raise RuntimeError(f'quant_init method is not supported for {type(runner.model)}')
            #
            if is_wrapped:
                runner.model = runner.wrap_model(runner.cfg.get('model_wrapper_cfg'), runner.model)
            #

    def before_train(self, runner) -> None:
        return self._quant_init(runner)

    def before_val(self, runner) -> None:
        return self._quant_init(runner)


@HOOKS.register_module()
class PTQFxHook(QATFxHook):
    def __init__(self, is_qat=False):
        super().__init__(is_qat=is_qat)
