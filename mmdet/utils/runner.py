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

import torch
import mmcv
from mmcv.runner.builder import RUNNERS
from mmcv.runner import EpochBasedRunner
from mmcv.runner import OptimizerHook, HOOKS, Hook

from edgeai_xvision import xnn
from .quantize import is_mmdet_quant_module


def is_dataparallel_module(model):
    return isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))


def mmdet_load_checkpoint(model, *args, **kwargs):
    model_orig = model.module if is_mmdet_quant_module(model) else model
    return mmcv.runner.load_checkpoint(model_orig, *args, **kwargs)


def mmdet_save_checkpoint(model, *args, **kwargs):
    model_orig = model.module if is_mmdet_quant_module(model) else model
    mmcv.runner.save_checkpoint(model_orig, *args, **kwargs)


# replace the EpochBasedRunner
@RUNNERS.register_module('EpochBasedRunner', force=True)
class XMMDetEpochBasedRunner(EpochBasedRunner):
    def _get_model_orig(self):
        model_orig = self.model
        is_model_orig = True
        if is_dataparallel_module(model_orig):
            model_orig = model_orig.module
            is_model_orig = False
        #
        if is_mmdet_quant_module(model_orig):
            model_orig = model_orig.module
            is_model_orig = False
        #
        return model_orig, is_model_orig

    def save_checkpoint(self, *args, **kwargs):
        model_backup = self.model
        self.model, is_model_orig = self._get_model_orig()
        super().save_checkpoint(*args, **kwargs)
        if not is_model_orig:
            self.model = model_backup
        #

    def load_checkpoint(self, *args, **kwargs):
        model_backup = self.model
        self.model, is_model_orig = self._get_model_orig()
        checkpoint = super().load_checkpoint(*args, **kwargs)
        if not is_model_orig:
            self.model = model_backup
        #
        return checkpoint

    def resume(self, *args, **kwargs):
        model_backup = self.model
        if is_mmdet_quant_module(self.model):
            self.model = self.model.module
        #
        super().resume(*args, **kwargs)
        if is_mmdet_quant_module(self.model):
            self.model = model_backup
        #


@HOOKS.register_module()
class XMMDetNoOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        pass


@HOOKS.register_module()
class FreezeRangeHook(Hook):
    def before_train_epoch(self, runner):
        freeze_bn_epoch = (runner.max_epochs // 2) - 1
        freeze_range_epoch = (runner.max_epochs // 2) + 1
        if runner.epoch >= 1 and runner.epoch >= freeze_bn_epoch:
            xnn.utils.print_once('Freezing BN')
            xnn.utils.freeze_bn(runner.model)
        #
        if runner.epoch >= 2 and runner.epoch >= freeze_range_epoch:
            xnn.utils.print_once('Freezing Activation ranges')
            xnn.layers.freeze_quant_range(runner.model)
        #

