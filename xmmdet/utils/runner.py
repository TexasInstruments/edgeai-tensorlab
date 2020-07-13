import torch
import mmcv
from mmcv.runner import EpochBasedRunner
from mmcv.runner import OptimizerHook
from .quantize import is_mmdet_quant_module


def is_dataparallel_module(model):
    return isinstance(model, torch.nn.DataParallel)


def mmdet_load_checkpoint(model, *args, **kwargs):
    model_orig = model.module if is_mmdet_quant_module(model) else model
    return mmcv.runner.load_checkpoint(model_orig, *args, **kwargs)


def mmdet_save_checkpoint(model, *args, **kwargs):
    model_orig = model.module if is_mmdet_quant_module(model) else model
    mmcv.runner.save_checkpoint(model_orig, *args, **kwargs)


class XMMDetEpochBasedRunner(EpochBasedRunner):
    def _get_model_orig(self):
        model_orig = self.model
        is_model_orig = True
        if is_mmdet_quant_module(model_orig):
            model_orig = model_orig.module
            is_model_orig = False
        #
        if is_dataparallel_module(model_orig):
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


class XMMDetNoOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        pass



