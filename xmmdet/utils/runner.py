import torch
import mmcv
from mmcv.runner import EpochBasedRunner
from mmcv.runner import OptimizerHook
from pytorch_jacinto_ai import xnn
from .quantize import is_mmdet_quant_module


def is_dataparallel_module(model):
    return isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))


def mmdet_load_checkpoint(model, *args, **kwargs):
    model_orig = model.module if is_mmdet_quant_module(model) else model
    return mmcv.runner.load_checkpoint(model_orig, *args, **kwargs)


def mmdet_save_checkpoint(model, *args, **kwargs):
    model_orig = model.module if is_mmdet_quant_module(model) else model
    mmcv.runner.save_checkpoint(model_orig, *args, **kwargs)


class XMMDetEpochBasedRunner(EpochBasedRunner):
    def __init__(self, *args, **kwargs):
        freeze_range = kwargs.pop('freeze_range', False)
        super().__init__(*args, **kwargs)
        self.freeze_range = freeze_range

    def train(self, data_loader, **kwargs):
        if self.freeze_range:
            # currently we don't have a parameter that indicates whether we are doing QAT or not.
            # Let us do it for all cases of training for the time being.
            freeze_bn_epoch = (self.max_epochs//2)-1
            freeze_range_epoch = (self.max_epochs//2)+1
            if self.epoch > 0 and self.epoch >= freeze_bn_epoch:
                xnn.utils.freeze_bn(self.model)
            #
            if self.epoch > 1 and self.epoch >= freeze_range_epoch:
                xnn.layers.freeze_quant_range(self.model)
            #
        #
        super().train(data_loader, **kwargs)


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


class XMMDetNoOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        pass



