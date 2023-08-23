import torch
import mmcv
from mmcv.runner.builder import RUNNERS
from mmcv.runner import EpochBasedRunner
from mmcv.runner import OptimizerHook, HOOKS, Hook
from torchvision.edgeailite import xnn
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
    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

    def val(self, *args, **kwargs):
        super().val(*args, **kwargs)

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
        # #offset_epoch = runner.max_epochs // 10
        
        # #if offset_epoch<= 0:
        # #    offset_epoch = 1
        
        # offset_epoch = 1

        # freeze_bn_epoch = (runner.max_epochs // 2) - offset_epoch
        # freeze_range_epoch = (runner.max_epochs // 2) + offset_epoch
        # if runner.epoch >= 1 and runner.epoch >= freeze_bn_epoch:
        #     xnn.utils.print_once('Freezing BN')
        #     xnn.utils.freeze_bn(runner.model)
        # #
        # if runner.epoch >= 2 and runner.epoch >= freeze_range_epoch:
        #     xnn.utils.print_once('Freezing Activation ranges')
        #     xnn.layers.freeze_quant_range(runner.model)
        # #
        # this freezing is now done inside the QuantTrainModule()
        # so this hook is not required
        pass
