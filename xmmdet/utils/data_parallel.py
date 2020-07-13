# modified from: https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_parallel.py
# Copyright (c) Open-MMLab. All rights reserved.
from itertools import chain
from warnings import warn

from mmcv.parallel import MMDataParallel

class XMMDetDataParallel(MMDataParallel):

    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.train_step(*inputs, **kwargs)

        if len(self.device_ids) > 1:
            warn('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             'instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.val_step(*inputs, **kwargs)

        if len(self.device_ids) > 1:
            warn('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             'instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.val_step(*inputs[0], **kwargs[0])
