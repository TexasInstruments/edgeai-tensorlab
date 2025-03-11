# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Number

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmdet3d.registry import MODELS
from mmdet3d.utils import OptConfigType
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmengine.utils import is_seq_of
from mmengine.model import stack_batch
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
from mmdet3d.models.data_preprocessors.utils import multiview_img_stack_batch

@MODELS.register_module()
class Petr3DDataPreprocessor(Det3DDataPreprocessor):
    """Points / Image pre-processor for point clouds / vision-only / multi-
    modality 3D detection tasks.
    """

    def collate_data(self, data: dict) -> dict:
        """Copy data to the target device and perform normalization, padding
        and bgr2rgb conversion and stack based on ``BaseDataPreprocessor``.

        Collates the data sampled from dataloader into a list of dict and list
        of labels, and then copies tensor to the target device.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore

        if 'img' in data['inputs']:
            _batch_imgs = data['inputs']['img']
            # Process data with `pseudo_collate`.
            if is_seq_of(_batch_imgs, torch.Tensor):
                batch_imgs = []
                img_dim = _batch_imgs[0].dim()
                for _batch_img in _batch_imgs:
                    if img_dim == 3:  # standard img
                        _batch_img = self.preprocess_img(_batch_img)
                    elif img_dim == 4:
                        _batch_img = [
                            self.preprocess_img(_img) for _img in _batch_img
                        ]

                        _batch_img = torch.stack(_batch_img, dim=0)
                    elif img_dim == 5:
                        _batch_img = [
                            self.preprocess_img(_img) for _img in _batch_img[0]
                        ]
                        # Do not unqueeze here, intetead do after padding
                        #_batch_img = torch.stack(_batch_img, dim=0).unsqueeze(dim=0)
                        _batch_img = torch.stack(_batch_img, dim=0)

                    batch_imgs.append(_batch_img)

                # Pad and stack Tensor.
                if img_dim == 3:
                    batch_imgs = stack_batch(batch_imgs, self.pad_size_divisor,
                                             self.pad_value)
                elif img_dim == 4:
                    batch_imgs = multiview_img_stack_batch(
                        batch_imgs, self.pad_size_divisor, self.pad_value)
                elif img_dim == 5:
                    temp_imgs = multiview_img_stack_batch(
                        batch_imgs, self.pad_size_divisor, self.pad_value)

                    # Somehow the following codes do not work
                    #     for batch_img in batch_imgs:
                    #         batch_img = batch_img.unsqueeze(dim=0)
                    # So create batch_imgs from temp_imgs.
                    batch_imgs = []
                    for temp_img in temp_imgs:
                        batch_imgs.append(temp_img.unsqueeze(dim=0))

            # Process data with `default_collate`.
            elif isinstance(_batch_imgs, torch.Tensor):
                assert _batch_imgs.dim() == 4, (
                    'The input of `ImgDataPreprocessor` should be a NCHW '
                    'tensor or a list of tensor, but got a tensor with '
                    f'shape: {_batch_imgs.shape}')
                if self._channel_conversion:
                    _batch_imgs = _batch_imgs[:, [2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_imgs = _batch_imgs.float()
                if self._enable_normalize:
                    _batch_imgs = (_batch_imgs - self.mean) / self.std
                h, w = _batch_imgs.shape[2:]
                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                batch_imgs = F.pad(_batch_imgs, (0, pad_w, 0, pad_h),
                                   'constant', self.pad_value)
            else:
                raise TypeError(
                    'Output of `cast_data` should be a list of dict '
                    'or a tuple with inputs and data_samples, but got '
                    f'{type(data)}: {data}')

            data['inputs']['imgs'] = batch_imgs

        data.setdefault('data_samples', None)

        return data


