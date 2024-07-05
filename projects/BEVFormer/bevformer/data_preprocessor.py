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
class BEVFormer3DDataPreprocessor(Det3DDataPreprocessor):
    """Points / Image pre-processor for point clouds / vision-only / multi-
    modality 3D detection tasks.

    It provides the data pre-processing as follows

    - Collate and move image and point cloud data to the target device.

    - 1) For image data:

      - Pad images in inputs to the maximum size of current batch with defined
        ``pad_value``. The padding size can be divisible by a defined
        ``pad_size_divisor``.
      - Stack images in inputs to batch_imgs.
      - Convert images in inputs from bgr to rgb if the shape of input is
        (3, H, W).
      - Normalize images in inputs with defined std and mean.
      - Do batch augmentations during training.

    - 2) For point cloud data:

      - If no voxelization, directly return list of point cloud data.
      - If voxelization is applied, voxelize point cloud according to
        ``voxel_type`` and obtain ``voxels``.

    Args:
        voxel (bool): Whether to apply voxelization to point cloud.
            Defaults to False.
        voxel_type (str): Voxelization type. Two voxelization types are
            provided: 'hard' and 'dynamic', respectively for hard voxelization
            and dynamic voxelization. Defaults to 'hard'.
        voxel_layer (dict or :obj:`ConfigDict`, optional): Voxelization layer
            config. Defaults to None.
        batch_first (bool): Whether to put the batch dimension to the first
            dimension when getting voxel coordinates. Defaults to True.
        max_voxels (int, optional): Maximum number of voxels in each voxel
            grid. Defaults to None.
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be divisible by
            ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic segmentation
            maps. Defaults to 255.
        bgr_to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): Whether to convert image from RGB to BGR.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
        batch_augments (List[dict], optional): Batch-level augmentations.
            Defaults to None.
    """

    def __init__(self,
                 voxel: bool = False,
                 voxel_type: str = 'hard',
                 voxel_layer: OptConfigType = None,
                 batch_first: bool = True,
                 max_voxels: Optional[int] = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: bool = False,
                 batch_augments: Optional[List[dict]] = None) -> None:
        super(BEVFormer3DDataPreprocessor, self).__init__(
            voxel=voxel,
            voxel_type=voxel_type,
            voxel_layer=voxel_layer,
            batch_first=batch_first,
            max_voxels=max_voxels,
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments)

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
                        # For training with queue
                        Q, N, C, H, W = _batch_img.shape
                        _batch_img = _batch_img.reshape(Q*N, C, H, W)
                        _batch_img = [
                            self.preprocess_img(_img) for _img in _batch_img
                        ]
                        _batch_img = torch.stack(_batch_img, dim=0)

                    batch_imgs.append(_batch_img)

                # Pad and stack Tensor.
                if img_dim == 3:
                    batch_imgs = stack_batch(batch_imgs, self.pad_size_divisor,
                                             self.pad_value)
                elif img_dim == 4 or img_dim == 5:
                    batch_imgs = multiview_img_stack_batch(
                        batch_imgs, self.pad_size_divisor, self.pad_value)

                    if img_dim == 5:
                        # reshpae batch_imgs
                        _, _, C, H, W = batch_imgs.shape
                        batch_imgs = batch_imgs.reshape(-1, Q, N, C, H, W)


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

    def _get_pad_shape(self, data: dict) -> List[Tuple[int, int]]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        # rewrite `_get_pad_shape` for obtaining image inputs.        

        #if isinstance(data['inputs']['img'], list):
        #    _batch_inputs = data['inputs']['img'][0]
        #else:
        #    _batch_inputs = data['inputs']['img']

        _batch_inputs = data['inputs']['img']

        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                if ori_input.dim() == 5:
                    # mean temporal multiview input, select one of the
                    # image to calculate the pad shape
                    ori_input = ori_input[0, 0]
                elif ori_input.dim() == 4:
                    # mean multiview input, select one of the
                    # image to calculate the pad shape
                    ori_input = ori_input[0]

                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a list of dict '
                            'or a tuple with inputs and data_samples, but got '
                            f'{type(data)}: {data}')
        return batch_pad_shape


    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                    if training is True:
                        for _, idx in enumerate(data_sample.metainfo_queue):
                            data_sample.metainfo_queue[idx]['pad_shape'] = pad_shape
                            data_sample.metainfo_queue[idx]['batch_input_shape'] = batch_input_shape

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        # update img_shape        
        for data_sample in data_samples:
            for i in range(imgs.shape[-4]):
                data_sample.img_shape[i] = (imgs.shape[-2], imgs.shape[-1], imgs.shape[-3])

                if training is True:
                    for _, idx in enumerate(data_sample.metainfo_queue):
                        data_sample.metainfo_queue[idx]['img_shape'][i] = \
                            (imgs.shape[-2], imgs.shape[-1], imgs.shape[-3])

        # should we return batch_metainfo?
        return {'inputs': batch_inputs, 'data_samples': data_samples}

