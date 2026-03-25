# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import mmcv
import numpy as np

from einops import rearrange
import torch

from mmengine.fileio import get
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles, LoadAnnotations3D
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class StreamPETRLoadAnnotations3D(LoadAnnotations3D):

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        The only difference is it remove the proceess for
        `ignore_flag`

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        results['gt_bboxes'] = results['ann_info']['bboxes']
        gt_bboxes_ignore = results['ann_info'].get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        results['gt_bboxes_labels'] = results['ann_info']['labels']

    def _load_bboxes_depth(self, results: dict) -> dict:
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """

        results['depths'] = results['ann_info']['depths']
        results['centers_2d'] = results['ann_info']['centers_2d']
        return results


