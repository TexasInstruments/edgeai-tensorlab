# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import mmcv

from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D


@TRANSFORMS.register_module()
class RandomScaleImageMultiViewImage(BaseTransform):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]
        results['ori_shape'] = [img.shape for img in results['img']]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        results['img_shape'] = [img.shape for img in results['img']]

        ####
        scale_factor = np.eye(3)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale

        for i in range(len(results['cam2img'])):
            results['cam2img'][i] = scale_factor @ results['cam2img'][i]

            lidar2cam = results['lidar2cam'][i]
            intrinsic = results['cam2img'][i]
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic

            results['lidar2img'][i] = (viewpad @ lidar2cam)
        ####


        #scale_factor = np.eye(4)
        #scale_factor[0, 0] *= rand_scale
        #scale_factor[1, 1] *= rand_scale
        #results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
        #                  enumerate(results['img'])]
        #lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        #results['lidar2img'] = lidar2img
        #results['img_shape'] = [img.shape for img in results['img']]
        #results['ori_shape'] = [img.shape for img in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@TRANSFORMS.register_module()
class CustomMultiScaleFlipAug3D(MultiScaleFlipAug3D):
    def transform(self, results):
        results = super().transform(results)
        if isinstance(results, list):
            results = results[0]

        return results

