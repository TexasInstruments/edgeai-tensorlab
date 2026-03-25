import numpy as np
import mmcv

from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.transforms_3d import MultiViewWrapper
from mmdet3d.datasets.transforms.transforms_3d import RandomResize3D


@TRANSFORMS.register_module()
class CustomMultiViewWrapper(MultiViewWrapper):
    """Wrap transformation from single-view into multi-view.

    The wrapper processes the images from multi-view one by one. For each
    image, it constructs a pseudo dict according to the keys specified by the
    'process_fields' parameter. After the transformation is finished, desired
    information can be collected by specifying the keys in the 'collected_keys'
    parameter. Multi-view images share the same transformation parameters
    but do not share the same magnitude when a random transformation is
    conducted.

    Args:
        transforms (list[dict]): A list of dict specifying the transformations
            for the monocular situation.
        override_aug_config (bool): flag of whether to use the same aug config
            for multiview image. Defaults to True.
        process_fields (list): Desired keys that the transformations should
            be conducted on. Defaults to ['img', 'cam2img', 'lidar2cam'].
        collected_keys (list): Collect information in transformation
            like rotate angles, crop roi, and flip state. Defaults to
                ['scale', 'scale_factor', 'crop',
                 'crop_offset', 'ori_shape',
                 'pad_shape', 'img_shape',
                 'pad_fixed_size', 'pad_size_divisor',
                 'flip', 'flip_direction', 'rotate'].
        randomness_keys (list): The keys that related to the randomness
            in transformation. Defaults to
                    ['scale', 'scale_factor', 'crop_size', 'flip',
                     'flip_direction', 'photometric_param']
    """

    def __init__(
        self,
        transforms: dict,
        override_aug_config: bool = True,
        process_fields: list = ['img', 'cam2img', 'lidar2cam', 'lidar2img'],
        collected_keys: list = [
            'scale', 'scale_factor', 'crop', 'img_crop_offset', 'ori_shape',
            'pad_shape', 'img_shape', 'pad_fixed_size', 'pad_size_divisor',
            'flip', 'flip_direction', 'rotate'
        ],
        randomness_keys: list = [
            'scale', 'scale_factor', 'crop_size', 'img_crop_offset', 'flip',
            'flip_direction', 'photometric_param'
        ]
    ) -> None:
        super(CustomMultiViewWrapper, self).__init__(transforms, override_aug_config, process_fields, collected_keys, randomness_keys)

@TRANSFORMS.register_module()
class CustomRandomResize3D(RandomResize3D):
    """The difference between RandomResize3D and RandomResize:

    1. Compared to RandomResize, this class would further
        check if scale is already set in results.
    2. During resizing, this class would modify the centers_2d
        and cam2img with ``results['scale']``.
    """

    def _resize_3d(self, results: dict) -> None:
        #super(CustomRandomResize3D)._resize_3d(results)

        """Resize centers_2d and modify camera intrinisc with
        ``results['scale']``."""
        if 'centers_2d' in results:
            results['centers_2d'] *= results['scale_factor'][:2]
        results['cam2img'][0] *= np.array(results['scale_factor'][0])
        results['cam2img'][1] *= np.array(results['scale_factor'][1])
        
        lidar2cam = results['lidar2cam']
        intrinsic = results['cam2img']
        viewpad = np.eye(4)
        viewpad[:3, :3] = intrinsic

        results['lidar2img'] = (viewpad @ lidar2cam)
        


@TRANSFORMS.register_module()
class PadMultiViewImage(BaseTransform):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class NormalizeMultiviewImage(BaseTransform):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str



@TRANSFORMS.register_module()
class CustomPack3DDetInputs(Pack3DDetInputs):
    def __init__(
            self,
            keys: tuple,
            meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                                'depth2img', 'cam2img', 'pad_shape',
                                'scale_factor', 'flip', 'pcd_horizontal_flip',
                                'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                                'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                                'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                                'pcd_rotation_angle', 'lidar_path',
                                'transformation_3d_flow', 'trans_mat',
                                'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                                'cam2global', 'crop_offset', 'img_crop_offset',
                                'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                                'num_ref_frames', 'num_views', 'ego2global',
                                'prev_idx', 'next_idx', 'scene_token', 'can_bus',
                                'gt_bboxes_3d', 'gt_labels_3d')
    ) -> None:
        super(CustomPack3DDetInputs, self).__init__(keys, meta_keys)


