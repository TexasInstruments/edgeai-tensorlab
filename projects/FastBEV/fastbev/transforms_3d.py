# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import random as prandom
import warnings
from PIL import Image
import string

from typing import List, Optional, Sequence, Tuple, Union

from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.transforms_3d import RandomFlip3D, GlobalRotScaleTrans

@TRANSFORMS.register_module()
class RandomAugImageMultiViewImage(BaseTransform):
    """Random scale the image
    Args:
        scales
    """
    def __init__(self, data_config=None, is_train=True, is_debug=False, is_exit=False, tmp='./figs'):
        self.data_config = data_config
        self.is_train = is_train
        self.is_debug = is_debug
        self.is_exit = is_exit
        self.tmp = tmp

    def random_id(self, N=8, seed=None):
        if seed is not None:
            prandom.seed(seed)
        return ''.join(prandom.choice(string.ascii_uppercase + string.digits) for _ in range(N))

    def transform(self, results, fix=''):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        aug_imgs = []
        aug_extrinsics = []
        for cam_id, img in enumerate(results['img']):
            pil_img = Image.fromarray(img, mode='RGB')
            resize, resize_dims, crop, flip, rotate, pad = self.sample_augmentation(
                H=pil_img.height,
                W=pil_img.width,
            )
            post_pil_img, post_rot, post_tran = self.img_transform(
                pil_img, torch.eye(2), torch.zeros(2),
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
                pad=pad
            )
            aug_imgs.append(np.asarray(post_pil_img))
            aug_extrinsics.append(
                self.rts2proj(results['lidar2img']['lidar2img_aug'][cam_id], post_rot, post_tran)
            )
        results['img'] = aug_imgs
        results['lidar2img']['extrinsic'] = aug_extrinsics
        results['img_shape'] = [img.shape for img in results['img']]

        if self.is_exit:
            exit()
        return results

    def sample_augmentation(self, H, W):
        if self.is_train:
            fH, fW = self.data_config['input_size']  # (640, 1600),
            resize = float(fW)/float(W)  # 1600 / 1600
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))  # 900 1600

            newW, newH = resize_dims  # 1600 900
            crop_h_start = (newH - fH) // 2  # (900 - 640)  // 2 = 130
            crop_w_start = (newW - fW) // 2  # 1600 1600
            crop_h_start += int(np.random.uniform(*self.data_config['crop']) * fH)
            crop_w_start += int(np.random.uniform(*self.data_config['crop']) * fW)

            # (0, 130, 1600, 130+640)
            crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)

            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            fH, fW = self.data_config['test_input_size']
            resize = float(fW)/float(W)
            resize += self.data_config.get('test_resize', 0.0)
            resize_dims = (int(W * resize), int(H * resize))

            newW, newH = resize_dims
            crop_h_start = (newH - fH) // 2
            crop_w_start = (newW - fW) // 2
            crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)

            flip = self.data_config['test_flip']
            rotate = self.data_config['test_rotate']

        pad_data = self.data_config['pad']
        pad_divisor = self.data_config['pad_divisor']
        pad_color = self.data_config['pad_color']
        pad = (pad_data, pad_color)

        return resize, resize_dims, crop, flip, rotate, pad

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate, pad):
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate, pad)

        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        top, right, bottom, left = pad[0]
        post_tran[0] = post_tran[0] + left  # left
        post_tran[1] = post_tran[1] + top  # top

        ret_post_rot, ret_post_tran = np.eye(3), np.zeros(3)
        ret_post_rot[:2, :2] = post_rot
        ret_post_tran[:2] = post_tran

        return img, ret_post_rot, ret_post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate, pad):
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        top, right, bottom, left = pad[0]
        pad_color = pad[1]
        img = self.img_pad(img, top, right, bottom, left, pad_color)
        return img

    def img_pad(self, pil_img, top, right, bottom, left, color):
        if top == right == bottom == left == 0:
            return pil_img
        assert top == bottom, (top, right, bottom, left)
        assert left == right, (top, right, bottom, left)
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def rts2proj(self, cam_info, post_rot=None, post_tran=None):
        if cam_info is None:
            return None

        #lidar2cam_r = np.linalg.inv(cam_info['rot'])
        #lidar2cam_t = cam_info['tran'] @ lidar2cam_r.T
        #lidar2cam_rt = np.eye(4)
        #lidar2cam_rt[:3, :3] = lidar2cam_r.T
        #lidar2cam_rt[3, :3] = -lidar2cam_t
        lidar2cam_rt = cam_info['lidar2cam']
        intrinsic = cam_info['intrin']

        viewpad = np.eye(4)
        if post_rot is not None:
            assert post_tran is not None, [post_rot, post_tran]
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = post_rot @ intrinsic
            viewpad[:3, 2] += post_tran
        else:
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        #lidar2img_rt = (viewpad @ lidar2cam_rt.T)
        #lidar2img_rt = np.array((viewpad @ lidar2cam_rt.T), dtype=np.float32)
        lidar2img_rt = (viewpad @ lidar2cam_rt)
        lidar2img_rt = np.array((viewpad @ lidar2cam_rt), dtype=np.float32)

        return lidar2img_rt.astype(np.float32)

    def __repr__(self):
        repr_str = self.__class__.__name__
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
                            'axis_align_matrix', 'scene_token', 'lidar2ego')
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys


@TRANSFORMS.register_module()
class CustomRandomFlip3D(RandomFlip3D):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        sync_2d (bool): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float): The flipping probability
            in vertical direction. Defaults to 0.0.
        flip_box3d (bool): Whether to flip bounding box. In most of the case,
            the box should be fliped. In cam-based bev detection, this is set
            to False, since the flip of 2D images does not influence the 3D
            box. Defaults to True.
    """

    def __init__(self,
                 flip_2d: bool = True,
                 sync_2d: bool = True,
                 flip_ratio_bev_horizontal: float = 0.0,
                 flip_ratio_bev_vertical: float = 0.0,
                 flip_box3d: bool = True,
                 update_img2lidar = False,
                 **kwargs) -> None:
        super(CustomRandomFlip3D, self).__init__(
            sync_2d, flip_ratio_bev_horizontal, flip_ratio_bev_vertical,
            flip_box3d, **kwargs)

        self.flip_2d = flip_2d
        self.update_img2lidar = update_img2lidar

    def update_transform(self, input_dict):
        for _, cam_info in enumerate(input_dict['lidar2img']['lidar2img_aug']):
            transform = cam_info['cam2lidar'].copy()

            aug_transform = np.eye(4, dtype=np.float32)
            if input_dict['pcd_horizontal_flip']:
                aug_transform[1, 1] = -1
            if input_dict['pcd_vertical_flip']:
                aug_transform[0, 0] = -1
            new_transform = aug_transform @ transform

            cam_info['cam2_lidar'] = new_transform

    def transform(self, input_dict: dict) -> dict:
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
            'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
            into result dict.
        """
        # flip 2D image and its annotations
        if self.flip_2d and 'img' in input_dict:
            super(RandomFlip3D, self).transform(input_dict)

        if self.sync_2d and 'img' in input_dict:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio_bev_horizontal else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])

        if self.update_img2lidar:
            self.update_transform(input_dict)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f'(flip_2d={self.flip_2d},'
        repr_str += f' flip_ratio_bev_horizontal={self.flip_ratio_bev_horizontal})'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@TRANSFORMS.register_module()
class CustomGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Apply global rotation, scaling and translation to a 3D scene.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of
            translation noise applied to a scene, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range: List[float] = [-0.78539816, 0.78539816],
                 scale_ratio_range: List[float] = [0.95, 1.05],
                 translation_std: List[int] = [0, 0, 0],
                 shift_height: bool = False,
                 update_img2lidar: bool = False) -> None:

        super(CustomGlobalRotScaleTrans, self).__init__(
            rot_range, scale_ratio_range, translation_std, shift_height)

        self.update_img2lidar = update_img2lidar

    def update_transform(self, input_dict):
        for _, cam_info in enumerate(input_dict['lidar2img']['lidar2img_aug']):
            transform = cam_info['cam2lidar'].copy()

            aug_transform = np.zeros((4, 4), dtype=np.float32)
            if 'pcd_rotation' in input_dict:
                aug_transform[:3, :3] = input_dict['pcd_rotation'].T * input_dict['pcd_scale_factor']
            else:
                aug_transform[:3, :3] = np.eye(3) * input_dict['pcd_scale_factor']
            aug_transform[:3, -1] = input_dict['pcd_trans']
            aug_transform[-1, -1] = 1.0

            new_transform = aug_transform @ transform
            cam_info['cam2lidar'] = new_transform

    def _rot_bbox_points(self, input_dict: dict) -> None:
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        if not 'point' in input_dict:
            return

        super()._rot_bbox_points(input_dict)


    def _trans_bbox_points(self, input_dict: dict) -> None:
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        if 'points' in input_dict:
            input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        if 'gt_bboxes_3d' in input_dict:
            input_dict['gt_bboxes_3d'].translate(trans_factor)

    def _scale_bbox_points(self, input_dict: dict) -> None:
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points' and
            `gt_bboxes_3d` is updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        if 'points' in input_dict:
            points = input_dict['points']
            points.scale(scale)
            if self.shift_height:
                assert 'height' in points.attribute_dims.keys(), \
                    'setting shift_height=True but points have no height attribute'
                points.tensor[:, points.attribute_dims['height']] *= scale
            input_dict['points'] = points

        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            input_dict['gt_bboxes_3d'].scale(scale)


    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        input_dict = super().transform(input_dict)

        if self.update_img2lidar:
            self.update_transform(input_dict)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        repr_str += f' update_img2lidar={self.update_img2lidar})'
        return repr_str
