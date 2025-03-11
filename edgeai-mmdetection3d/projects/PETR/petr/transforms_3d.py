# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs,  to_tensor
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D

from mmengine.structures import InstanceData

import mmcv
import warnings

@TRANSFORMS.register_module()
class ResizeCropFlipRotImage(BaseTransform):
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    def transform(self, results):

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.training and self.with_2d: # sync_2d bbox labels
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers_2d'][i]
                gt_labels = results['gt_bboxes_labels'][i]
                depths = results['depths'][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes, 
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths =  self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths)

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        results['gt_bboxes'] = new_gt_bboxes
        results['centers_2d'] = new_centers2d
        results['gt_bboxes_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]

        return results

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths,resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths


    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            #crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@TRANSFORMS.register_module()
class ResizeCropFlipImage(BaseTransform):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        results['lidar2cam'] = np.array(results['lidar2cam'])
        for i in range(N):
            intrinsic = np.array(results['cam2img'][i])
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            results['cam2img'][i] = viewpad
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['cam2img'][
                i][:3, :3] = ida_mat @ results['cam2img'][i][:3, :3]

        results['img'] = new_imgs

        return results

    def _get_rot(self, h):

        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@TRANSFORMS.register_module()
class GlobalRotScaleTransImage(BaseTransform):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)

        self.rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle *= -1
        results['gt_bboxes_3d'].rotate(np.array(rot_angle))

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        results['gt_bboxes_3d'].scale(scale_ratio)

        # TODO: support translation
        if not self.reverse_angle:
            gt_bboxes_3d = results['gt_bboxes_3d'].numpy()
            gt_bboxes_3d[:, 6] -= 2 * rot_angle
            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=9)

        return results

    def rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        if self.reverse_angle:
            rot_mat = torch.tensor([[rot_cos, rot_sin, 0, 0],
                                    [-rot_sin, rot_cos, 0, 0], [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        else:
            rot_mat = torch.tensor([[rot_cos, -rot_sin, 0, 0],
                                    [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        rot_mat_inv = torch.inverse(rot_mat)
        num_view = len(results['lidar2cam'])
        for view in range(num_view):
            #results['lidar2cam'][view] = (
            #    torch.tensor(np.array(results['lidar2cam'][view]).T).float()
            #    @ rot_mat_inv).T.numpy()
            results["lidar2cam"][view] = \
                (torch.tensor(results["lidar2cam"][view]).float() @ rot_mat_inv).numpy()

        # For StreamPETR
        if 'ego_pose' in results:
            results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ rot_mat_inv).numpy()
        if 'ego_pose_inv' in results:
            results['ego_pose_inv'] = (rot_mat.float() @ torch.tensor(results["ego_pose_inv"]).float()).numpy()
        if 'lidar2img' in results:
            for view in range(num_view):
                results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()

        return

    def scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])

        scale_mat_inv = torch.inverse(scale_mat)

        num_view = len(results['lidar2cam'])
        for view in range(num_view):
            #results['lidar2cam'][view] = (torch.tensor(
            #    scale_mat_inv.T
            #    @ results['lidar2cam'][view].T).float()).T.numpy()
            results['lidar2cam'][view] = \
                (torch.tensor(results["lidar2cam"][view]).float() @ scale_mat_inv).numpy()


        # For StreamPETR
        if 'ego_pose' in results:
            results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ scale_mat_inv).numpy()
        if 'ego_pose_inv' in results:
            results['ego_pose_inv'] = (scale_mat @ torch.tensor(results["ego_pose_inv"]).float()).numpy()
        if 'lidar2img' in results:
            for view in range(num_view):
                results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv).numpy()

        return


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
class CustomMultiScaleFlipAug3D(MultiScaleFlipAug3D):
    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip = False,
                 flip_direction = 'horizontal',
                 pcd_horizontal_flip = False,
                 pcd_vertical_flip = False):
        super().__init__(transforms,
                         img_scale,
                         pts_scale_ratio,
                         flip,
                         flip_direction,
                         pcd_horizontal_flip,
                         pcd_vertical_flip)

    def transform(self, results):
        results = super().transform(results)
        if isinstance(results, list):
            results = results[0]

        return results



@TRANSFORMS.register_module()
class CustomPack3DDetInputs(Pack3DDetInputs):
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
        'depths',
        'centers_2d'
    ]
    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d', 'prev_exists'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif 'images' in results:
                if len(results['images'].keys()) == 1:
                    cam_type = list(results['images'].keys())[0]
                    # single-view image
                    if key in results['images'][cam_type]:
                        data_metas[key] = results['images'][cam_type][key]
                else:
                    # multi-view image
                    img_metas = []
                    cam_types = list(results['images'].keys())
                    for cam_type in cam_types:
                        if key in results['images'][cam_type]:
                            img_metas.append(results['images'][cam_type][key])
                    if len(img_metas) > 0:
                        data_metas[key] = img_metas
            elif 'lidar_points' in results:
                if key in results['lidar_points']:
                    data_metas[key] = results['lidar_points'][key]
        data_sample.set_metainfo(data_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    warnings.warn(f'{key} is not in any KEYS')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results
