# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from numpy import random
import torch

from PIL import Image

import mmcv
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
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
            if 'lidar2img' in results:
                results['lidar2img'][i] = results['cam2img'][i] @ results['lidar2cam'][i]

        results['img'] = new_imgs
        results['img_shape'] = [img.shape[:2] for img in results['img']]
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
class CustomMultiScaleFlipAug3D(MultiScaleFlipAug3D):
    def transform(self, results):
        results = super().transform(results)
        if isinstance(results, list):
            results = results[0]

        return results


@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def transform(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str

