# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import numpy as np
import torch
import mmcv

from mmcv.transforms import BaseTransform
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D


@TRANSFORMS.register_module()
class ImageAug(BaseTransform):
    """Random resize, Crop and flip the image

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        ida_aug_conf,
        is_train=False,
        sequential=False
    ):
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.sequential = sequential

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        # transform image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        
        return img, ida_mat


    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # transform image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def sample_augmentation(self, flip = None, scale=None):

        H,  W  = self.ida_aug_conf['src_size']
        fH, fW = self.ida_aug_conf['input_size']

        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.ida_aug_conf['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            random_crop_height = \
                self.ida_aug_conf.get('random_crop_height', False)
            if random_crop_height:
                crop_h = int(np.random.uniform(max(0.3*newH, newH-fH),
                                               newH-fH))
            else:
                crop_h = \
                    int((1 - np.random.uniform(*self.ida_aug_conf['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.ida_aug_conf['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.ida_aug_conf['rot'])
            if self.ida_aug_conf.get('vflip', False) and np.random.choice([0, 1]):
                rotate += 180
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.ida_aug_conf.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def photo_metric_distortion(self, img, pmd):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if np.random.rand()>pmd.get('rate', 1.0):
            return img

        img = np.array(img).astype(np.float32)
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-pmd['brightness_delta'],
                                   pmd['brightness_delta'])
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(pmd['saturation_lower'],
                                          pmd['saturation_upper'])

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-pmd['hue_delta'], pmd['hue_delta'])
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        return Image.fromarray(img.astype(np.uint8))


    def transform(self, results):

        imgs = results['img']
        N = len(imgs)
        canvas = []
        #intrinsics = []
        sensor2ego = []
        #ego2globals = []
        #cam2imgs = []
        post_rts = []
        new_imgs = []
        
        results['lidar2cam'] = np.array(results['lidar2cam'])

        img_augs = self.sample_augmentation()
        resize, resize_dims, crop, flip, rotate = img_augs

        # map camera index to camera names
        # it could be from a config
        camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        # BEVDet config
        #camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        for i in range(N):
            intrinsic = np.array(results['cam2img'][i])
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            results['cam2img'][i] = viewpad
            img = Image.fromarray(np.uint8(imgs[i]))

            # augmentation (resize, crop, horizontal flip, rotate)
            # different view use different aug (BEV Det)
            img, ida_mat = self.img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            if self.is_train and self.ida_aug_conf.get('pmd', None) is not None:
                img = self.photo_metric_distortion(img, self.ida_aug_conf['pmd'])

            canvas.append(np.array(img))
            post_rts.append(ida_mat)
            sensor2ego.append(results['images'][camera_names[i]]['cam2ego'])

            new_imgs.append(np.array(img).astype(np.float32))

            results['cam2img'][
                i][:3, :3] = ida_mat @ results['cam2img'][i][:3, :3]

            #cam2imgs.append(results['cam2img'][i])

        results['img'] = new_imgs
        results['sensor2ego'] = sensor2ego
        results['post_rts'] = post_rts

        results['canvas'] = canvas

        return results



@TRANSFORMS.register_module()
class BEVAug(BaseTransform):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy, tran_bda):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, :3] = gt_boxes[:, :3] + tran_bda
        return gt_boxes, rot_mat

    def transform(self, results):
        if self.is_train:
            gt_boxes = results['ann_info']['gt_bboxes_3d'].tensor
        else:
            gt_boxes = results['eval_ann_info']['gt_bboxes_3d'].tensor

        gt_boxes[:,2] = gt_boxes[:,2] + 0.5*gt_boxes[:,5]
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = \
            self.sample_bda_augmentation()
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy, tran_bda)
        if 'points' in results:
            points = results['points'].tensor
            points_aug = (bda_rot @ points[:, :3].unsqueeze(-1)).squeeze(-1)
            points[:,:3] = points_aug + tran_bda
            points = results['points'].new_point(points)
            results['points'] = points
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))

        # To REVISIT
        results['bda_mat'] = bda_mat

        if 'voxel_semantics' in results:
            if flip_dx:
                results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][::-1,...].copy()
            if flip_dy:
                results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()
        return results


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
                                'post_rts', 'bda_mat', 'sensor2ego') 
    ) -> None:
        super(CustomPack3DDetInputs, self).__init__(keys, meta_keys)

