# Copyright (c) OpenMMLab. All rights reserved.
# Some of the classes are copied from projects/PETR/petr/transforms_3d.py 
###############################
import numpy as np
from numpy import random
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet3d.structures.points import BasePoints
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
class CircleObjectRangeFilter(BaseTransform):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        self.class_dist_thred = class_dist_thred

    def transform(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


@TRANSFORMS.register_module()
class NuScenesSparse4DAdaptor(BaseTransform):
    def __init(self):
        pass

    def transform(self, input_dict):
        input_dict["projection_mat"] = to_tensor(
            np.stack(input_dict["lidar2img"])
        )
        input_dict["image_wh"] = to_tensor(input_dict["img_shape"])[:, :2][:, [1, 0]].to(torch.float32)

        input_dict["T_global_inv"] = to_tensor(np.linalg.inv(input_dict["lidar2global"]))
        input_dict["T_global"] = to_tensor(input_dict["lidar2global"])
        if "cam2img" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam2img"])
            )
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            #input_dict["gt_bboxes_3d"] = to_tensor(input_dict["gt_bboxes_3d"]).float()
        #if "gt_labels_3d" in input_dict:
        #    input_dict["gt_labels_3d"] = to_tensor(input_dict["gt_labels_3d"]).long()

        #imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        #imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        #input_dict["img"] = to_tensor(imgs)
        #input_dict["img"] = imgs
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@TRANSFORMS.register_module()
class MultiScaleDepthMapGenerator(BaseTransform):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def transform(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict


@TRANSFORMS.register_module()
class InstanceNameFilter(BaseTransform):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def transform(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][
                gt_bboxes_mask
            ]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
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

'''
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
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d', 'prev_exists', 'instance_id'
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
'''

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

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
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
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

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
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str