# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import random as prandom
import warnings
from PIL import Image
import string

from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

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

