import torch
import numpy as np


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def rts2proj(cam_info, post_rot=None, post_tran=None):
    if cam_info is None:
        return None

    lidar2cam_rt = cam_info['lidar2cam']
    intrinsic = cam_info['intrin']

    viewpad = np.eye(4)
    if post_rot is not None:
        assert post_tran is not None, [post_rot, post_tran]
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = post_rot @ intrinsic
        viewpad[:3, 2] += post_tran
    else:
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

    lidar2img_rt = (viewpad @ lidar2cam_rt)
    lidar2img_rt = np.array((viewpad @ lidar2cam_rt), dtype=np.float32)

    return lidar2img_rt.astype(np.float32)

def get_augmented_img_params(img_meta):
    fH, fW = img_meta['pad_shape']
    H, W   = img_meta['ori_shape']

    resize = float(fW)/float(W)
    resize_dims = (int(W * resize), int(H * resize))

    newW, newH = resize_dims
    crop_h_start = (newH - fH) // 2
    crop_w_start = (newW - fW) // 2
    crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)

    return resize, resize_dims, crop

def scale_augmented_img_params(post_rot, post_tran, resize_r, resize_dims, crop):
    post_rot *= resize_r
    post_tran -= torch.Tensor(crop[:2])
    
    A = get_rot(0)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    ret_post_rot, ret_post_tran = np.eye(3), np.zeros(3)
    ret_post_rot[:2, :2] = post_rot
    ret_post_tran[:2] = post_tran

    return ret_post_rot, ret_post_tran
