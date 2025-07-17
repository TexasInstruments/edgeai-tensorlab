# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from torch import nn
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet3d.structures.bbox_3d.utils import limit_period


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    length = bboxes[..., 3:4].log()
    width = bboxes[..., 4:5].log()
    height = bboxes[..., 5:6].log()

    rot = -bboxes[..., 6:7] - np.pi / 2
    rot = limit_period(rot, period=np.pi * 2)
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, length, width, cz, height, rot.sin(), rot.cos(), vx, vy),
            dim=-1)
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, length, width, cz, height, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes

# For StreamPETR
# It corresponds to StreamPETR::forward()
# If we modify StreamPETRHead::forward(),
# we could use the same normailize_bbox() as PETR
def normalize_bbox_streampetr(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    rot = -rot - np.pi / 2
    rot = limit_period(rot, period=np.pi * 2)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    length = normalized_bboxes[..., 2:3]
    width = normalized_bboxes[..., 3:4]
    height = normalized_bboxes[..., 5:6]

    width = width.exp()
    length = length.exp()
    height = height.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, length, width, height, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, length, width, height, rot], dim=-1)

    return denormalized_bboxes


# For StreamPETR
def denormalize_bbox_streampetr(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]

    # size
    w = normalized_bboxes[..., 3:4]
    l = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

@torch.no_grad()
def locations(features, stride, pad_h, pad_w):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """

        h, w = features.size()[-2:]
        device = features.device

        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)

        locations = locations.reshape(h, w, 2)

        return locations


def apply_center_offset(locations, center_offset):
    """
    :param locations:  (1, H, W, 2)
    :param pred_ltrb:  (N, H, W, 4) 
    """
    centers_2d = torch.zeros_like(center_offset)
    locations = inverse_sigmoid(locations)
    centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
    centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
    centers_2d = centers_2d.sigmoid()

    return centers_2d


def apply_ltrb(locations, pred_ltrb):
    """
    :param locations:  (1, H, W, 2)
    :param pred_ltrb:  (N, H, W, 4) 
    """
    pred_boxes = torch.zeros_like(pred_ltrb)
    pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])# x1
    pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])# y1
    pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])# x2
    pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])# y2
    min_xy = pred_boxes[..., 0].new_tensor(0)
    max_xy = pred_boxes[..., 0].new_tensor(1)
    pred_boxes  = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
    pred_boxes  = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
    pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)

    return pred_boxes


def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:]) 
    return memory * prev_exist


def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape

        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)

        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points


class SELayer_Linear(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out
