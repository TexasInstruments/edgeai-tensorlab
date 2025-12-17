# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet.models.layers.positional_encoding import LearnedPositionalEncoding


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
    """
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()),
                        dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb
    """

    pos_xyz = pos[..., None] / dim_t
    pos_xyz = torch.stack((pos_xyz[..., 0::2].sin(), pos_xyz[..., 1::2].cos()),
                          dim=-1).flatten(-2)
    posemb_xyz = torch.cat((pos_xyz[..., 1, :], pos_xyz[..., 0, :], pos_xyz[..., 2, :]), dim=-1)
    return posemb_xyz


@TASK_UTILS.register_module()
class BEVFormerLearnedPositionalEncoding(LearnedPositionalEncoding):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(BEVFormerLearnedPositionalEncoding, self).__init__(
            num_feats,
            row_num_embed=row_num_embed,
            col_num_embed=col_num_embed,
            init_cfg=init_cfg)
