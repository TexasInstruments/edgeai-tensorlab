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

from mmdet.models.layers.positional_encoding import SinePositionalEncoding


@TASK_UTILS.register_module()
class Detr3DSinePositionalEncoding(SinePositionalEncoding):
    """Position encoding with sine and cosine functions. See `End-to-End Object
    Detection with Transformers.

    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super().__init__(
            num_feats,
            temperature=temperature,
            normalize=normalize,
            scale=scale,
            eps=eps,
            offset=offset,
            init_cfg=init_cfg)
