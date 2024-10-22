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
