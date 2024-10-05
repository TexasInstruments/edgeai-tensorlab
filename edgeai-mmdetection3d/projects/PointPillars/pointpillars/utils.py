# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import build_conv_layer
from torch import Tensor, nn
from torch.nn import functional as F



class PPPFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict, optional): Config dict of normalization layers.
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        last_layer (bool, optional): If last_layer, there is no
            concatenation of features. Defaults to False.
        mode (str, optional): Pooling model to gather features inside voxels.
            Defaults to 'max'.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer: Optional[bool] = False,
                 replace_mat_mul=False,
                 mode: Optional[str] = 'max'):

        super().__init__()
        self.name = 'PPPFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if replace_mat_mul is False:
            self.norm = build_norm_layer(norm_cfg, self.units)[1]
            self.linear = nn.Linear(in_channels, self.units, bias=False)
        else:
            self.linear = build_conv_layer(cfg=dict(type='Conv2d'),
                                        in_channels=in_channels,
                                        out_channels=self.units,
                                        kernel_size=1)
            norm_cfg['type'] = 'BN2d'
            self.norm = build_norm_layer(norm_cfg, self.units)[1]

        self.replace_mat_mul = replace_mat_mul

        assert mode in ['max', 'avg']
        self.mode = mode

    def forward(self,
                inputs: Tensor,
                num_voxels: Optional[Tensor] = None,
                aligned_distance: Optional[Tensor] = None,
                ip_tensor_dim_correct=False) -> Tensor:
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """

        if self.replace_mat_mul is False:
            x = self.linear(inputs)
            x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                                1).contiguous()
            x = F.relu(x)

            if self.mode == 'max':
                if aligned_distance is not None:
                    x = x.mul(aligned_distance.unsqueeze(-1))
                x_max = torch.max(x, dim=1, keepdim=True)[0]
            elif self.mode == 'avg':
                if aligned_distance is not None:
                    x = x.mul(aligned_distance.unsqueeze(-1))
                x_max = x.sum(
                    dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                        -1, 1, 1)

            if self.last_vfe:
                return x_max
            else:
                x_repeat = x_max.repeat(1, inputs.shape[1], 1)
                x_concatenated = torch.cat([x, x_repeat], dim=2)
                return x_concatenated

        else:
            # replacating whole code for simplicity
            # keeping intact the input data dimension and format
            if self.mode == 'max':
                if ip_tensor_dim_correct is False:
                    inputs = inputs.permute(2,1,0)
                    inputs = inputs.unsqueeze(0)
                x = self.linear(inputs)
                x = self.norm(x.contiguous()).contiguous()
                x_max = torch.max(x, dim=2, keepdim=True)[0]
                return x_max
            else:
                print('Not supported')

