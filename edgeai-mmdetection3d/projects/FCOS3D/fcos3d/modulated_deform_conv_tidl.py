# Copyright (c) OpenMMLab. All rights reserved.
import torch
import copy
import torch.nn.functional as F
from mmengine.registry import MODELS

from mmcv.ops import ModulatedDeformConv2dPack

pad_h          = 1
pad_w          = 1
dilation_h     = 1
dilation_w     = 1
stride_h       = 1
stride_w       = 1
kernel_h       = 3


@MODELS.register_module('DCNv2_tidl')
class ModulatedDeformConv2dTIDL(ModulatedDeformConv2dPack):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(x)

        #o1, o2, mask = torch.chunk(out, 3, dim=1)
        o1 = out[:, 0:18:2, :, :].to(torch.float32)
        o2 = out[:, 1:18:2, :, :].to(torch.float32)
        mask = out[:,-9:, :, :]

        mask = torch.sigmoid(mask)

        """ Deformable Convolution using GridSample
        x (1,Ni,H,W) --------------------> Pad (1, Ni, H+2, W+2) -------------------------> GridSample (1,Ni,Fr*Fc*H,W) -> Mul (1,Ni,9*H,W) ->  Reshape (1,Ni*9*H,W) -> Conv1x1 (1,No,H,W)
                                                                                                 ^                          ^                                              ^
        offset_x (1,Fr*Fc,H,W) -> Unsqueeze(-1) (1,Fr*Fc,H,W,1) --                               |                          |                                              |
                                                                 |-> Concat (1,Fr*Fc,H,W,2) -> Reshape (1,Fr*Fc*H,W,2)      |                                              |
        offset_y (1,Fr*Fc,H,W) -> Unsqueeze(-1) (1,Fr*Fc,H,W,1) --                                                          |                                              |
                                                                                                                            |                                              |
        mask (1,Fr*Fc,H,W) ------------------------------------------------------------------> Reshape (1, 1, 9*H, W) -------                                              |
                                                                                                                                                                           |
        weight (No,Ni,Fr,Fc) --------------------------------------------------------------------------------------------------------  -----> Reshape (No,Ni*Fr*Fc,1,1)-----
        """

        # 1. input feature padding
        x = F.pad(x, [pad_w, pad_w, pad_h, pad_h, 0, 0])

        # 2. Feature map and mask size, where m = Fr*Fc
        b, n, h, w   = x.shape
        _, m, ho, wo = mask.shape
        _, _, fr, fc = self.weight.shape

        # 3. zero-offset location
        grid_y, grid_x = torch.meshgrid(
            torch.arange((dilation_h * (fr - 1)) // 2 + 0.5,
                         (dilation_h * (fr - 1)) // 2 + 0.5 + (ho - 1) * stride_h + 1,
                         1, device=o1.device, dtype=o1.dtype),
            torch.arange((dilation_w * (fc - 1)) // 2 + 0.5,
                          (dilation_w * (fc - 1)) // 2 + 0.5 + (wo - 1) * stride_w + 1,
                          1, device=o1.device, dtype=o1.dtype))

        grid_y = grid_y.repeat(m, 1, 1)
        grid_x = grid_x.repeat(m, 1, 1)

        # 4. 3x3 filter location without DCN
        k_y, k_x = torch.meshgrid(
            torch.arange(-(dilation_h*(fr-1))//2, (dilation_h*(fr-1))//2 +1, 1, device=o1.device, dtype=o1.dtype),
            torch.arange(-(dilation_w*(fc-1))//2, (dilation_w*(fc-1))//2 +1, 1, device=o1.device, dtype=o1.dtype))

        k_y = k_y.reshape(m, 1, 1)
        k_x = k_x.reshape(m, 1, 1)

        grid_y = grid_y + k_y
        grid_x = grid_x + k_x

        # 5. Normalizing sampling location (o2/o1 is x/y) 
        grid_y = (o1 + grid_y) / float(h) # 1x9xHxW # quantization does not su[pport double, making it quantization friendly
        grid_x = (o2 + grid_x) / float(w) # 1x9xHxW

        # in (x, y) order
        offset_grid = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1) # 1x9xHxWx2

        # 6. Scale sampling location to [-1 to 1]
        offset_grid = float(2) * offset_grid - float(1)
        offset_grid = offset_grid.reshape(b, m*ho, wo, 2) # 1x(9*H)xWx2

        # 7. Sample features
        # x: 1xCx(H+2)x(W+2), offset_grid: 1x(9*H)xWx2
        # output: 1xCx(9*Ho)xWo
        sampling_feature = F.grid_sample(x,
                                         offset_grid,
                                         mode='bilinear',
                                         padding_mode='zeros',
                                         align_corners=False)

        sampling_feature = sampling_feature * mask.reshape(b, 1, m*ho, wo)
        sampling_feature = sampling_feature.reshape(b, n*m, ho, wo)

        # 8. Reshape self.weight to (1x1) weight
        weight_1x1 = self.weight.reshape(self.weight.shape[0], -1, 1, 1)

        # 9. 1x1 convolution
        out = F.conv2d(sampling_feature, weight_1x1, bias=self.bias)
        return out
