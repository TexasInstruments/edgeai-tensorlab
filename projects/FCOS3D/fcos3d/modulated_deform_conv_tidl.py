# Copyright (c) OpenMMLab. All rights reserved.
import torch
import copy
import torch.nn.functional as F
from mmengine.registry import MODELS

from mmcv.ops import ModulatedDeformConv2dPack

from torchvision.ops import deform_conv2d

def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    # y: 1.5 ~ 58.5
    # x: 1.5 ~ 100.5
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y), -1).reshape(
        1, H_out, W_out, 1, 2)

    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    """
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device))
    """
    y, x = torch.meshgrid(
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device))

    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid


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

        self.zero_padding = torch.nn.ZeroPad2d(1)


    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_y*w + offset_x
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        #x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        #x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        x_offset = x_offset.permute(0, 1, 4, 2, 3)
        x_offset = x_offset.reshape(b, c, ks, ks, h, w)
        x_offset = x_offset.permute(0, 1, 4, 2, 5, 3)
        x_offset = x_offset.reshape(b, c, h*ks, w*ks)

        return x_offset


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        #o1 = torch.ones_like(o1) * 0.75
        #o2 = torch.ones_like(o2) * 0.25
        offset = torch.cat((o1, o2), dim=1)

        mask = torch.sigmoid(mask)

        """ Deformable Convolution using GridSample
        x (1,Ni,H,W) ---------------------------------------------------------------------> GridSample (1,Ni,Fr*Fc*H,W) -> Mul (1,Ni,9*H,W) ->  Reshape (1,Ni*9*H,W) -> Conv1x1 (1,No,H,W)
                                                                                                 ^                          ^                                              ^
        offset_x (1,Fr*Fc,H,W) -> Unsqueeze(-1) (1,Fr*Fc,H,W,1) --                               |                          |                                              |
                                                                 |-> Concat (1,Fr*Fc,H,W,2) -> Reshape (1,Fr*Fc*H,W,2)      |                                              |
        offset_y (1,Fr*Fc,H,W) -> Unsqueeze(-1) (1,Fr*Fc,H,W,1) --                                                          |                                              |
                                                                                                                            |                                              |
        mask (1,Fr*Fc,H,W) ------------------------------------------------------------------> Reshape (1, 1, 9*H, W) -------                                              |
                                                                                                                                                                           |
        weight (No,Ni,Fr,Fc) --------------------------------------------------------------------------------------------------------  -----> Reshape (No,Ni*Fr*Fc,1,1)-----
        """

        # 1. Concat offset_x and offset_y, and get feature map using grid_sample, 
        #    where m = Fr*Fc
        b, n, h, w   = x.shape
        _, m, _, _   = mask.shape
        _, _, fr, fc = self.weight.shape

        # 2. zero-offset location
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h, 1, device=o1.device, dtype=o1.dtype),
                                        torch.arange(0, w, 1, device=o1.device, dtype=o1.dtype))
        grid_y = grid_y.repeat(m, 1, 1)
        grid_x = grid_x.repeat(m, 1, 1)

        k_y, k_x = torch.meshgrid(torch.arange(-(fr-1)//2, (fr-1)//2+1, 1, device=o1.device, dtype=o1.dtype),
                                  torch.arange(-(fc-1)//2, (fc-1)//2+1, 1, device=o1.device, dtype=o1.dtype))

        k_y = k_y.reshape(m, 1, 1)
        k_x = k_x.reshape(m, 1, 1)

        if 1:
            grid_y = grid_y + k_y # p + pk
            grid_x = grid_x + k_x

            #x = self.zero_padding(x)

            # o2/o1 is x/y
            grid_y = (o1 + grid_y) / (h-1) # 1x9x58x100
            grid_x = (o2 + grid_x) / (w-1) # 1x9x58x100
            # in (x, y) order
            offset_grid = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1) # 1x9x58x100x2

            # scale to [-1 to 1]
            offset_grid = 2 * offset_grid - 1
            offset_grid = offset_grid.reshape(b, m*h, w, 2) # 1x522x100x2
            # x: 1x256x58x100, offset_grid: 1x522x100x2
            # output: 1x256x522x100
            sampling_feature = F.grid_sample(x,
                                             offset_grid,
                                             mode='bilinear',
                                             padding_mode='zeros',
                                             align_corners=False)

            sampling_feature = sampling_feature * mask.reshape(b, 1, m*h, w)
            sampling_feature = sampling_feature.reshape(b, n*m, h, w)

            # 3. Reshape self.weight to (1x1) weight
            weight_1x1 = self.weight.reshape(self.weight.shape[0], -1, 1, 1)

            # 4. 1x1 convolution
            out1 = F.conv2d(sampling_feature, weight_1x1, bias=self.bias)
            
            # For comparison
            #new_mask = torch.ones_like(mask)
            #new_offset = torch.ones_like(offset)
            #out1 = deform_conv2d(x, offset, self.weight, self.bias,
            #                     self.stride, self.padding, self.dilation, mask)

            return out1
        else:
            grid_y = grid_y + k_y + 1
            grid_x = grid_x + k_x + 1

            x_org = copy.deepcopy(x)
            mask_org = copy.deepcopy(mask)

            x = self.zero_padding(x)

            # Add offset
            p = torch.cat((grid_y.unsqueeze(0) + o1, grid_x.unsqueeze(0) + o2), dim=1)
            p = p.contiguous().permute(0, 2, 3, 1)
            q_tl = p.floor()
            q_br = q_tl + 1

            q_tl = torch.cat([torch.clamp(q_tl[..., :m], 0, x.size(2)-1), torch.clamp(q_tl[..., m:], 0, x.size(3)-1)], dim=-1).long()
            q_br = torch.cat([torch.clamp(q_br[..., :m], 0, x.size(2)-1), torch.clamp(q_br[..., m:], 0, x.size(3)-1)], dim=-1).long()
            q_tr = torch.cat([q_tl[..., :m], q_br[..., m:]], dim=-1)
            q_bl = torch.cat([q_br[..., :m], q_tl[..., m:]], dim=-1)

            # clip p
            p = torch.cat([torch.clamp(p[..., :m], 0, x.size(2)-1), torch.clamp(p[..., m:], 0, x.size(3)-1)], dim=-1)

            # bilinear kernel (b, h, w, N)
            g_tl = (1 - (p[..., :m] - q_tl[..., :m].type_as(p))) * (1 - (p[..., m:] - q_tl[..., m:].type_as(p)))
            g_br = (p[..., :m] - q_tl[..., :m].type_as(p))       * (p[..., m:] - q_tl[..., m:].type_as(p))
            g_tr = (1 - (p[..., :m] - q_tl[..., :m].type_as(p))) * (p[..., m:] - q_tl[..., m:].type_as(p))
            g_bl = (p[..., :m] - q_tl[..., :m].type_as(p))       * (1 - (p[..., m:] - q_tl[..., m:].type_as(p)))

            # (b, c, h, w, N)
            x_q_tl = self._get_x_q(x, q_tl, m)
            x_q_br = self._get_x_q(x, q_br, m)
            x_q_tr = self._get_x_q(x, q_tr, m)
            x_q_bl = self._get_x_q(x, q_bl, m)

            # (b, c, h, w, N)
            x_offset = g_tl.unsqueeze(dim=1) * x_q_tl + \
                       g_br.unsqueeze(dim=1) * x_q_br + \
                       g_tr.unsqueeze(dim=1) * x_q_tr + \
                       g_bl.unsqueeze(dim=1) * x_q_bl

            mask = mask.permute(0, 2, 3, 1).unsqueeze(dim=1)
            x_offset = x_offset * mask
            x_offset = self._reshape_x_offset(x_offset, fr)

            out1 = F.conv2d(x_offset, self.weight, bias=self.bias, stride=3)

            # For comparison
            #out1 = deform_conv2d(x_org, offset, self.weight, self.bias,
            #                     self.stride, self.padding, self.dilation, mask_org)

            #diff = out1-out2

            return out1


    '''
    # Based on DCNv3
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        #offset = torch.cat((o1, o2), dim=1)
        offset = torch.cat((o2.permute(0, 2, 3, 1).unsqueeze(-1), o1.permute(0, 2, 3, 1).unsqueeze(-1)), dim=-1)
        offset = offset.reshape(mask.shape[0], mask.shape[2], mask.shape[3], -1)

        mask = torch.sigmoid(mask)

        # Set params
        pad_h          = 1
        pad_w          = 1
        dilation_h     = 1
        dilation_w     = 1
        stride_h       = 1
        stride_w       = 1
        kernel_h       = 3
        kernel_w       = 3
        group          = 1
        group_channels = x.shape[1] // group
        offset_scale   = 1

        input    = x.permute(0, 2, 3, 1)        # N, H, W, C
        #offset   = offset.permute(0, 2, 3, 1)   # N, H, W, 18
        mask     = mask.permute(0, 2, 3, 1)     # N, H, W, 9

        input    = F.pad(input, [0, 0, pad_w, pad_w, pad_h, pad_h])
        N_, H_in, W_in, _   = input.shape
        _, H_out, W_out, _ = offset.shape

        # ref: (x - y) order
        ref = _get_reference_points(
            input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
        # grid: (x - y) order. But 3x3 position is in column-row order
        grid = _generate_dilation_grids(
            input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
        spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
            repeat(1, 1, 1, group*kernel_h*kernel_w).to(input.device)

        # sampling location: (x - y) order. For each pixel, 9 offsets is in colume-row order
        sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + \
            offset * offset_scale / spatial_norm

        P_ = kernel_h * kernel_w
        sampling_grids = 2 * sampling_locations - 1
        # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
        input_ = input.view(N_, H_in*W_in, group*group_channels).transpose(1, 2).\
            reshape(N_*group, group_channels, H_in, W_in)
        # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
        sampling_grid_ = sampling_grids.view(N_, H_out*W_out, group, P_, 2).transpose(1, 2).\
            flatten(0, 1)
        # N_*group, group_channels, H_out*W_out, P_
        sampling_input_ = F.grid_sample(
            input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)

        # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
        mask = mask.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
            reshape(N_*group, 1, H_out*W_out, P_)
        output = (sampling_input_ * mask).view(N_,
                                               group*group_channels, H_out, W_out, P_).\
            permute(0, 1, 4, 2, 3).reshape(N_, group*group_channels*P_, H_out, W_out)

        # 1x1 convolution
        weight_1x1 = self.weight.reshape(self.weight.shape[0], -1, 1, 1)
        #weight_1x1 = self.weight.permute(0, 1, 3, 2).reshape(self.weight.shape[0], -1, 1, 1)
        return F.conv2d(output, weight_1x1, bias=self.bias)

        #output = (sampling_input_ * mask).sum(-1).view(N_,
        #                                               group*group_channels, H_out*W_out)
        ##return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()

        #output = output.reshape(N_, -1, H_out, W_out).contiguous()
        #return F.conv2d(output, self.weight, bias=self.bias, \
        #                stride=self.stride, padding=self.padding, dilation=self.dilation)
    '''