#################################################################################
# Copyright (c) 2018-2024, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################

import warnings
import torch
import torchvision
import torch.nn.functional as F


###########################################################################################
# Core DeformConv that takes feature, offset  & mask
###########################################################################################

class DeformConvOP2d(torchvision.ops.DeformConv2d):
    def __init__(self, *args, **kwargs):
        '''
        Deformable convolution operator (derived from torchvision.ops.DeformConv2d)
        '''
        mode = kwargs.pop('mode', 'bilinear')
        assert mode in ('bilinear',), 'mode should be: bilinear'
        super().__init__(*args, **kwargs)
        self.mode = mode
        warnings.warn('DeformConvOP2d or torchvision.ops.DeformConv2d is not our recommended '
            'Deformable Convolution implementation. Please use DeformConvWithGS2d instead.')


class DeformConvWithGS2d(torchvision.ops.DeformConv2d):
    """A ModulatedDeformable Conv Encapsulation using GridSample, that acts as normal Conv layers.
    """

    _version = 2
    def __init__(self, *args, **kwargs):
        """A ModulatedDeformable Conv Encapsulation using GridSample that acts as normal Conv
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
            mode (str): Interpolation model, 'bilinear' (default) or 'nearest'
        """

        mode = kwargs.pop('mode', 'bilinear')
        assert mode in ('bilinear', 'nearest'), 'mode should be one of: bilinear, nearest'
        super().__init__(*args, **kwargs)
        self.mode = mode

    def forward(self, feat, offset, mask):
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

        # [1,18,W,H] => 2 x [1,9,W,H]
        offset_y = offset[:,0::2,...]
        offset_x = offset[:,1::2,...]

        # 1. input feature padding
        _, _, fr, fc = self.weight.shape

        dilation_h = self.dilation[0]
        dilation_w = self.dilation[1]
        stride_h   = self.stride[0]
        stride_w   = self.stride[1]
        pad_h = (dilation_h * (fr - 1)) // 2
        pad_w = (dilation_w * (fc - 1)) // 2

        feat = F.pad(feat, [pad_w, pad_w, pad_h, pad_h, 0, 0])

        # 2. Feature map and mask size, where m = Fr*Fc
        b, n, h, w   = feat.shape
        _, m, ho, wo = mask.shape

        # 3. zero-offset location
        grid_y, grid_x = torch.meshgrid(
            torch.arange((dilation_h * (fr - 1)) // 2 + 0.5,
                         (dilation_h * (fr - 1)) // 2 + 0.5 + (ho - 1) * stride_h + 1,
                         1, device=offset_y.device, dtype=offset_y.dtype),
            torch.arange((dilation_w * (fc - 1)) // 2 + 0.5,
                         (dilation_w * (fc - 1)) // 2 + 0.5 + (wo - 1) * stride_w + 1,
                         1, device=offset_y.device, dtype=offset_y.dtype))

        grid_y = grid_y.repeat(m, 1, 1)
        grid_x = grid_x.repeat(m, 1, 1)

        # 4. 3x3 filter location without DCN
        k_y, k_x = torch.meshgrid(
            torch.arange(-(dilation_h*(fr-1))//2, (dilation_h*(fr-1))//2 +1, 1, device=offset_y.device, dtype=offset_y.dtype),
            torch.arange(-(dilation_w*(fc-1))//2, (dilation_w*(fc-1))//2 +1, 1, device=offset_y.device, dtype=offset_y.dtype))

        k_y = k_y.reshape(m, 1, 1)
        k_x = k_x.reshape(m, 1, 1)

        grid_y = grid_y + k_y
        grid_x = grid_x + k_x

        # 5. Normalizing sampling location (o2/o1 is x/y) 
        # quantization does not suppport double, float() is for making it quantization friendly
        grid_y = (offset_y + grid_y) / float(h) # 1x9xHxW
        grid_x = (offset_x + grid_x) / float(w) # 1x9xHxW

        # in (x, y) order
        offset_grid = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1) # 1x9xHxWx2

        # 6. Scale sampling location to [-1 to 1]
        offset_grid = float(2) * offset_grid - float(1)
        offset_grid = offset_grid.reshape(b, m*ho, wo, 2) # 1x(9*H)xWx2

        # 7. Sample features
        # feat: 1xCx(H+2)x(W+2), offset_grid: 1x(9*H)xWx2
        # output: 1xCx(9*Ho)xWo
        sampling_feature = F.grid_sample(feat,
                                         offset_grid,
                                         mode=self.mode,
                                         padding_mode='zeros',
                                         align_corners=False)

        sampling_feature = sampling_feature * mask.reshape(b, 1, m*ho, wo)
        sampling_feature = sampling_feature.reshape(b, n*m, ho, wo)

        # 8. Reshape self.weight to (1x1) weight
        weight_1x1 = self.weight.reshape(self.weight.shape[0], -1, 1, 1)

        # 9. 1x1 convolution
        out = F.conv2d(sampling_feature, weight_1x1, bias=self.bias)
        return out


# There is an DeformConv implementation in mmcv - see if it is installed
# https://mmcv.readthedocs.io/en/latest/
# https://github.com/open-mmlab/mmcv
try:
    import mmcv.ops
    MMCVDeformConv = mmcv.ops.ModulatedDeformConv2d
except:
    MMCVDeformConv = None


###########################################################################################
# The complete DCNv2 types with the Convolution that generates the offfset and the DeformConv
# These can be used as a drop in replacement for regular convolution.
###########################################################################################

def make_new_dcn_type(base_class, cls_name):
    """
    Create a new DCN type from the given base class.

    This should be ideally done by overloading the __new__, but for now this is quick workaround.
    """

    class _FactoryDCNv2(base_class):
        """
        DCNv2 - Deformable Convolution Layer

        Includes a regular convolution that generates the offset, mask
        and also the deformable convolution that uses it
        """
        def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            mode: str = 'bilinear',
            deform_groups: int = 1,
            offset_clip = None,
            offset_conv_split = True):

            # restricting the deformable convolution configuration
            super().__init__(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                             bias=bias, mode=mode)

            if offset_clip is None:
                warnings.warn("offset_clip is not set - recommend to set a reasonable value (eg. 32) to restrict the offsets")

            self.offset_conv_split = offset_conv_split
            ks = (kernel_size,kernel_size) if isinstance(kernel_size, int) else kernel_size
            self.mask_out_channels = deform_groups * ks[0] * ks[1]
            self.offset_out_channels = self.mask_out_channels * 2

            if self.offset_conv_split:
                self.conv_offset = torch.nn.Conv2d(in_channels=in_channels,
                        out_channels=self.offset_out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
                self.conv_mask = torch.nn.Conv2d(in_channels=in_channels,
                        out_channels=self.mask_out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
            else:
                self.conv_offset = torch.nn.Conv2d(in_channels=in_channels,
                        out_channels=(self.offset_out_channels+self.mask_out_channels),
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)

            if offset_clip is not None:
                offset_clip = (-offset_clip, offset_clip) if isinstance(offset_clip, (int,float)) else offset_clip
                self.offset_clip = torch.nn.Hardtanh(min_val=offset_clip[0], max_val=offset_clip[1])
            else:
                self.offset_clip = None

            self._initialize_weights()

        def forward(self, feat):
            if self.offset_conv_split:
                offset_yx = self.conv_offset(feat)
                mask = self.conv_mask(feat)
            else:
                offset_mask = self.conv_offset(feat)
                offset_yx = offset_mask[:,:self.offset_out_channels,...]
                mask = offset_mask[:,self.offset_out_channels:,...]

            if self.offset_clip is not None:
                offset_yx = self.offset_clip(offset_yx)

            mask = torch.sigmoid(mask)

            output = super().forward(feat, offset_yx, mask)
            return output

        def _initialize_weights(self):
            for name, module in self.named_modules():
                if hasattr(module, 'weight'):
                    torch.nn.init.kaiming_normal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        def copy_weights(self, other):
            if not isinstance(other , (_FactoryDCNv2, MMCVDeformConv)):
                warnings.warn(f"{self.__class__.__name__}.copy_weights cannot copy weights from {other.__class__.__name__}")

            other_split_conv_offset = other.offset_conv_split if hasattr(other, "offset_conv_split") else False
            if self.offset_conv_split and other_split_conv_offset:
                self.conv_offset.weight.detach().copy_(other.conv_offset.weight)
                if self.conv_offset.bias is not None and other.conv_offset.bias is not None:
                    self.conv_offset.bias.detach().copy_(other.conv_offset.bias)
            elif self.offset_conv_split:
                self.conv_offset.weight.detach().copy_(other.conv_offset.weight[:18, ...])
                self.conv_mask.weight.detach().copy_(other.conv_offset.weight[18:, ...])
                if self.conv_offset.bias is not None and other.conv_offset.bias is not None:
                    self.conv_offset.bias.detach().copy_(other.conv_offset.bias[:18])
                    self.conv_mask.bias.detach().copy_(other.conv_offset.bias[18:])
            elif other_split_conv_offset:
                self.conv_offset.weight[:18, ...].detach().copy_(other.conv_offset.weight)
                self.conv_mask.weight[18:, ...].detach().copy_(other.conv_offset.weight)
                if self.conv_offset.bias is not None and other.conv_offset.bias is not None:
                    self.conv_offset.bias[:18].detach().copy_(other.conv_offset.bias)
                    self.conv_mask.bias[18:].detach().copy_(other.conv_offset.bias)
            else:
                self.conv_offset.weight.detach().copy_(other.conv_offset.weight)
                if self.conv_offset.bias is not None and other.conv_offset.bias is not None:
                    self.conv_offset.bias.detach().copy_(other.conv_offset.bias)

            self.weight.detach().copy_(other.weight)
            if self.bias is not None and other.bias is not None:
                self.bias.detach().copy_(other.bias)

    return type(cls_name, (_FactoryDCNv2, base_class), {})

# DCNv2 with torchvision DeformConv Operator - this doesn't support onnx export yet
# This supports only bilinear mode of interpolation
# This may consume less memory and may be faster for training as the DeformConv is implemented as a single operator.
DCNOPv2 = make_new_dcn_type(DeformConvOP2d, 'DCNOPv2')

# DCNv2 with torch operators - ths is preferred for onnx export.
# Also, this supports bilinear and nearest modes of interpolation - nearest mode may be faster for inference.
DCNWithGSv2 = make_new_dcn_type(DeformConvWithGS2d, 'DCNWithGSv2')

# There is an DCNv2 implementation in mmcv - see if it is installed
# https://mmcv.readthedocs.io/en/latest/
# https://github.com/open-mmlab/mmcv
try:
    import mmcv.ops
    MMCVDCNv2 = mmcv.ops.ModulatedDeformConv2dPack
except:
    MMCVDCNv2 = None


###########################################################################################
# unit tests
###########################################################################################

def run_test_deform_op():
    # Set test params
    IN_HEIGHT = 58
    IN_WIDTH = 100
    OUT_HEIGHT = IN_HEIGHT
    OUT_WIDTH = IN_WIDTH

    IN_CHANNEL = 256
    OUT_CHANNEL = 256
    KERNEL_SIZE = 3
    STRIDE = 1
    PADDING = 1
    DILATION = 1
    GROUPS = 1
    BIAS = True
    DEFORM_GROUPS = 1
    INTP_MODE = 'bilinear'

    deform_op = DeformConvOP2d(IN_CHANNEL,
                                     OUT_CHANNEL,
                                     kernel_size=KERNEL_SIZE,
                                     stride=STRIDE,
                                     padding=PADDING,
                                     dilation=DILATION,
                                     groups=GROUPS,
                                     bias=BIAS)

    deform_with_gs  = DeformConvWithGS2d(IN_CHANNEL,
                                     OUT_CHANNEL,
                                     mode=INTP_MODE,
                                     kernel_size=KERNEL_SIZE,
                                     stride=STRIDE,
                                     padding=PADDING,
                                     dilation=DILATION,
                                     groups=GROUPS,
                                     bias=BIAS)

    # For evaluation, make weigth and bias of two models the same
    torch.nn.init.normal_(deform_op.weight)
    deform_with_gs.weight = deform_op.weight
    if deform_with_gs.bias is not None:
        deform_with_gs.bias   = deform_op.bias

    # Input to the model: feature map, offset_y, offste_x, mask
    feat = torch.randn(1, IN_CHANNEL, IN_HEIGHT, IN_WIDTH) * 10
    offset_y = torch.randn(1, KERNEL_SIZE*KERNEL_SIZE, OUT_HEIGHT, OUT_WIDTH)
    offset_x = torch.randn(1, KERNEL_SIZE*KERNEL_SIZE, OUT_HEIGHT, OUT_WIDTH)
    mask = torch.randn(1, KERNEL_SIZE*KERNEL_SIZE, OUT_HEIGHT, OUT_WIDTH)

    # Run deform_op
    perm_offset_y = offset_y.permute(1, 0, 2, 3)
    perm_offset_x = offset_x.permute(1, 0, 2, 3)
    offset = torch.cat((perm_offset_y, perm_offset_x), dim=1)
    offset = torch.reshape(offset, (1, 2*KERNEL_SIZE*KERNEL_SIZE, OUT_HEIGHT, OUT_WIDTH))
    deform_op.eval()
    with torch.no_grad():
        out_deform_op = deform_op(feat, offset, torch.sigmoid(mask))

    # Run deform_with_gs
    deform_with_gs.eval()
    with torch.no_grad():
        out_deform_with_gs = deform_with_gs(feat, offset, torch.sigmoid(mask))

    # Export deform_with_gs model
    input_names  = ["feat", "offset", "mask"]
    output_names = ["output"]
    model_input = (
        feat,
        offset,
        mask)
    torch.onnx.export(deform_with_gs,
                      model_input,
                      "deform_conv_with_gs.onnx",
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16)

    # simplify the onnx model
    from onnxsim import simplify
    import onnx
    onnx_model, simplify_ok = simplify("deform_conv_with_gs.onnx")
    onnx.save(onnx_model, "deform_conv_with_gs.onnx")

    # Check differences between deform_op and deform_with_gs
    diff = out_deform_with_gs - out_deform_op
    max_diff = torch.max(torch.flatten(torch.abs(diff)))
    rel_diff = torch.flatten(torch.abs(diff) / (torch.abs(out_deform_op) + 1e-8))
    mean_rel_diff = torch.mean(rel_diff)
    if torch.sum((mean_rel_diff > 1e-4) == True) > 0:
        test_output = False
    else:
        test_output = True

    max_diff = round(max_diff.item(), 8)
    mean_rel_diff = round(mean_rel_diff.item(), 8)
    if test_output:
        print(f'\n\ntorhcvision based DeformConvOP2d and DeformConvWithGS2d matches. Max difference = {max_diff}, Mean Rel difference = {mean_rel_diff}')
        print("DeformConv: Test PASSED")
    else:
        print(f'\n\ntorhcvision based DeformConvOP2d and DeformConvWithGS2d do not match!. Max difference = {max_diff}, Mean Rel difference = {mean_rel_diff}')
        assert test_output, "DeformConv: Test FAILED"


def run_test_dcnv2():
    # Set test params
    IN_HEIGHT = 58
    IN_WIDTH = 100
    OUT_HEIGHT = IN_HEIGHT
    OUT_WIDTH = IN_WIDTH

    IN_CHANNEL = 256
    OUT_CHANNEL = 256
    KERNEL_SIZE = 3
    STRIDE = 1
    PADDING = 1
    DILATION = 1
    BIAS = True
    GROUPS = 1
    DEFORM_GROUPS = 1
    INTP_MODE = 'bilinear'

    # DCNv2 / ModulatedDeformConv2dPack
    # from https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/modulated_deform_conv.py
    dcnv2_op = MMCVDCNv2(IN_CHANNEL,
                     OUT_CHANNEL,
                     kernel_size=KERNEL_SIZE,
                     stride=STRIDE,
                     padding=PADDING,
                     dilation=DILATION,
                     groups=GROUPS,
                     bias=BIAS,
                     deform_groups=DEFORM_GROUPS)

    dcnv2_with_gs  = DCNWithGSv2(IN_CHANNEL,
                     OUT_CHANNEL,
                     kernel_size=KERNEL_SIZE,
                     stride=STRIDE,
                     padding=PADDING,
                     dilation=DILATION,
                     groups=GROUPS,
                     bias=BIAS,
                     mode=INTP_MODE,
                     deform_groups=DEFORM_GROUPS,
                     offset_clip=None)

    # For evaluation, make weight and bias of two models the same
    torch.nn.init.normal_(dcnv2_op.conv_offset.weight)
    torch.nn.init.normal_(dcnv2_op.weight)
    dcnv2_with_gs.copy_weights(dcnv2_op)

    # Input to the model: feature map, offset_y, offste_x, mask
    feat = torch.randn(1, IN_CHANNEL, IN_HEIGHT, IN_WIDTH) * 10

    dcnv2_op.eval()
    with torch.no_grad():
        out_deform_op = dcnv2_op(feat)

    # Run dcnv2_with_gs
    dcnv2_with_gs.eval()
    with torch.no_grad():
        out_deform_with_gs = dcnv2_with_gs(feat)

    # Check differences between dcnv2_op and dcnv2_with_gs
    diff = out_deform_with_gs - out_deform_op
    max_diff = torch.max(torch.flatten(torch.abs(diff)))
    rel_diff = torch.abs(diff) / (torch.abs(out_deform_op) + 1e-8)
    mean_rel_diff = torch.mean(torch.flatten(rel_diff))
    if torch.sum((mean_rel_diff > 1e-4) == True) > 0:
        test_output = False
    else:
        test_output = True

    max_diff = round(max_diff.item(), 8)
    mean_rel_diff = round(mean_rel_diff.item(), 8)
    if test_output:
        print(f'\n\nmmcv.ops.ModulatedDeformConv2dPack and and DCNWithGSv2 matches. Max difference = {max_diff}, Mean Rel difference = {mean_rel_diff}')
        print("DCNv2: Test PASSED")
    else:
        print(f'\n\nmmcv.ops.ModulatedDeformConv2dPack and DCNWithGSv2 do not match!. Max difference = {max_diff}, Mean Rel difference = {mean_rel_diff}')
        assert test_output, "DCNv2: Test FAILED - one reason for failure could be offset_clip passed to DCNWithGSv2. Use None to check if the test passes"

    # Export DCNWithGSv2 model with offset_clip
    # in practical implementation, we may need to use offset_clip to limit the range of offset
    dcnv2_with_gs_clip  = DCNWithGSv2(IN_CHANNEL,
                     OUT_CHANNEL,
                     kernel_size=KERNEL_SIZE,
                     stride=STRIDE,
                     padding=PADDING,
                     dilation=DILATION,
                     groups=GROUPS,
                     bias=BIAS,
                     mode=INTP_MODE,
                     deform_groups=DEFORM_GROUPS,
                     offset_clip=32)

    input_names  = ["feat"]
    output_names = ["output"]
    model_input = (feat,)
    torch.onnx.export(dcnv2_with_gs_clip,
                      model_input,
                      "dcnv2_with_gs.onnx",
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16)

    # simplify the onnx model
    from onnxsim import simplify
    import onnx
    onnx_model, simplify_ok = simplify("dcnv2_with_gs.onnx")
    onnx.save(onnx_model, "dcnv2_with_gs.onnx")


if __name__ == "__main__":
    run_test_deform_op()
    run_test_dcnv2()