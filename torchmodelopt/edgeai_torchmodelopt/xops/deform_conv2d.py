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


class DeformConvOp2d(torchvision.ops.DeformConv2d):
    def __init__(self, *args, **kwargs):
        '''
        Deformable convolution operator (derived from torchvision.ops.DeformConv2d)
        '''
        mode = kwargs.pop('mode', 'bilinear')
        assert mode in ('bilinear',), 'mode should be one of: bilinear, nearest'
        super().__init__(*args, **kwargs)
        self.mode = mode
        warnings.warn('DeformConvOp2d or torchvision.ops.DeformConv2d is not our recommended '
            'Deformable Convolution implementation. Please use DeformConvWithGS2d instead.')


class DeformConvWithGS2d(torchvision.ops.DeformConv2d):
    '''
    Deformable convolution with Grid Sample
    '''
    def __init__(self, *args, **kwargs):
        mode = kwargs.pop('mode', 'bilinear')
        assert mode in ('bilinear', 'nearest'), 'mode should be one of: bilinear, nearest'
        super().__init__(*args, **kwargs)
        self.mode = mode

    def forward(self, feat, offset, mask):

        # [1,18,W,H] => 2 x [1,9,W,H]
        offset_y = offset[:,0::2,...]
        offset_x = offset[:,1::2,...]

        # 0. sigmod of mask
        mask = torch.sigmoid(mask)

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
            torch.arange(-(dilation_h*(fr-1))//2,
                          (dilation_h*(fr-1))//2 +1,
                         1, device=offset_y.device, dtype=offset_y.dtype),
            torch.arange(-(dilation_w*(fc-1))//2,
                          (dilation_w*(fc-1))//2 +1,
                         1, device=offset_y.device, dtype=offset_y.dtype))

        k_y = k_y.reshape(m, 1, 1)
        k_x = k_x.reshape(m, 1, 1)

        grid_y = grid_y + k_y
        grid_x = grid_x + k_x

        # 5. Normalizing sampling location
        grid_y = (offset_y + grid_y) / h # 1x9xHxW
        grid_x = (offset_x + grid_x) / w # 1x9xHxW

        # in (x, y) order
        offset_grid = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1) # 1x9xHxWx2

        # 6. Scale sampling location to [-1 to 1]
        offset_grid = 2 * offset_grid - 1
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


class DCNv2(torch.nn.Module):
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
        deform_layer_type = None):

        super().__init__()

        deform_layer_type = deform_layer_type or DeformConvWithGS2d

        ks = (kernel_size,kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.mask_out_channels = deform_groups * ks[0] * ks[1]
        self.offset_out_channels = self.mask_out_channels*2

        self.conv_offset = torch.nn.Conv2d(in_channels, (self.offset_out_channels+self.mask_out_channels),
                kernel_size, stride, padding, dilation, groups, bias)

        self.conv_deform = deform_layer_type(in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias, mode=mode)

    def forward(self, feat):
        offset_mask = self.conv_offset(feat)
        offset_yx = offset_mask[:,:self.offset_out_channels,...]
        mask = offset_mask[:,self.offset_out_channels:,...]
        return self.conv_deform(feat, offset_yx, mask)


###########################################################################################
# unit tests
###########################################################################################

def test_deform_op():
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
    DEFORM_GROUPS = 1
    INTP_MODE = 'bilinear'

    deform_op = DeformConvOp2d(IN_CHANNEL,
                                     OUT_CHANNEL,
                                     kernel_size=KERNEL_SIZE,
                                     stride=STRIDE,
                                     padding=PADDING,
                                     dilation=DILATION,
                                     bias=BIAS,
                                     groups=DEFORM_GROUPS)

    deform_with_gs  = DeformConvWithGS2d(IN_CHANNEL,
                                     OUT_CHANNEL,
                                     mode=INTP_MODE,
                                     kernel_size=KERNEL_SIZE,
                                     stride=STRIDE,
                                     padding=PADDING,
                                     dilation=DILATION,
                                     bias=BIAS,
                                     groups=DEFORM_GROUPS)

    # For evaluation, make weigth and bias of two models the same
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
        out_deform_with_gs = deform_with_gs(feat, offset, mask)

    # Check differences between deform_op and deform_with_gs
    diff = out_deform_with_gs - out_deform_op
    if torch.sum((abs(diff) > 1e-4) == True) > 0:
        test_output = False
    else:
        test_output = True

    # Export deform_with_gs model
    input_names  = ["feat", "offset", "mask"]
    output_names = ["output"]
    modelInput = (
        feat,
        offset,
        mask)
    torch.onnx.export(deform_with_gs,
                      modelInput,
                      "deform_conv_pytorch.onnx",
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16)

    # simplify the onnx model
    from onnxsim import simplify
    import onnx
    onnx_model, simplify_ok = simplify("deform_conv_pytorch.onnx")
    onnx.save(onnx_model, "deform_conv_pytorch.onnx")

    if test_output:
        print('\n\ndeform_op and deform_with_gs matches')
        print("DeformConv Op: Test PASSED")
    else:
        print('\n\ndeform_op and deform_with_gs do not match!')
        assert test_output, "DeformConv Op: Test FAILED"


def test_dcnv2():
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

    deform_op = DCNv2(IN_CHANNEL,
                     OUT_CHANNEL,
                     kernel_size=KERNEL_SIZE,
                     stride=STRIDE,
                     padding=PADDING,
                     dilation=DILATION,
                     groups=GROUPS,
                     bias=BIAS,
                     deform_groups=DEFORM_GROUPS,
                     deform_layer_type=DeformConvOp2d)

    deform_with_gs  = DCNv2(IN_CHANNEL,
                     OUT_CHANNEL,
                     kernel_size=KERNEL_SIZE,
                     stride=STRIDE,
                     padding=PADDING,
                     dilation=DILATION,
                     groups=GROUPS,
                     bias=BIAS,
                     mode=INTP_MODE,
                     deform_groups=DEFORM_GROUPS,
                     deform_layer_type=DeformConvWithGS2d)

    # For evaluation, make weigth and bias of two models the same
    deform_with_gs.conv_offset.weight = deform_op.conv_offset.weight
    deform_with_gs.conv_deform.weight = deform_op.conv_deform.weight
    if deform_with_gs.conv_offset.bias is not None:
        deform_with_gs.conv_offset.bias   = deform_op.conv_offset.bias
    if deform_with_gs.conv_deform.bias is not None:
        deform_with_gs.conv_deform.bias   = deform_op.conv_deform.bias

    # Input to the model: feature map, offset_y, offste_x, mask
    feat = torch.randn(1, IN_CHANNEL, IN_HEIGHT, IN_WIDTH) * 10

    deform_op.eval()
    with torch.no_grad():
        out_deform_op = deform_op(feat)

    # Run deform_with_gs
    deform_with_gs.eval()
    with torch.no_grad():
        out_deform_with_gs = deform_with_gs(feat)

    # Check differences between deform_op and deform_with_gs
    diff = out_deform_with_gs - out_deform_op
    if torch.sum((abs(diff) > 1e-4) == True) > 0:
        test_output = False
    else:
        test_output = True

    # Export deform_with_gs model
    input_names  = ["feat"]
    output_names = ["output"]
    model_input = (feat,)
    torch.onnx.export(deform_with_gs,
                      model_input,
                      "dcnv2_pytorch.onnx",
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16)

    # simplify the onnx model
    from onnxsim import simplify
    import onnx
    onnx_model, simplify_ok = simplify("dcnv2_pytorch.onnx")
    onnx.save(onnx_model, "dcnv2_pytorch.onnx")

    if test_output:
        print('\n\nDCNv2 with deform_op and and DCNv2 deform_with_gs matches')
        print("DCNv2: Test PASSED")
    else:
        print('\n\nDCNv2 with deform_op and and DCNv2 deform_with_gs do not match!')
        assert test_output, "DCNv2: Test FAILED"


if __name__ == "__main__":
    test_deform_op()
    test_dcnv2()