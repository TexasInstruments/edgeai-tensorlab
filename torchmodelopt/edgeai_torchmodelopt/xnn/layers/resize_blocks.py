#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
from .deconv_blocks import *


##############################################################################################
# Newer Resize/Upsample mopdules to resize with scale factor that outputs a simple onnx graph.
# Please use these resize_with function or and ResizeWith/UpsampleWith modules instead of the
# older ResizeTo, UpsampleTo. The older modules may be removed in a later version.
##############################################################################################

# resize with output size or scale factor
def resize_with(x, size=None, scale_factor=None, mode='nearest', align_corners=None, force_scale_factor=False,
                recompute_scale_factor=False):
    assert size is None or scale_factor is None, 'both size and scale_factor must not be specified'
    assert size is not None or scale_factor is not None, 'at least one of size or scale factor must be specified'
    # print(type(x))
    if not isinstance(x, (torch.Tensor,torch.fx.Proxy)):
        raise RuntimeError('must provide a single tensor as input')

    # check if interpolate has been monkey-patched
    if hasattr(torch.nn.functional, '_interpolate_orig'):
        interpolate_fn = torch.nn.functional._interpolate_orig
    else:
        interpolate_fn = torch.nn.functional.interpolate
    #

    try:
        # Newer PyTorch versions support recompute_scale_factor = False, that exports a clean onnx graph
        # Attempt it first. Works with onnx opset_version=9 & opset_version=11
        if scale_factor is None and force_scale_factor:
            size = size[-2:] if len(size) > 2 else size
            x_size = x.data.size()[-2:]
            try:
                # caused error for pt2e
                scale_factor = [float(torch.true_divide(size[0],x_size[0])), float(torch.true_divide(size[1],x_size[1]))]
            except:
                scale_factor = [size[0]/x_size[0], size[1]/x_size[1]]
            size = None
        #
        y = interpolate_fn(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    except:
        if torch.onnx.is_in_onnx_export():
            warnings.warn('To generate a simple Upsample/Resize ONNX graph, please use pytorch>=1.5 or the nightly, as explained here: https://pytorch.org/')
        #
        if scale_factor is not None:
            # A workaround for older versions of PyTorch to generate a clean onnx graph with onnx opset_version=9 (may not work in onnx opset_version=11).
            # Generate size as a tuple and pass it - as onnx export inserts scale_factor if size is a non-tensor.
            scale_factor = (scale_factor,scale_factor) if not isinstance(scale_factor,(list,tuple)) else scale_factor
            size = [int(round(float(shape)*scale)) for shape, scale in zip(x.shape[2:],scale_factor)]
        #
        y = interpolate_fn(x, size=size, mode=mode, align_corners=align_corners)
    #
    return y


# always use scale factor to do the rescaling. if scale factor is not provided, generate it from the size.
def resize_with_scale_factor(x, size=None, scale_factor=None, mode='nearest', align_corners=None,
                             recompute_scale_factor=False):
    y = resize_with(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners,
                    force_scale_factor=True, recompute_scale_factor=recompute_scale_factor)
    return y


class ResizeWith(torch.nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        y = resize_with(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners= self.align_corners)
        return y


def UpsampleWith(input_channels=None, output_channels=None, upstride=None, interpolation_type='upsample', interpolation_mode='bilinear',
               is_final_layer=False, final_activation=True):
    '''
         is_final_layer: Final layer in a edgeailite network should not typically use BatchNorm for best accuracy
         final_activation: use to control the range of final layer if required
     '''
    if interpolation_type == 'upsample':
        upsample = ResizeWith(scale_factor=upstride, mode=interpolation_mode)
    else:
        assert upstride is not None, 'upstride must not be None in this interpolation_mode'
        assert input_channels is not None, 'input_channels must not be None in this interpolation_mode'
        assert output_channels is not None, 'output_channels must not be None in this interpolation_mode'
        final_norm = (False if is_final_layer else True)
        normalization = (True, final_norm)
        activation = (False, final_activation)
        #this should be removed
        if interpolation_type == 'upsample_dwconv' and not is_final_layer:
            interpolation_type = 'upsample_dwconv3_dil3'

        if interpolation_type == 'deconv':
            upsample = [DeConvDWSepNormAct2d(input_channels, output_channels, kernel_size=upstride * 2, stride=upstride,
                                      normalization=normalization, activation=activation)]
        elif interpolation_type == 'upsample_conv':
            upsample = [ResizeWith(scale_factor=upstride, mode=interpolation_mode),
                        ConvDWSepNormAct2d(input_channels, output_channels, kernel_size=int(upstride * 1.5 + 1),
                                      normalization=normalization, activation=activation)]
        elif interpolation_type == 'upsample_dwconv':
            upsample = [ResizeWith(scale_factor=upstride, mode=interpolation_mode),
                        ConvDWNormAct2d(input_channels, output_channels, kernel_size=int(upstride * 1.5 + 1),
                                           normalization=False, activation=final_activation, bias=True)]
        elif interpolation_type == 'upsample_dwconv3_dil3':
            upsample = [ResizeWith(scale_factor=upstride, mode=interpolation_mode),
                        ConvDWNormAct2d(input_channels, output_channels, kernel_size=3, dilation=3,
                                           normalization=False, activation=final_activation, bias=True)]
        elif (interpolation_type == 'subpixel_conv' or interpolation_type == 'pixel_shuffle'):
            upsample = [ConvDWSepNormAct2d(input_channels, output_channels*upstride*upstride, kernel_size=int(upstride + 1),
                                      normalization=normalization, activation=activation),
                        torch.nn.PixelShuffle(upscale_factor=int(upstride))]
        else:
            assert False, f'invalid interpolation_type: {interpolation_type}'
        #
        upsample = torch.nn.Sequential(*upsample)
    #
    return upsample



class UpsampleWithGeneric(torch.nn.Module):
    def __init__(self, input_channels=None, output_channels=None, upstride=None, interpolation_type='upsample', interpolation_mode='bilinear',
                 is_final_layer=False, final_activation=False):
        '''
            A Resize module that breaks downscale factors > 4 to multiples of 2 and 4
            is_final_layer: Final layer in a edgeailite network should not typically use BatchNorm
            final_activation: use to control the range of final layer if required
        '''
        super().__init__()
        self.upsample_list = torch.nn.ModuleList()
        while upstride >= 2:
            upstride_layer = 4 if upstride > 4 else upstride
            upsample = UpsampleWith(input_channels, output_channels, upstride_layer, interpolation_type, interpolation_mode,
                                  is_final_layer=is_final_layer, final_activation=final_activation)
            self.upsample_list.append(upsample)
            upstride = upstride//4

    def forward(self, x):
        assert not isinstance(x, (list,tuple)), 'must provide a single tensor as input'
        for idx, upsample in enumerate(self.upsample_list):
            x = upsample(x)
        #
        return x



##############################################################################################
# The following modules will be deprecated in a later version. Please use the modules above.
##############################################################################################

class ResizeTo(torch.nn.Module):
    def __init__(self, mode='bilinear'):
        '''
            Resize to the target size
        '''
        super().__init__()
        self.mode = mode

    def forward(self, input):
        assert isinstance(input, (list,tuple)), 'must provide two tensors - input and target'
        x = input[0]
        xt = input[1]
        target_size = (int(xt.size(2)), int(xt.size(3)))
        y = torch.nn.functional.interpolate(x, size=target_size, mode=self.mode)
        return y


def UpsampleTo(input_channels=None, output_channels=None, upstride=None, interpolation_type='upsample', interpolation_mode='bilinear',
               is_final_layer=False, final_activation=True):
    '''
         is_final_layer: Final layer in a edgeailite network should not typically use BatchNorm for best accuracy
         final_activation: use to control the range of final layer if required
     '''
    if interpolation_type == 'upsample':
        upsample = ResizeTo(mode=interpolation_mode)
    else:
        assert upstride is not None, 'upstride must not be None in this interpolation_mode'
        assert input_channels is not None, 'input_channels must not be None in this interpolation_mode'
        assert output_channels is not None, 'output_channels must not be None in this interpolation_mode'
        final_norm = (False if is_final_layer else True)
        normalization = (True, final_norm)
        activation = (False, final_activation)
        if interpolation_type == 'deconv':
            upsample = [SplitListTakeFirst(),
                        DeConvDWSepNormAct2d(input_channels, output_channels, kernel_size=upstride * 2, stride=upstride,
                                      normalization=normalization, activation=activation)]
        elif interpolation_type == 'upsample_conv':
            upsample = [ResizeTo(mode=interpolation_mode),
                        ConvDWSepNormAct2d(input_channels, output_channels, kernel_size=int(upstride * 1.5 + 1),
                                      normalization=normalization, activation=activation)]
        elif interpolation_type == 'subpixel_conv':
            upsample = [SplitListTakeFirst(),
                        ConvDWSepNormAct2d(input_channels, output_channels*upstride*upstride, kernel_size=int(upstride + 1),
                                      normalization=normalization, activation=activation),
                        torch.nn.PixelShuffle(upscale_factor=int(upstride))]
        else:
            assert False, f'invalid interpolation_type: {interpolation_type}'
        #
        upsample = torch.nn.Sequential(*upsample)
    #

    return upsample


class UpsampleGenericTo(torch.nn.Module):
    def __init__(self, input_channels=None, output_channels=None, upstride=None, interpolation_type='upsample', interpolation_mode='bilinear',
                 is_final_layer=False, final_activation=False):
        '''
            A Resize module that breaks downscale factors > 4 to multiples of 2 and 4
            is_final_layer: Final layer in a edgeailite network should not typically use BatchNorm
            final_activation: use to control the range of final layer if required
        '''
        super().__init__()
        self.upsample_list = torch.nn.ModuleList()
        self.upstride_list = []
        while upstride >= 2:
            upstride_layer = 4 if upstride > 4 else upstride
            upsample = UpsampleTo(input_channels, output_channels, upstride_layer, interpolation_type, interpolation_mode,
                                  is_final_layer=is_final_layer, final_activation=final_activation)
            self.upsample_list.append(upsample)
            self.upstride_list.append(upstride_layer)
            upstride = upstride//4

    def forward(self, x):
        assert isinstance(x, (list,tuple)) and len(x)==2, 'input must be a tuple/list of size 2'
        x, x_target = x
        xt_shape = x.shape
        for idx, (upsample, upstride) in enumerate(zip(self.upsample_list,self.upstride_list)):
            xt_shape = (xt_shape[0], xt_shape[1], xt_shape[2]*upstride, xt_shape[3]*upstride)
            xt = torch.zeros(xt_shape).to(x.device)
            x = upsample((x, xt))
            xt_shape = x.shape
        #
        return x

