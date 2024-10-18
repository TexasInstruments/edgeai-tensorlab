#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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

from edgeai_torchmodelopt import xnn
import torch

# add prediction and final upsample blocks to edgeailite models
def add_lite_prediction_modules(self, model_config, current_channels, module_names, UpsampleClass = xnn.layers.UpsampleWith):
    # prediction and upsample
    if self.model_config.final_prediction:
        # use UpsampleWithGeneric() instead of UpsampleWith(), to break down large upsampling factors to multiples of 4 and 2
        # useful if scale_factor other than 4 and 2 are not supported.

        # can control the range of final output with output_range
        output_range = model_config.output_range
        final_activation = xnn.layers.get_fixed_hardtanh_type(output_range[0],output_range[1]) \
            if (output_range is not None) else False
        upstride2 = model_config.shortcut_strides[0]

        if model_config.prediction_channels is not None and self.model_config.final_upsample and \
                self.model_config.interpolation_type in ('deconv','upsample_conv','subpixel_conv'):
            prediction_channels = max(model_config.output_channels, model_config.prediction_channels)
        else:
            prediction_channels = model_config.output_channels
        #
        # prediction followed by conventional interpolation
        ConvXWSepBlock = xnn.layers.ConvGWSepNormAct2d if model_config.groupwise_sep else xnn.layers.ConvDWSepNormAct2d
        group_size_dw = model_config.group_size_dw if hasattr(model_config, 'group_size_dw') else None
        
        # Do not use output range yet as this is not final layer. Next upsample has conv layer.
        final_activation_conv_block = final_activation
        if (model_config.interpolation_type == 'upsample_dwconv') and self.model_config.final_upsample:
            final_activation_conv_block = False 

        pred = ConvXWSepBlock(current_channels, prediction_channels, kernel_size=3,
                                   normalization=((not model_config.linear_dw),False),
                                   activation=(False,final_activation_conv_block), groups=1, group_size_dw=group_size_dw)
        setattr(self, module_names[0], pred)

        if self.model_config.final_upsample:
            upstride2 = (upstride2//self.model_config.target_input_ratio)
            if upstride2 > 1:
                upsample2 = UpsampleClass(prediction_channels, model_config.output_channels, upstride2,
                                          model_config.interpolation_type, model_config.interpolation_mode,
                                          is_final_layer=True, final_activation=final_activation)
            else:
                upsample2 = xnn.layers.BypassBlock()
            #
            setattr(self, module_names[1], upsample2)
        #
    #

class UpsampleWithSlice(torch.nn.Module):
    def __init__(self, input_channels=None, output_channels=None, upstride=None, interpolation_type='upsample', interpolation_mode='bilinear',
               is_final_layer=False, final_activation=True):
        super().__init__()
        self.input_channels = input_channels
        self.slice_size = 1
        self.upsample2_slice1 = xnn.layers.UpsampleWith(self.slice_size, self.slice_size, upstride,
                                         interpolation_type, interpolation_mode,
                                         is_final_layer=is_final_layer, final_activation=final_activation)
        self.upsample2_slice2 = xnn.layers.UpsampleWith(input_channels - self.slice_size, output_channels - self.slice_size, upstride,
                                         interpolation_type, interpolation_mode,
                                         is_final_layer=is_final_layer, final_activation=final_activation)
        self.cat = xnn.layers.CatBlock()

    def forward(self, x):
        x1, x2 = x[:, :self.slice_size, ...], x[:, self.slice_size:self.input_channels, :]
        x1 = self.upsample2_slice1(x1)
        x2 = self.upsample2_slice2(x2)
        x = torch.cat((x1, x2), dim=1)
        return x