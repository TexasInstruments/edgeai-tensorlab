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

import torch

from ....xnn import layers


#################################################################################################
# # Note: currently the creation of lite models follows the following syntax:
# def mobilenet_v3_large_lite( **kwargs):
#     return xnn.model_surgery.create_lite_model(mobilenet_v3_large, **kwargs)
# # i.e., the model sugery is handled after the model creation is completely over.
# #
# # yet another potential option will be to handle model_sugery inside the model
# # and then to pass a suitable model_surgery function from outside
# # this will need model_surgery to be handled n every model,
# # so not using this method for the time being
# def mobilenet_v3_large_lite( **kwargs):
#     return mobilenet_v3_large(model_surgery=xnn.model_surgery._convert_to_lite_model, **kwargs)
#################################################################################################


# __all__ = ['_convert_to_lite_model', 'create_lite_model', '_get_replacement_dict_default']


def _check_dummy(current_m):
    '''
    a dummy check function to demonstrate a posible key in replacements dict.
    replace with your condition check.
    '''
    return isinstance(current_m, torch.nn.Identity)


def _replace_dummy(current_m):
    '''
    a dummy replace function to demonstrate a posible value in replacements dict.
    replace with the new object that you wish to replace
    Note: if the output is same as input, no replacement is done.
    '''
    return current_m


def _replace_conv2d(current_m=None, groups_dw=None, group_size_dw=None,
                    with_normalization=(True,False), with_activation=(True,False)):
    '''replace a regular convolution such as 3x3 or 5x5 with depthwise separable.
    with_normalization: introduce a normaliztion after point convolution default: (True,False)
    with_activation: introduce an activation fn between the depthwise and pointwise default: (True,False).
    - the above default choices are done to mimic a regulr convolution at a much lower complexity.
    - while replacing a regular convolution with a dws, we may not want to put a bn at the end,
    as a bn may be already there after this conv. same case with activation

    Note: it is also possible to do checks inside this function and if we dont want to replace, return the original module
    '''
    assert current_m is not None, 'for replacing Conv2d the current module must be provided'
    if isinstance(current_m, torch.nn.Conv2d) and (groups_dw is not None or group_size_dw is not None):
        kernel_size = current_m.kernel_size if isinstance(current_m.kernel_size, (list,tuple)) else (current_m.kernel_size,current_m.kernel_size)
        if (current_m.groups == 1 and current_m.in_channels >= 16 and kernel_size[0] > 1 and kernel_size[1] > 1):
            with_bias = current_m.bias is not None
            normalization = (current_m.with_normalization[0],current_m.with_normalization[1]) if hasattr(current_m, "with_normalization") else \
                (with_normalization[0],with_normalization[1])
            activation = (current_m.with_activation[0],current_m.with_activation[1]) if hasattr(current_m, "with_activation") else \
                (with_activation[0],with_activation[1])
            new_m = layers.ConvDWSepNormAct2d(in_planes=current_m.in_channels, out_planes=current_m.out_channels,
                        kernel_size=current_m.kernel_size, stride=current_m.stride, groups_dw=groups_dw, group_size_dw=group_size_dw,
                        bias=with_bias, normalization=normalization, activation=activation)
            return new_m
        #
    #
    return current_m


def _replace_group_norm(current_m=None):
    assert current_m is not None, 'for replacing GroupNorm the current module must be provided'
    if isinstance(current_m, torch.nn.GroupNorm):
        new_m = torch.nn.BatchNorm2d(num_features=current_m.num_channels)
        return new_m
    #
    return current_m


class SequentialMaxPool2d(torch.nn.Sequential):
    def __init__(self, kernel_size, stride):
        num_pool = (kernel_size - 1) // 2
        pool_modules = []
        strides_remaining = stride
        for n in range(num_pool):
            strides_remaining = strides_remaining // 2
            s = 2 if strides_remaining > 0 else 1
            pool_modules += [torch.nn.MaxPool2d(kernel_size=3, stride=s, padding=1)]
        #
        # reverse the list, so that the one with larger stride comes last
        pool_modules = pool_modules[::-1]
        super().__init__(*pool_modules)
        self.kernel_size = kernel_size
        self.stride = stride


def replace_maxpool2d(m):
    if m.kernel_size > 3:
        new_m = SequentialMaxPool2d(m.kernel_size, m.stride)
    else:
        new_m = m
    #
    return new_m
