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

import copy
import functools
import torch
import inspect

from ....ops.misc import SqueezeExcitation
from .. import utils
from .. import layers


__all__ = {'convert_to_lite_model', 'create_lite_model', 'get_replacements_dict'}


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
#     return mobilenet_v3_large(model_surgery=xnn.model_surgery.convert_to_lite_model, **kwargs)
#################################################################################################


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


def _replace_conv2d(current_m=None, groups_dw=None, group_size_dw=1,
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
    if isinstance(current_m, torch.nn.Conv2d):
        kernel_size = current_m.kernel_size if isinstance(current_m.kernel_size, (list,tuple)) else (current_m.kernel_size,current_m.kernel_size)
        if (current_m.groups == 1 and current_m.in_channels >= 16 and kernel_size[0] > 1 and kernel_size[1] > 1) and \
            (groups_dw is not None or group_size_dw is not None):
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


'''
A dictionary with the fllowing structure.
key: a torch.nn.Module that has to be replaced OR a callable which takes a module as input and returns boolean
value: a list. the fist entry is a constructor or a callable that creates the replacement module
               the remaining entries are properties that have to be copied from old module to newly created module.
'''
REPLACEMENTS_DICT_DEFAULT = {
    torch.nn.ReLU6: [torch.nn.ReLU, 'inplace'],
    torch.nn.Hardswish: [torch.nn.ReLU, 'inplace'],
    torch.nn.SiLU: [torch.nn.ReLU, 'inplace'],
    SqueezeExcitation: [torch.nn.Identity],
    torch.nn.Conv2d: [_replace_conv2d],
    # just a dummy entry to show that the key and value can be a functions
    # the key should return a boolean and the first entry of value(list) should return an instance of torch.nn.Module
    _check_dummy: [_replace_dummy]
}


def get_replacements_dict():
    return REPLACEMENTS_DICT_DEFAULT


# this function can be used after creating the model to transform it into a lite model.
def create_lite_model(model_function, pretrained_backbone_names=None, pretrained=None, model_urls_dict=None,
                      model_name_lite=None, replacements_dict=None, **kwargs):
    model_name_lite = model_name_lite or f'{model_function.__name__}_lite'
    lookup_pretrained = pretrained is True and model_urls_dict is not None and model_name_lite in model_urls_dict
    pretrained = model_urls_dict[model_name_lite] if lookup_pretrained else pretrained
    model = _create_lite_model_impl(model_function, pretrained_backbone_names, pretrained=pretrained,
                                   replacements_dict=replacements_dict, **kwargs)
    return model


def _create_lite_model_impl(model_function, pretrained_backbone_names=None, groups_dw=None, group_size_dw=1,
                           with_normalization=(True,False), with_activation=(True,False), replacements_dict=None, **kwargs):
    pretrained = kwargs.pop('pretrained', None)
    pretrained_backbone = kwargs.pop('pretrained_backbone', None)
    # if pretrained is set to true, we will try to hanlde it inside model
    if pretrained_backbone is not None:
        model = model_function(pretrained=(pretrained is True), pretrained_backbone=(pretrained_backbone is True), **kwargs)
    else:
        model = model_function(pretrained=(pretrained is True), **kwargs)
    #
    model = convert_to_lite_model(model, groups_dw=groups_dw, group_size_dw=group_size_dw,
                                  with_normalization=with_normalization, with_activation=with_activation,
                                  replacements_dict=replacements_dict)
    if pretrained and pretrained is not True:
        utils.load_weights(model, pretrained, state_dict_name=['state_dict', 'model'])
    elif pretrained_backbone and pretrained_backbone is not True:
        pretrained_backbone_names = pretrained_backbone_names if pretrained_backbone_names else {'^features.':'backbone.'}
        utils.load_weights(model, pretrained_backbone, state_dict_name=['state_dict', 'model'],
                           change_names_dict=pretrained_backbone_names)
    #
    return model


def convert_to_lite_model(model, inplace=True, groups_dw=None, group_size_dw=1,
                          with_normalization=(True,False), with_activation=(True,False),
                          replacements_dict=None):
    '''
    with_normalization: whether to insert BN after replacing 3x3/5x5 conv etc. with dw-seperable conv
    with_activation: whether to insert ReLU after replacing conv with dw-seperable conv
    '''
    replacements_dict = replacements_dict or REPLACEMENTS_DICT_DEFAULT
    model = model if inplace else copy.deepcopy(inplace)
    num_trails = len(list(model.modules()))
    for trial_id in range(num_trails):
        for p_name, p in model.named_modules():
            is_replaced = False
            for c_name, c in p.named_children():
                # replacing Conv2d is not trivial. it needs several arguments
                if isinstance(c, torch.nn.Conv2d):
                    replace_kwargs = dict(groups_dw=groups_dw, group_size_dw=group_size_dw,
                                          with_normalization=with_normalization, with_activation=with_activation)
                else:
                    replace_kwargs = dict()
                #
                is_replaced = _replace_with_new_module(p, c_name, c, replacements_dict, **replace_kwargs)
                if is_replaced:
                    break
                #
            #
            if is_replaced:
                break
            #
        #
    #
    return model


def _replace_with_new_module(parent, c_name, current_m, replacements_dict, **kwargs):
    for k_check, v_params in replacements_dict.items():
        assert callable(k_check), f'the key in replacements_dict must be a class or function: {k_check}'
        if inspect.isclass(k_check):
            do_replace = isinstance(current_m, k_check)
        else:
            do_replace = k_check(current_m)
        #
        if do_replace:
            # first entry is the constructor or a callable that constructs
            new_constructor = v_params[0]
            assert callable(new_constructor), f'the value in replacements_dict must be a class or function: {new_constructor}'
            # the parameters of the new moulde that has to be copied from current
            new_args = {}
            for k in v_params[1:]:
                new_args.update({k:getattr(current_m,k)})
            #
            # create the new module that replaces the existing
            if inspect.isclass(new_constructor):
                new_m = new_constructor(**new_args, **kwargs)
            else:
                new_m = new_constructor(current_m, **new_args, **kwargs)
            #
            # now initialize the new module and replace it in the parent
            if new_m is not current_m:
                _initialize_module(new_m)
                new_m.train(current_m.training)
                # requires_grad setting of the source is used for the newly created module
                requires_grad = None
                for param_cur in current_m.parameters():
                    requires_grad = requires_grad or param_cur.requires_grad
                #
                if requires_grad is not None:
                    for param_new in new_m.parameters():
                        param_new.requires_grad = requires_grad
                    #
                #
                setattr(parent, c_name, new_m)
                return True
            #
        #
    #
    return False


def _initialize_module(module):
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Linear):
            init_range = 1.0 / math.sqrt(m.out_features)
            torch.nn.init.uniform_(m.weight, -init_range, init_range)
            torch.nn.init.zeros_(m.bias)
        #
    #
