import torch
from torch import nn
from . import custom_modules, custom_surgery_functions
from typing import Union, Dict, Any
from copy import deepcopy
from torch.fx import GraphModule, symbolic_trace
from inspect import isfunction
from .replacer import graph_pattern_replacer


_unsupported_module_dict={
    custom_modules.SEModule() : nn.Identity(),
    custom_modules.SEModule1() : nn.Identity(),
    nn.ReLU(inplace=True):nn.ReLU(),
    nn.Dropout(inplace=True):nn.Dropout(),
    nn.Hardswish():nn.ReLU(),
    nn.ReLU6():nn.ReLU(),
    nn.GELU():nn.ReLU(),
    nn.SiLU():nn.ReLU(),
    nn.LeakyReLU():nn.ReLU(),
    nn.Hardsigmoid():nn.ReLU(),
    custom_modules.Focus():custom_modules.ConvBNRModule(3,12,(5,5),(2,2),2),
    'upsample':custom_surgery_functions.replace_resize_with_scale_factor,
    'maxpool_ge_5':custom_surgery_functions.replace_maxpool2d_kernel_size_ge_5,
    'avgpool_ge_5':custom_surgery_functions.replace_avgpool2d_kernel_size_ge_5,
    'conv_ge_7':custom_surgery_functions.replace_conv2d_kernel_size_ge_7,
}


def _is_replacable(pattern:Union[GraphModule,nn.Module,callable]):
    if not isinstance(pattern,GraphModule):
        pattern=symbolic_trace(pattern)
    #TODO
    return True


def replace_unsuppoted_layers(model:nn.Module,replacement_dict:Dict[Any,Union[nn.Module,callable]]=None):
    replacement_dict = replacement_dict or _unsupported_module_dict
    model=deepcopy(model)
    for pattern, replacement in replacement_dict.items():
        if isfunction(replacement):
            model=replacement(model)
        else:
            if pattern.__class__.__name__ in dir(nn):
                pattern= custom_modules.InstaModule(pattern)
            model=graph_pattern_replacer(model,pattern,replacement)
    return model


def get_replacement_dict_default():
    return _unsupported_module_dict


class SurgeryModule(torch.nn.Module):
    def __int__(self, model, replace_unsupported_ops=True, replacement_dict=None):
        super().__init__()
        self.module = replace_unsuppoted_layers(model, replacement_dict)

    def get_replacement_dict_default(self):
        return get_replacement_dict_default()

