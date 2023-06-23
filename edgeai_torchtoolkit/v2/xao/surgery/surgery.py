import torch
from torch import nn
from torchvision import ops
from . import custom_modules, custom_surgery_functions
from typing import Union, Dict, Any
from copy import deepcopy
from torch.fx import GraphModule, symbolic_trace
from inspect import isfunction
from .replacer import graph_pattern_replacer,_get_parent_name as get_parent_name
from timm.models._efficientnet_blocks import SqueezeExcite


__all__ = ['replace_unsuppoted_layers', 'get_replacement_dict_default','SurgeryModule']


#put composite modules first, then primary module
_unsupported_module_dict={
    custom_modules.SEModule() : nn.Identity(),
    custom_modules.SEModule1() : nn.Identity(),
    SqueezeExcite(32) : nn.Identity(),
    SqueezeExcite(32,gate_layer=nn.Hardsigmoid) : nn.Identity(),
    'layerNorm':custom_surgery_functions.replace_layer_norm_and_permute, #based on convnext structure | not effective till date
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
    try:
        if not isinstance(pattern,GraphModule):
            pattern=symbolic_trace(pattern)
    except:
        return False
        #TODO
    return True


def replace_unsuppoted_layers(model:nn.Module,replacement_dict:Dict[Any,Union[nn.Module,callable]]=None):
    '''
    main function that does the surgery

    it does default surgery if no replacement dictionry is given
    replacement dictionry may contain
    keys            value
    Any             ->  Callabe     : any self-made surgery function 
    nn.Module       nn.Module       : any nn.Module pattern to replace with another nn.Module
    '''
    
    replacement_dict = replacement_dict or _unsupported_module_dict
    model=deepcopy(model)
    
    for pattern, replacement in replacement_dict.items():
        if isfunction(replacement):
            # for self-made surgery function 
            model=replacement(model)
        
        elif isinstance(model,nn.Module):
            #class of MOdule of 
            if isinstance(pattern,type):
                modules= dict(model.named_modules)
                for key_name, module in modules.items():
                    if isinstance(module,pattern):
                        parent_name, name= get_parent_name(key_name)
                        if isinstance(replacement, type):
                            replacement = replacement()
                        else:
                            replacement = deepcopy(replacement) 
                        modules[key_name] = replacement
                        modules[parent_name].__setattr__(name, modules[key_name])
            #for nn.Module 
            if pattern.__class__.__name__ in dir(nn):
                # if the pattern is present in nn directory,
                # a wrapper module is required, for successful 
                # surgery on that module
                pattern= custom_modules.InstaModule(pattern)
            
            #calls the main surgery function
            model=graph_pattern_replacer(model,pattern,replacement)
    return model

# returns default dictionary for replacement
def get_replacement_dict_default():
    return _unsupported_module_dict


class SurgeryModule(torch.nn.Module):
    '''
    wrapper module  for performing surgery on module

    it will do default surgery on model if no replacement dictionary is passed 
    while initializing.
    '''
    
    def __init__(self, model, replacement_dict=None) -> None:
        '''perform surgery on the model and creates a new model'''
        super().__init__()
        self.module = replace_unsuppoted_layers(model, replacement_dict)

    def forward(self,x,*args,**kwargs):
        '''
        atleast one input required 
        for more input, add them as a part of args
        '''
        return self.module(x,*args,**kwargs)

    def get_replacement_dict_default(self):
        '''returns the default replacement dictionary that can be updated further'''
        return get_replacement_dict_default()

