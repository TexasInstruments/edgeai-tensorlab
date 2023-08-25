import enum
import torch
from typing import Union, Dict, Any

from . import custom_modules, custom_surgery_functions,surgery
from .surgery import SurgeryModule, replace_unsuppoted_layers, get_replacement_dict_default


def convert_to_lite_fx(model:torch.nn.Module,replacement_dict:Dict[Any,Union[torch.nn.Module,callable]]=None, verbose_mode:bool=False, **kwargs):
    return replace_unsuppoted_layers(model, replacement_dict=replacement_dict, verbose_mode=verbose_mode, **kwargs)

