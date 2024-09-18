import torch
from torch import nn, fx
from typing import Any
import copy

from . import surgery, pruning, quantization

__all__ = ['apply_model_optimization', 'apply_model_surgery', 'apply_pruning', 'apply_quantization', 'prepare_model_for_onnx', 'apply_tranformation_to_submodules', 'TransformationWrapper']

def apply_model_optimization(model: nn.Module, example_inputs: list=[], example_kwargs: dict={}, model_surgery_dict: dict[str,Any]=None, pruning_dict: dict[str,Any]=None, quantization_dict: dict[str,Any]=None,):
    '''
    A wrappper function to apply surgery, pruning and quantization
    
    Args:
    model               : model to be optimized
    example_inputs      : a list of example inputs (default: [] -> No positional args)
    example_kwargs      : a dict for example kwargs for the model (if any) (default: {} -> No kwargs)
    model_surgery_dict  : a dict containing details for applying surgery (default: None -> No Surgery) 
    pruning_dict        : a dict containing details for applying pruning (default: None -> No Pruning) 
    quantization_dict   : a dict containing details for applying quantization (default: None -> No Quantization) 
    
    Note:
        1. All dict containing details for any kind of optimization must contain a int value for key 'version'
        2. All dict containing details for any kind of optimization can contain a function as a value for there respective keys as below
            i)      Surgery         -> 'surgery_func'
            ii)     Pruning         -> 'pruning_func'
            iii)    Quantization    -> 'quantization_func'
    '''
    model(*example_inputs,**example_kwargs)
    model_surgery_dict = copy.deepcopy(model_surgery_dict)
    pruning_dict = copy.deepcopy(pruning_dict)
    quantization_dict = copy.deepcopy(quantization_dict)
    if model_surgery_dict:
        assert isinstance(model_surgery_dict,dict), 'If model surgery is defined, it must be a dict.'
        assert 'version' in model_surgery_dict, 'A version key must be in model surgery dict to specify version'
        model_surgery_version = model_surgery_dict.pop('version')
        model_surgery_func = model_surgery_dict.pop('surgery_func',apply_model_surgery)
        model = model_surgery_func(model, example_inputs, example_kwargs, model_surgery_version, model_surgery_dict)

    if pruning_dict:
        assert isinstance(pruning_dict,dict), 'If pruning is defined, it must be a dict.'
        assert 'version' in pruning_dict, 'A version key must be in pruning dict to specify version'
        pruning_version = pruning_dict.pop('version')
        pruning_func = pruning_dict.pop('pruning_func',apply_pruning)
        model = pruning_func(model, example_inputs, example_kwargs, pruning_version, pruning_dict)

    if quantization_dict:
        assert isinstance(quantization_dict,dict), 'If quantization is defined, it must be a dict.'
        assert 'version' in quantization_dict, 'A version key must be in quantization dict to specify version'
        quantization_version = quantization_dict.pop('version')
        quantization_func = quantization_dict.pop('quantization_func',apply_quantization)
        model = quantization_func(model, example_inputs, example_kwargs, quantization_version, quantization_dict)
    return model


def prepare_model_for_onnx(orig_model: nn.Module, trained_model: nn.Module, example_inputs: list=[], example_kwargs: dict={}, model_surgery_dict: dict[str,Any]=None, pruning_dict: dict[str,Any]=None, quantization_dict: dict[str,Any]=None,):
    final_model = apply_model_optimization(orig_model, example_inputs, example_kwargs, model_surgery_dict, pruning_dict, quantization_dict)
    state_dict = trained_model.state_dict()
    final_model.load_state_dict(state_dict)
    return final_model

def apply_model_surgery(model: nn.Module, example_inputs: list=[], example_kwargs: dict={}, version: int=3, model_surgery_kwargs: dict[str,Any]=None, *args, **kwargs):
    model(*example_inputs, **example_kwargs)
    model_surgery_kwargs = copy.deepcopy(model_surgery_kwargs)
    if version in (1,2,3):
        assert 'replacement_dict' in model_surgery_kwargs, "A 'replacement_dict' must be present in surgery kwargs."
    if version == 1:
        replacement_dict = model_surgery_kwargs.pop('replacement_dict')
        model = surgery.v1.convert_to_lite_model(model, replacement_dict, **model_surgery_kwargs)
    elif version == 2:
        replacement_dict = model_surgery_kwargs.pop('replacement_dict')
        model = surgery.v2.convert_to_lite_fx(model, replacement_dict, example_inputs, example_kwargs, **model_surgery_kwargs)
    elif version == 3:
        replacement_dict = model_surgery_kwargs.pop('replacement_dict')
        model = surgery.v3.convert_to_lite_pt2e(model,replacement_dict, example_inputs, example_kwargs, **model_surgery_kwargs)
        # model,_ = torch._dynamo.export(model,aten_graph=False,assume_static_by_default=True)(*example_inputs,**example_kwargs)
    return model


def apply_pruning(model: nn.Module, example_inputs: list=[], example_kwargs: dict={}, version: int=3, pruning_kwargs: dict[str,Any]=None, *args, **kwargs):
    model(*example_inputs, **example_kwargs)
    pruning_kwargs = copy.deepcopy(pruning_kwargs)
    if version == 1:
        assert False, "Pruning is currently not supported in the legacy modules based method"
    if version in (2,3):
        required_kwargs = ['pruning_ratio', 'total_epochs', 'pruning_init_train_ep', 'pruning_class', 'pruning_type', 'pruning_global', 'pruning_m', 'p']
        for kwarg in required_kwargs:
            assert kwarg in pruning_kwargs, f"A '{kwarg}' must be present in pruning kwargs. Pruning Kwargs must contain the followings\n\t" + ', '.join(required_kwargs)
    if version == 2:
        model = pruning.v2.PrunerModule(model, **pruning_kwargs)
    elif version == 3:
        model = pruning.v3.PrunerModule(model, example_args=example_inputs, example_kwargs=example_kwargs, **pruning_kwargs)
    return model


def apply_quantization(model: nn.Module, example_inputs: list=[], example_kwargs: dict={}, version: int=2, quantization_kwargs: dict[str,Any]=None, *args, **kwargs):
    model(*example_inputs, **example_kwargs)
    quantization_kwargs = copy.deepcopy(quantization_kwargs)
    if version in (1,2,3):
        # for common kwargs
        pass
    if version == 1:
        model = quantization.v1.QuantTrainModule(model,*example_inputs,**quantization_kwargs)
    elif version == 2:
        quantization_method = quantization_kwargs.pop('quantization_method',None)
        if quantization_method == 'QAT':
            model = quantization.v2.QATFxModule(model,**quantization_kwargs)
        elif quantization_method in ('PTQ', 'PTC'):
            model = quantization.v2.PTCFxModule(model,**quantization_kwargs)
    elif version == 3:
        # assert False, "Quantization is currently not supported in the PT2E based method"
        quantization_method = quantization_kwargs.pop('quantization_method',None)
        if quantization_method == 'QAT':
            model = quantization.v3.QATPT2EModule(model, example_inputs=example_inputs, **quantization_kwargs)
        elif quantization_method in ('PTQ', 'PTC'):
            model = quantization.v3.PTQPT2EModule(model, example_inputs=example_inputs, **quantization_kwargs)
    return model


def apply_tranformation_to_submodules(model:nn.Module, transformation_dict: dict, *args, **kwargs):
    module_dict = dict(model.named_modules())
    for name, wrapper_fn in transformation_dict.items() :
        if name not in module_dict:
            continue
        module = module_dict[name]
        splits = name.rsplit('.',1)
        if len(splits) == 1:
            splits = '',splits[0]
        parent_module, sub_module_name = splits
        parent_module = model if parent_module == '' else module_dict[parent_module]
        module = wrapper_fn(module, *args, **kwargs)
        setattr(parent_module,sub_module_name,module)
    return model


class TransformationWrapper:
    def __init__(self, wrapper:callable, fn:callable=None):
        self.wrapper = wrapper
        self.fn = fn
        
    def __call__(self, *args, **kwargs):
        if self.fn is None:
            raise ValueError("the fn function to be wrapped should not be None")
        return self.wrapper(self.fn, *args, **kwargs)
