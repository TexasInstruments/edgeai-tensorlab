import torch
from torch import nn, fx
from typing import Any
import copy
import types

from . import model_optimzation_v1, model_optimzation_v2, model_optimzation_v3

__all__ = ['apply_model_optimization', 'apply_model_surgery', 'apply_pruning', 'apply_quantization', 
           'prepare_model_for_onnx', 'apply_tranformation_to_submodules', 'TransformationWrapper']

def apply_model_optimization(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, model_surgery_version=None, 
                             pruning_version=None, quantization_version=None, model_surgery_kwargs: dict[str, Any]=None, 
                             pruning_kwargs: dict[str, Any]=None, quantization_kwargs: dict[str, Any]=None, transformation_dict=None, copy_attrs=None):
    '''
    A wrapper function to apply surgery, pruning, and quantization
    
    Args:
    model               : model to be optimized
    example_inputs      : a list of example inputs (default: [] -> No positional args)
    example_kwargs      : a dict for example kwargs for the model (if any) (default: {} -> No kwargs)
    
    model_surgery_version  : an integer representing the version of model surgery to apply (default: None -> No surgery)
                            - 0: No surgery
                            - 1: Version 1 of model surgery
                            - 2: Version 2 of model surgery
                            - 3: Version 3 of model surgery
    
    pruning_version        : an integer representing the version of pruning to apply (default: None -> No pruning)
                            - 0: No pruning
                            - 1: Version 1 of pruning
                            - 2: Version 2 of pruning
                            - 3: Version 3 of pruning
    
    quantization_version   : an integer representing the version of quantization to apply (default: None -> No quantization)
                            - 0: No quantization
                            - 1: Version 1 of quantization
                            - 2: Version 2 of quantization
                            - 3: Version 3 of quantization
    
    model_surgery_kwargs   : a dict containing details for applying model surgery (default: None -> No surgery)
    
    pruning_kwargs         : a dict containing details for applying pruning (default: None -> No pruning)
    
    quantization_kwargs    : a dict containing details for applying quantization (default: None -> No quantization)
    '''
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
    model_surgery_version = model_surgery_version or (0 if model_surgery_kwargs is None else 2)
    pruning_version = pruning_version or (0 if pruning_kwargs is None else 2)
    quantization_version = quantization_version or (0 if quantization_kwargs is None else 2)
    copy_attrs= copy_attrs or []
    model(*example_inputs, **example_kwargs)
    model_surgery_kwargs = copy.deepcopy(model_surgery_kwargs)
    pruning_kwargs = copy.deepcopy(pruning_kwargs)
    quantization_kwargs = copy.deepcopy(quantization_kwargs)
    main_model_optimization_version = None
    main_kwargs = dict(transformation_dict=transformation_dict, copy_attrs=copy_attrs)
    
    if model_surgery_version != 0:
        assert model_surgery_version in (1, 2, 3)
        main_model_optimization_version = model_surgery_version
        main_kwargs.update(model_surgery_kwargs=model_surgery_kwargs)

    if pruning_version != 0:
        if main_model_optimization_version is None:
            assert pruning_version in (1, 2, 3)
            main_model_optimization_version = pruning_version
        else:
            assert pruning_version == main_model_optimization_version, f'''
            Different versions of model optimization techniques aren't supported together!
            Previously got Surgery v{main_model_optimization_version} but got Pruning v{pruning_version}.
            '''
        main_kwargs.update(pruning_kwargs=pruning_kwargs)

    if quantization_version != 0:
        if main_model_optimization_version is None:
            assert quantization_version in (1, 2, 3)
            main_model_optimization_version = quantization_version
        else:
            assert quantization_version == main_model_optimization_version, f'''
            Different versions of model optimization techniques aren't supported together!
            Previously got Surgery and Pruning v{main_model_optimization_version} but got Quantization v{quantization_version}.
            '''
        main_kwargs.update(quantization_kwargs=quantization_kwargs)
            
    if main_model_optimization_version == 1:
        pass
    elif main_model_optimization_version == 2:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            device = next(iter(model.module.named_parameters()))[1].device
            for i, example_input in enumerate(example_inputs):
                example_inputs[i] = example_input.to(device=device)
            for key, val in example_kwargs.items():
                if isinstance(val, list):
                    for i, v in enumerate(val):
                        val[i] = v.to(device=device)
                else:
                    val = val.to(device=device)
                example_kwargs[key] = val
            model.module = model_optimzation_v2.ModelOptimizationWrapperV2(model.module, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)
        else:
            model = model_optimzation_v2.ModelOptimizationWrapperV2(model, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)
    elif main_model_optimization_version == 3:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            device = next(iter(model.module.named_parameters()))[1].device
            for i, example_input in enumerate(example_inputs):
                example_inputs[i] = example_input.to(device=device)
            for key, val in example_kwargs.items():
                if isinstance(val, list):
                    for i, v in enumerate(val):
                        val[i] = v.to(device=device)
                else:
                    val = val.to(device=device)
                example_kwargs[key] = val
            model.module = model_optimzation_v3.ModelOptimizationWrapperV3(model.module, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)
        else:
            model = model_optimzation_v3.ModelOptimizationWrapperV3(model, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)
            
    return model


def prepare_model_for_onnx(orig_model: nn.Module, trained_model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, model_surgery_version=None, 
                           pruning_version=None, quantization_version=None, model_surgery_kwargs: dict[str,Any]=None, pruning_kwargs: dict[str,Any]=None, 
                           quantization_kwargs: dict[str,Any]=None, transformation_dict=None, copy_attrs=None):
    
    example_inputs = example_inputs or []
    example_kwargs = example_kwargs or {}
    model_surgery_version = model_surgery_version or (0 if model_surgery_kwargs is None else 2)
    pruning_version = pruning_version or (0 if pruning_kwargs is None else 2)
    quantization_version = quantization_version or (0 if quantization_kwargs is None else 2)
    
    orig_model = orig_model.to('cpu')
    example_inputs = [input_tensor.to('cpu') if isinstance(input_tensor, torch.Tensor) else input_tensor for input_tensor in example_inputs]
    example_kwargs = {key: value.to('cpu') if isinstance(value, torch.Tensor) else value for key, value in example_kwargs.items()}
    
    final_model = apply_model_optimization(orig_model, example_inputs, example_kwargs, model_surgery_version=model_surgery_version, pruning_version=pruning_version, 
                                           quantization_version=quantization_version, model_surgery_kwargs=model_surgery_kwargs, pruning_kwargs=pruning_kwargs, 
                                           quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)
    
    state_dict = trained_model.state_dict()
    final_model.load_state_dict(state_dict)
    
    return final_model


def apply_model_surgery(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, version: int=3, model_surgery_kwargs: dict[str,Any]=None, *args, **kwargs):
    
    from .. import surgery
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
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
        # model,_ = torch._dynamo.export(model,aten_graph=True,pre_dispatch=True,assume_static_by_default=True)(*example_inputs,**example_kwargs)
    return model


def apply_pruning(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, version: int=3, pruning_kwargs: dict[str,Any]=None, *args, **kwargs):
    
    from .. import pruning
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
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
        model = pruning.v3.PrunerModule(model, example_inputs=example_inputs, example_kwargs=example_kwargs, **pruning_kwargs)
    return model


def apply_quantization(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, version: int=2, quantization_kwargs: dict[str,Any]=None, *args, **kwargs):
    
    from .. import quantization
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
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