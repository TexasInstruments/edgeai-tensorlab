import torch
from torch import nn, fx
from typing import Any
import copy
import types

from .model_optimzation_v1 import ModelOptimizationWrapperV1
from .model_optimzation_v2 import ModelOptimizationWrapperV2
from .model_optimzation_v3 import ModelOptimizationWrapperV3

__all__ = ['apply_model_optimization', 'apply_model_optimization_v1', 'apply_model_optimization_v2', 'apply_model_optimization_v3', 
           'prepare_model_for_onnx']


def apply_model_optimization_v1(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None,  model_surgery_kwargs: dict[str,Any]=None, pruning_kwargs: dict[str,Any]=None, quantization_kwargs: dict[str,Any]=None, transformation_dict=None, copy_attrs=None):
    """Applies model optimization using version 1 techniques.
    
    This is a convenience function that calls apply_model_optimization with version 1
    for all optimization techniques.
    
    Args:
        model (nn.Module): The model to optimize.
        example_inputs (list, optional): Example inputs for the model. Defaults to None.
        example_kwargs (dict, optional): Example keyword arguments for the model. Defaults to None.
        model_surgery_kwargs (dict[str,Any], optional): Keyword arguments for model surgery. Defaults to None.
        pruning_kwargs (dict[str,Any], optional): Keyword arguments for pruning. Defaults to None.
        quantization_kwargs (dict[str,Any], optional): Keyword arguments for quantization. Defaults to None.
        transformation_dict (dict, optional): Dictionary mapping module names to transformation functions. Defaults to None.
        copy_attrs (list, optional): List of attribute names to copy. Defaults to None.
        
    Returns:
        nn.Module: The optimized model.
    """
    return apply_model_optimization(model, example_inputs, example_kwargs, model_surgery_version=1, pruning_version=1, quantization_version=1, model_surgery_kwargs=model_surgery_kwargs, pruning_kwargs=pruning_kwargs, quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)


def prepare_model_for_onnx_v1(orig_model, trained_model, example_inputs, example_kwargs, model_surgery_kwargs=None, pruning_kwargs=None, quantization_kwargs=None, transformation_dict=None, copy_attrs=None):
    """Prepares a model for ONNX export using version 1 techniques.
    
    This is a convenience function that calls prepare_model_for_onnx with version 1
    for all optimization techniques.
    
    Args:
        orig_model (nn.Module): The original model architecture.
        trained_model (nn.Module): The trained model with weights to use.
        example_inputs: Example inputs for the model.
        example_kwargs: Example keyword arguments for the model.
        model_surgery_kwargs (dict, optional): Keyword arguments for model surgery. Defaults to None.
        pruning_kwargs (dict, optional): Keyword arguments for pruning. Defaults to None.
        quantization_kwargs (dict, optional): Keyword arguments for quantization. Defaults to None.
        transformation_dict (dict, optional): Dictionary mapping module names to transformation functions. Defaults to None.
        copy_attrs (list, optional): List of attribute names to copy. Defaults to None.
        
    Returns:
        nn.Module: The model prepared for ONNX export.
    """
    return prepare_model_for_onnx(orig_model, trained_model, example_inputs, example_kwargs, model_surgery_version=1, pruning_version=1, quantization_version=1, model_surgery_kwargs=model_surgery_kwargs, pruning_kwargs=pruning_kwargs, quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)


def apply_model_optimization_v2(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None,  model_surgery_kwargs: dict[str,Any]=None, pruning_kwargs: dict[str,Any]=None, quantization_kwargs: dict[str,Any]=None, transformation_dict=None, copy_attrs=None):
    """Applies model optimization using version 2 techniques.
    
    This is a convenience function that calls apply_model_optimization with version 2
    for all optimization techniques.
    
    Args:
        model (nn.Module): The model to optimize.
        example_inputs (list, optional): Example inputs for the model. Defaults to None.
        example_kwargs (dict, optional): Example keyword arguments for the model. Defaults to None.
        model_surgery_kwargs (dict[str,Any], optional): Keyword arguments for model surgery. Defaults to None.
        pruning_kwargs (dict[str,Any], optional): Keyword arguments for pruning. Defaults to None.
        quantization_kwargs (dict[str,Any], optional): Keyword arguments for quantization. Defaults to None.
        transformation_dict (dict, optional): Dictionary mapping module names to transformation functions. Defaults to None.
        copy_attrs (list, optional): List of attribute names to copy. Defaults to None.
        
    Returns:
        nn.Module: The optimized model.
    """
    return apply_model_optimization(model, example_inputs, example_kwargs, model_surgery_version=2, pruning_version=2, quantization_version=2, model_surgery_kwargs=model_surgery_kwargs, pruning_kwargs=pruning_kwargs, quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)


def prepare_model_for_onnx_v2(orig_model, trained_model, example_inputs, example_kwargs, model_surgery_kwargs=None, pruning_kwargs=None, quantization_kwargs=None, transformation_dict=None, copy_attrs=None):
    """Prepares a model for ONNX export using version 2 techniques.
    
    This is a convenience function that calls prepare_model_for_onnx with version 2
    for all optimization techniques.
    
    Args:
        orig_model (nn.Module): The original model architecture.
        trained_model (nn.Module): The trained model with weights to use.
        example_inputs: Example inputs for the model.
        example_kwargs: Example keyword arguments for the model.
        model_surgery_kwargs (dict, optional): Keyword arguments for model surgery. Defaults to None.
        pruning_kwargs (dict, optional): Keyword arguments for pruning. Defaults to None.
        quantization_kwargs (dict, optional): Keyword arguments for quantization. Defaults to None.
        transformation_dict (dict, optional): Dictionary mapping module names to transformation functions. Defaults to None.
        copy_attrs (list, optional): List of attribute names to copy. Defaults to None.
        
    Returns:
        nn.Module: The model prepared for ONNX export.
    """
    return prepare_model_for_onnx(orig_model, trained_model, example_inputs, example_kwargs, model_surgery_version=2, pruning_version=2, quantization_version=2, model_surgery_kwargs=model_surgery_kwargs, pruning_kwargs=pruning_kwargs, quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)


def apply_model_optimization_v3(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None,  model_surgery_kwargs: dict[str,Any]=None, pruning_kwargs: dict[str,Any]=None, quantization_kwargs: dict[str,Any]=None, transformation_dict=None, copy_attrs=None):
    """Applies model optimization using version 3 techniques.
    
    This is a convenience function that calls apply_model_optimization with version 3
    for all optimization techniques.
    
    Args:
        model (nn.Module): The model to optimize.
        example_inputs (list, optional): Example inputs for the model. Defaults to None.
        example_kwargs (dict, optional): Example keyword arguments for the model. Defaults to None.
        model_surgery_kwargs (dict[str,Any], optional): Keyword arguments for model surgery. Defaults to None.
        pruning_kwargs (dict[str,Any], optional): Keyword arguments for pruning. Defaults to None.
        quantization_kwargs (dict[str,Any], optional): Keyword arguments for quantization. Defaults to None.
        transformation_dict (dict, optional): Dictionary mapping module names to transformation functions. Defaults to None.
        copy_attrs (list, optional): List of attribute names to copy. Defaults to None.
        
    Returns:
        nn.Module: The optimized model.
    """
    return apply_model_optimization(model, example_inputs, example_kwargs, model_surgery_version=3, pruning_version=3, quantization_version=3, model_surgery_kwargs=model_surgery_kwargs, pruning_kwargs=pruning_kwargs, quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)


def prepare_model_for_onnx_v3(orig_model, trained_model, example_inputs, example_kwargs, model_surgery_kwargs=None, pruning_kwargs=None, quantization_kwargs=None, transformation_dict=None, copy_attrs=None):
    """Prepares a model for ONNX export using version 3 techniques.
    
    This is a convenience function that calls prepare_model_for_onnx with version 3
    for all optimization techniques.
    
    Args:
        orig_model (nn.Module): The original model architecture.
        trained_model (nn.Module): The trained model with weights to use.
        example_inputs: Example inputs for the model.
        example_kwargs: Example keyword arguments for the model.
        model_surgery_kwargs (dict, optional): Keyword arguments for model surgery. Defaults to None.
        pruning_kwargs (dict, optional): Keyword arguments for pruning. Defaults to None.
        quantization_kwargs (dict, optional): Keyword arguments for quantization. Defaults to None.
        transformation_dict (dict, optional): Dictionary mapping module names to transformation functions. Defaults to None.
        copy_attrs (list, optional): List of attribute names to copy. Defaults to None.
        
    Returns:
        nn.Module: The model prepared for ONNX export.
    """
    return prepare_model_for_onnx(orig_model, trained_model, example_inputs, example_kwargs, model_surgery_version=3, pruning_version=3, quantization_version=3, model_surgery_kwargs=model_surgery_kwargs, pruning_kwargs=pruning_kwargs, quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)


def apply_model_optimization(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, model_surgery_version=3, pruning_version=3, quantization_version=3, model_surgery_kwargs: dict[str,Any]=None, pruning_kwargs: dict[str,Any]=None, quantization_kwargs: dict[str,Any]=None, transformation_dict=None, copy_attrs=None):
    """A wrapper function to apply surgery, pruning, and quantization to a model.
    
    This function coordinates the application of different optimization techniques
    to a model based on the provided configuration.
    
    Args:
        model (nn.Module): The model to optimize.
        example_inputs (list, optional): Example inputs for the model. Defaults to None.
        example_kwargs (dict, optional): Example keyword arguments for the model. Defaults to None.
        model_surgery_version (int, optional): Version of model surgery to apply. Defaults to 3.
            - 0: No surgery
            - 1: Version 1 of model surgery
            - 2: Version 2 of model surgery
            - 3: Version 3 of model surgery
        pruning_version (int, optional): Version of pruning to apply. Defaults to 3.
            - 0: No pruning
            - 1: Version 1 of pruning
            - 2: Version 2 of pruning
            - 3: Version 3 of pruning
        quantization_version (int, optional): Version of quantization to apply. Defaults to 3.
            - 0: No quantization
            - 1: Version 1 of quantization
            - 2: Version 2 of quantization
            - 3: Version 3 of quantization
        model_surgery_kwargs (dict[str,Any], optional): Keyword arguments for model surgery. Defaults to None.
        pruning_kwargs (dict[str,Any], optional): Keyword arguments for pruning. Defaults to None.
        quantization_kwargs (dict[str,Any], optional): Keyword arguments for quantization. Defaults to None.
        transformation_dict (dict, optional): Dictionary mapping module names to transformation functions. Defaults to None.
        copy_attrs (list, optional): List of attribute names to copy. Defaults to None.
        
    Returns:
        nn.Module: The optimized model.
    """
    # Prepare default values and handle empty inputs
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
    model_surgery_version = model_surgery_version or (0 if model_surgery_kwargs is None else 2)
    pruning_version = pruning_version or (0 if pruning_kwargs is None else 2)
    quantization_version = quantization_version or (0 if quantization_kwargs is None else 2)
    copy_attrs = copy_attrs or []
    
    # Return original model if no optimization is requested
    if not(pruning_version or model_surgery_version or quantization_version):
        return model

    # Run the model once with example inputs
    # model.eval()
    if isinstance(example_inputs, dict):
        model(example_inputs, **example_kwargs)
    else:
        model(*example_inputs, **example_kwargs)

    # Deep copy the kwargs to avoid modifying the originals
    model_surgery_kwargs = copy.deepcopy(model_surgery_kwargs)
    pruning_kwargs = copy.deepcopy(pruning_kwargs)
    quantization_kwargs = copy.deepcopy(quantization_kwargs)
    main_model_optimization_version = None
    main_kwargs = dict(transformation_dict=transformation_dict, copy_attrs=copy_attrs)

    # Determine the main optimization version and validate compatibility
    if model_surgery_version != 0 and model_surgery_kwargs is not None:
        assert model_surgery_version in (1, 2, 3)
        main_model_optimization_version = model_surgery_version
        main_kwargs.update(model_surgery_kwargs=model_surgery_kwargs)

    if pruning_version != 0 and pruning_kwargs is not None:
        if main_model_optimization_version is None:
            assert pruning_version in (1, 2, 3)
            main_model_optimization_version = pruning_version
        else:
            assert pruning_version == main_model_optimization_version, f'''
            Different versions of model optimization techniques aren't supported together!
            Previously got Surgery v{main_model_optimization_version} but got Pruning v{pruning_version}.
            '''
        main_kwargs.update(pruning_kwargs=pruning_kwargs)

    if quantization_version != 0 and quantization_kwargs is not None:
        if main_model_optimization_version is None:
            assert quantization_version in (1, 2, 3)
            main_model_optimization_version = quantization_version
        else:
            assert quantization_version == main_model_optimization_version, f'''
            Different versions of model optimization techniques aren't supported together!
            Previously got Surgery and Pruning v{main_model_optimization_version} but got Quantization v{quantization_version}.
            '''
        main_kwargs.update(quantization_kwargs=quantization_kwargs)
    
    # Handle DataParallel and DistributedDataParallel models
    is_wrapped = isinstance(model,(torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    if is_wrapped:
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Apply the appropriate optimization wrapper based on version
    if main_model_optimization_version == 1:
        pass
    elif main_model_optimization_version == 2:
        model_without_ddp = ModelOptimizationWrapperV2(model_without_ddp, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)
    elif main_model_optimization_version == 3:
        model_without_ddp = ModelOptimizationWrapperV3(model_without_ddp, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)
    
    # Update the original model reference
    if is_wrapped:
        model.module = model_without_ddp
    else:
        model = model_without_ddp
    
    # TODO: check this later
    # if main_model_optimization_version == 1:
    #     pass
    # elif main_model_optimization_version == 2:
    #     if isinstance(model, nn.parallel.DistributedDataParallel):
    #         device = next(iter(model.module.named_parameters()))[1].device
    #         for i, example_input in enumerate(example_inputs):
    #             example_inputs[i] = example_input.to(device=device)
    #         for key, val in example_kwargs.items():
    #             if isinstance(val, list):
    #                 for i, v in enumerate(val):
    #                     val[i] = v.to(device=device)
    #             else:
    #                 val = val.to(device=device)
    #             example_kwargs[key] = val
    #         model.module = model_optimzation_v2.ModelOptimizationWrapperV2(model.module, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)
    #     else:
    #         model = model_optimzation_v2.ModelOptimizationWrapperV2(model, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)
    # elif main_model_optimization_version == 3:
    #     if isinstance(model, nn.parallel.DistributedDataParallel):
    #         device = next(iter(model.module.named_parameters()))[1].device
    #         for i, example_input in enumerate(example_inputs):
    #             example_inputs[i] = example_input.to(device=device)
    #         for key, val in example_kwargs.items():
    #             if isinstance(val, list):
    #                 for i, v in enumerate(val):
    #                     val[i] = v.to(device=device)
    #             else:
    #                 val = val.to(device=device)
    #             example_kwargs[key] = val
    #         model.module = model_optimzation_v3.ModelOptimizationWrapperV3(model.module, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)
    #     else:
    #         model = model_optimzation_v3.ModelOptimizationWrapperV3(model, example_inputs=example_inputs, example_kwargs=example_kwargs, **main_kwargs)

    return model


def prepare_model_for_onnx(orig_model: nn.Module, trained_model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, model_surgery_version=3, pruning_version=3, quantization_version=3, model_surgery_kwargs: dict[str,Any]=None, pruning_kwargs: dict[str,Any]=None, quantization_kwargs: dict[str,Any]=None, transformation_dict=None, copy_attrs=None):
    """Prepares a model for ONNX export by applying optimization techniques.
    
    This function takes an original model architecture and a trained model with weights,
    applies optimization techniques to the original model, and then loads the weights
    from the trained model into the optimized model.
    
    Args:
        orig_model (nn.Module): The original model architecture.
        trained_model (nn.Module): The trained model with weights to use.
        example_inputs (list, optional): Example inputs for the model. Defaults to None.
        example_kwargs (dict, optional): Example keyword arguments for the model. Defaults to None.
        model_surgery_version (int, optional): Version of model surgery to apply. Defaults to 3.
        pruning_version (int, optional): Version of pruning to apply. Defaults to 3.
        quantization_version (int, optional): Version of quantization to apply. Defaults to 3.
        model_surgery_kwargs (dict[str,Any], optional): Keyword arguments for model surgery. Defaults to None.
        pruning_kwargs (dict[str,Any], optional): Keyword arguments for pruning. Defaults to None.
        quantization_kwargs (dict[str,Any], optional): Keyword arguments for quantization. Defaults to None.
        transformation_dict (dict, optional): Dictionary mapping module names to transformation functions. Defaults to None.
        copy_attrs (list, optional): List of attribute names to copy. Defaults to None.
        
    Returns:
        nn.Module: The model prepared for ONNX export.
    """
    # Handle DataParallel and DistributedDataParallel models
    if isinstance(trained_model,(torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        trained_model = trained_model.module
    
    if isinstance(orig_model,(torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        orig_model = orig_model.module
    
    # Prepare default values
    example_inputs = example_inputs or []
    example_kwargs = example_kwargs or {}
    
    # Move everything to CPU for ONNX export
    orig_model = orig_model.to('cpu')
    example_inputs = [input_tensor.to('cpu') if isinstance(input_tensor, torch.Tensor) else input_tensor for input_tensor in example_inputs]
    example_kwargs = {key: value.to('cpu') if isinstance(value, torch.Tensor) else value for key, value in example_kwargs.items()}
    
    # Apply optimization techniques
    final_model = apply_model_optimization(orig_model, example_inputs, example_kwargs, model_surgery_version=model_surgery_version, pruning_version=pruning_version, 
                                           quantization_version=quantization_version, model_surgery_kwargs=model_surgery_kwargs, pruning_kwargs=pruning_kwargs, 
                                           quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)
    
    # Load weights from the trained model
    state_dict = trained_model.state_dict()
    final_model.load_state_dict(state_dict)
    
    return final_model


def apply_model_surgery(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, version: int=3, model_surgery_kwargs: dict[str,Any]=None, *args, **kwargs):
    """Applies model surgery to a PyTorch model.
    
    Model surgery involves replacing certain modules or operations in a model
    with more efficient or specialized versions.
    
    Args:
        model (nn.Module): The model to apply surgery to.
        example_inputs (list, optional): Example inputs for the model. Defaults to None.
        example_kwargs (dict, optional): Example keyword arguments for the model. Defaults to None.
        version (int, optional): Version of model surgery to apply. Defaults to 3.
        model_surgery_kwargs (dict[str,Any], optional): Keyword arguments for model surgery. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        nn.Module: The model after surgery.
        
    Raises:
        AssertionError: If 'replacement_dict' is not in model_surgery_kwargs.
    """
    # Import surgery module
    from ..experimental import surgery
    
    # Prepare inputs and run the model once
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
    model(*example_inputs, **example_kwargs)
    
    # Deep copy the kwargs to avoid modifying the originals
    model_surgery_kwargs = copy.deepcopy(model_surgery_kwargs)
    
    # Validate inputs
    if version in (1, 2, 3):
        assert 'replacement_dict' in model_surgery_kwargs, "A 'replacement_dict' must be present in surgery kwargs."
    
    # Apply surgery based on version
    if version == 1:
        replacement_dict = model_surgery_kwargs.pop('replacement_dict')
        model = surgery.v1.convert_to_lite_model(model, replacement_dict, **model_surgery_kwargs)
    elif version == 2:
        replacement_dict = model_surgery_kwargs.pop('replacement_dict')
        model = surgery.v2.convert_to_lite_fx(model, replacement_dict, example_inputs, example_kwargs, **model_surgery_kwargs)
    elif version == 3:
        replacement_dict = model_surgery_kwargs.pop('replacement_dict')
        model = surgery.v3.convert_to_lite_pt2e(model, replacement_dict, example_inputs, example_kwargs, **model_surgery_kwargs)
    
    return model


def apply_pruning(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, version: int=3, pruning_kwargs: dict[str,Any]=None, *args, **kwargs):
    """Applies pruning to a PyTorch model.
    
    Pruning involves removing or zeroing out less important weights in a model
    to reduce its size and computational requirements.
    
    Args:
        model (nn.Module): The model to apply pruning to.
        example_inputs (list, optional): Example inputs for the model. Defaults to None.
        example_kwargs (dict, optional): Example keyword arguments for the model. Defaults to None.
        version (int, optional): Version of pruning to apply. Defaults to 3.
        pruning_kwargs (dict[str,Any], optional): Keyword arguments for pruning. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        nn.Module: The pruned model.
        
    Raises:
        AssertionError: If version is 1 or if required kwargs are missing for versions 2 and 3.
    """
    # Import pruning module
    from ..experimental import pruning
    
    # Prepare inputs and run the model once
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
    model(*example_inputs, **example_kwargs)
    
    # Deep copy the kwargs to avoid modifying the originals
    pruning_kwargs = copy.deepcopy(pruning_kwargs)
    
    # Validate inputs based on version
    if version == 1:
        assert False, "Pruning is currently not supported in the legacy modules based method"
    
    if version in (2, 3):
        required_kwargs = ['pruning_ratio', 'total_epochs', 'pruning_init_train_ep', 'pruning_class', 'pruning_type', 'pruning_global', 'pruning_m', 'p']
        for kwarg in required_kwargs:
            assert kwarg in pruning_kwargs, f"A '{kwarg}' must be present in pruning kwargs. Pruning Kwargs must contain the followings\n\t" + ', '.join(required_kwargs)
    
    # Apply pruning based on version
    if version == 2:
        model = pruning.v2.PrunerModule(model, **pruning_kwargs)
    elif version == 3:
        model = pruning.v3.PrunerModule(model, example_inputs=example_inputs, example_kwargs=example_kwargs, **pruning_kwargs)
    
    return model


def apply_quantization(model: nn.Module, example_inputs: list=None, example_kwargs: dict=None, version: int=2, quantization_kwargs: dict[str,Any]=None, *args, **kwargs):
    """Applies quantization to a PyTorch model.
    
    Quantization reduces the precision of the weights and activations in a model,
    typically from float32 to int8, to reduce memory usage and improve inference speed.
    
    Args:
        model (nn.Module): The model to apply quantization to.
        example_inputs (list, optional): Example inputs for the model. Defaults to None.
        example_kwargs (dict, optional): Example keyword arguments for the model. Defaults to None.
        version (int, optional): Version of quantization to apply. Defaults to 2.
        quantization_kwargs (dict[str,Any], optional): Keyword arguments for quantization. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        nn.Module: The quantized model.
    """
    # Import quantization module
    from .. import quantization
    
    # Prepare inputs and run the model once
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
    model(*example_inputs, **example_kwargs)
    
    # Deep copy the kwargs to avoid modifying the originals
    quantization_kwargs = copy.deepcopy(quantization_kwargs)
    
    # Common setup for all versions
    if version in (1, 2, 3):
        # For common kwargs
        pass
    
    # Apply quantization based on version
    if version == 1:
        model = quantization.v1.QuantTrainModule(model, *example_inputs, **quantization_kwargs)
    elif version == 2:
        quantization_method = quantization_kwargs.pop('quantization_method', None)
        if quantization_method == 'QAT':
            model = quantization.v2.QATFxModule(model, **quantization_kwargs)
        elif quantization_method in ('PTQ', 'PTC'):
            model = quantization.v2.PTCFxModule(model, **quantization_kwargs)
    elif version == 3:
        # assert False, "Quantization is currently not supported in the PT2E based method"
        quantization_method = quantization_kwargs.pop('quantization_method', None)
        if quantization_method == 'QAT':
            model = quantization.v3.QATPT2EModule(model, example_inputs=example_inputs, **quantization_kwargs)
        elif quantization_method in ('PTQ', 'PTC'):
            model = quantization.v3.PTQPT2EModule(model, example_inputs=example_inputs, **quantization_kwargs)
    
    return model