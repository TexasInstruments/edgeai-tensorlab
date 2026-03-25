import  types
import torch
from torch.fx import GraphModule
from .hooks import register_pre_hook_for_optimization, disable_pre_hook_for_optimization
from . import TransformationWrapper  


def wrap_fn_for_replace_PE(fn, module, *args, **kwargs):
    """Wraps a function to replace positional embeddings with constants.
    
    This wrapper function replaces a module's forward function to return a constant
    value instead of computing the positional embeddings each time. This is useful
    for optimizations where certain parts of the network can be pre-computed.
    
    Args:
        fn (callable): The function to wrap.
        module (nn.Module): The module whose positional embeddings will be replaced.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        nn.Module: The module with its forward function modified to return a constant.
        
    Note:
        If the module already has a 'constant_PE' attribute, it is returned unchanged.
        Otherwise, the module's forward function is modified to return a pre-computed
        value stored in the 'constant_PE' buffer.
    """
    # Return unchanged if already has constant PE
    if hasattr(module, 'constant_PE'):
        return module
    else:
        # Handle special hook functions
        if fn in (register_pre_hook_for_optimization, disable_pre_hook_for_optimization):
            module = fn(module, *args, **kwargs)
        else:
            # Pre-compute and store the constant positional embedding
            assert hasattr(module, '_example_inputs') 
            val = module(*module._example_inputs, **module._example_kwargs)
            module.register_buffer('constant_PE', val)
            def new_forward(self, *args, **kwargs) -> tuple[list]:
                return self.constant_PE
            module.forward = types.MethodType(new_forward, module)
    return module


def wrap_fn_for_bbox_head(fn, module, *args, **kwargs):
    """Wraps a function for modifying the bbox_head module's behavior.
    
    This wrapper function creates or updates a 'new_bbox_head' submodule within
    the given module, and modifies the module's forward function to use this
    new submodule. This is useful for applying optimizations to specific parts
    of a model's architecture.
    
    Args:
        fn (callable): The function to apply to the bbox_head.
        module (nn.Module): The module containing the bbox_head to modify.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        nn.Module: The module with its forward function modified to use the new bbox_head.
    """
    # Update existing new_bbox_head if it exists
    if hasattr(module, 'new_bbox_head'):
        module.new_bbox_head = fn(module.new_bbox_head, *args, **kwargs)
    else:
        # Create new_bbox_head by applying the function to the module
        new_bbox_head = fn(module, *args, **kwargs)
        if new_bbox_head is not module:    
            # Add the new bbox_head as a submodule and modify forward function
            module.add_module('new_bbox_head', new_bbox_head)
            def new_forward(self, x: tuple[torch.Tensor]) -> tuple[list]:
                return self.new_bbox_head(x)
            module.forward = types.MethodType(new_forward, module)
            
            # Clean up unnecessary parameters if using a GraphModule
            if isinstance(new_bbox_head, GraphModule):
                params = dict(module.named_parameters())
                for key in params:
                    if key.startswith("new_bbox_head."):
                        continue
                    split = key.rsplit('.',1)
                    if len(split) == 1:
                        param_name = split[0]
                        delattr(module, param_name)
                    else:
                        parent_module, param_name = split
                        main_module = parent_module.split('.',1)[0]
                        if hasattr(module, main_module):
                            delattr(module, main_module)
        else:
            module = new_bbox_head
    return module 


def DETR_transformation():
    """Creates a transformation dictionary for the DETR architecture.
    
    This function creates a dictionary mapping specific module names in the
    DETR architecture to transformation functions. This is used to selectively
    apply optimizations to different parts of the model.
    
    Returns:
        dict: A dictionary mapping module names to transformation functions.
    """
    # Create transformation dictionary for DETR model
    transformation_dict = dict(class_labels_classifier=None, bbox_predictor=None)
    transformation_dict["model.backbone.position_embedding"] = TransformationWrapper(wrap_fn_for_replace_PE)
    transformation_dict["model.backbone.conv_encoder"] = None
    transformation_dict["model.input_projection"] = None
    transformation_dict["model.encoder"] = None
    transformation_dict["model.decoder"] = None
    return transformation_dict


# Dictionary mapping model names to their transformation dictionaries
transformation_mapping = {
    "transformers_DETR" : DETR_transformation()
}


def get_transformation_for_model(model_name):
    """Gets the transformation dictionary for a specific model type.
    
    Args:
        model_name (str): The name of the model to get transformations for.
        
    Returns:
        dict: The transformation dictionary for the specified model.
        
    Raises:
        Exception: If the model name is not found in the transformation mapping.
    """
    if model_name not in transformation_mapping.keys():
        raise Exception("Transformation mapping for {} is not defined.".format(model_name))
    return transformation_mapping[model_name]