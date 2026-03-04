from ...utils.transformation_utils import wrapped_transformation_fn
from ... import utils
from . import sparsity_func


def init(module, *args, example_inputs, example_kwargs=None, transformation_dict=None, **kwargs):
    """Initializes a module for sparsity training with transformation handling.
    
    This function is a wrapper around sparsity_func.init that applies the wrapped_transformation_fn
    utility to handle model transformations properly. It ensures example inputs and kwargs
    are properly added to the module before initialization.
    
    Args:
        module: The PyTorch module to initialize for sparsity training.
        *args: Additional arguments to pass to the sparsity initialization function.
        example_inputs: Example inputs for model export and tracing.
        example_kwargs (dict, optional): Example keyword arguments for model export. Defaults to None.
        transformation_dict (dict, optional): Dictionary of transformation functions. Defaults to None.
        **kwargs: Additional keyword arguments for sparsity initialization.
        
    Returns:
        The initialized module with sparsity capabilities, after applying any transformations.
        
    Note:
        This wrapper ensures that model transformations (defined in transformation_dict)
        are properly applied before and after the sparsity initialization.
    """
    # Initialize example_kwargs to empty dict if None
    example_kwargs = example_kwargs or {}
    
    # Add example inputs and kwargs to the module for later use
    utils.add_example_args_kwargs(module, example_inputs=example_inputs, example_kwargs=example_kwargs, transformation_dict=transformation_dict)
    
    # Apply the wrapped transformation function to the sparsity initialization
    return wrapped_transformation_fn(sparsity_func.init, module, *args, example_inputs=example_inputs, example_kwargs=example_kwargs, transformation_dict=transformation_dict, **kwargs)


def calculate_sparsity(*args, **kwargs):
    """Calculates the sparsity level of a module with transformation handling.
    
    This function is a wrapper around sparsity_func.calculate_sparsity that applies
    the wrapped_transformation_fn utility to handle model transformations properly.
    
    Args:
        *args: Arguments to pass to the calculate_sparsity function.
        **kwargs: Keyword arguments to pass to the calculate_sparsity function,
            including transformation_dict if needed.
            
    Returns:
        The module with updated sparsity value, after applying any transformations.
    """
    return wrapped_transformation_fn(sparsity_func.calculate_sparsity, *args, **kwargs)


def train(*args, **kwargs):
    """Sets a module to training mode with sparsity handling and transformation handling.
    
    This function is a wrapper around sparsity_func.train that applies the
    wrapped_transformation_fn utility to handle model transformations properly.
    
    Args:
        *args: Arguments to pass to the train function, including the module and mode.
        **kwargs: Keyword arguments to pass to the train function,
            including transformation_dict if needed.
            
    Returns:
        The module with updated training mode and sparsity parametrization,
        after applying any transformations.
    """
    return wrapped_transformation_fn(sparsity_func.train, *args, **kwargs)


def eval(*args, **kwargs):
    """Sets a module to evaluation mode with sparsity handling and transformation handling.
    
    This function is a wrapper around sparsity_func.eval that applies the
    wrapped_transformation_fn utility to handle model transformations properly.
    
    Args:
        *args: Arguments to pass to the eval function, including the module and mode.
        **kwargs: Keyword arguments to pass to the eval function,
            including transformation_dict if needed.
            
    Returns:
        The module in evaluation mode with appropriate sparsity parametrization,
        after applying any transformations.
    """
    return wrapped_transformation_fn(sparsity_func.eval, *args, **kwargs)


def remove_parametrization(*args, **kwargs):
    """Removes parametrization from a module's parameters with transformation handling.
    
    This function is a wrapper around sparsity_func.remove_parametrization that applies
    the wrapped_transformation_fn utility to handle model transformations properly.
    
    Args:
        *args: Arguments to pass to the remove_parametrization function,
            including the module and leave_parameterized flag.
        **kwargs: Keyword arguments to pass to the remove_parametrization function,
            including transformation_dict if needed.
            
    Returns:
        The module with parametrization removed, after applying any transformations.
    """
    return wrapped_transformation_fn(sparsity_func.remove_parametrization, *args, **kwargs)


def insert_parametrization(*args, **kwargs):
    """Inserts sparsity parametrization into a module's parameters with transformation handling.
    
    This function is a wrapper around sparsity_func.insert_parametrization that applies
    the wrapped_transformation_fn utility to handle model transformations properly.
    
    Args:
        *args: Arguments to pass to the insert_parametrization function,
            including the module and binary_mask flag.
        **kwargs: Keyword arguments to pass to the insert_parametrization function,
            including transformation_dict if needed.
            
    Returns:
        The module with parametrization inserted, after applying any transformations.
    """
    return wrapped_transformation_fn(sparsity_func.insert_parametrization, *args, **kwargs)


def get_layer_sparsity_ratio(*args, **kwargs):
    """Calculates the sparsity ratio for each layer with transformation handling.
    
    This function is a wrapper around sparsity_func.get_layer_sparsity_ratio that applies
    the wrapped_transformation_fn utility to handle model transformations properly.
    
    Args:
        *args: Arguments to pass to the get_layer_sparsity_ratio function,
            including the module and sparsity_ratio.
        **kwargs: Keyword arguments to pass to the get_layer_sparsity_ratio function,
            including transformation_dict if needed.
            
    Returns:
        The module with layer sparsity ratios calculated, after applying any transformations.
        
    Note:
        This is currently a placeholder as the underlying function is not fully implemented.
    """
    return wrapped_transformation_fn(sparsity_func.get_layer_sparsity_ratio, *args, **kwargs)

def step(*args, **kwargs):
    return wrapped_transformation_fn(sparsity_func.step, *args, **kwargs)

def finalize(*args, **kwargs):
    return wrapped_transformation_fn(sparsity_func.finalize, *args, **kwargs)