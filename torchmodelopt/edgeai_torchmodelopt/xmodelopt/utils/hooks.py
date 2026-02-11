import torch
from torch import nn
from .transformation_utils import wrapped_transformation_fn

def detach_all_tensors(vals):
    """Recursively detaches all tensors from the computation graph.
    
    This function handles various data structures (lists, tuples, dictionaries)
    and detaches any PyTorch tensors found within them.
    
    Args:
        vals: The input value, which can be a tensor, a collection of tensors,
            or any other type of object.
            
    Returns:
        The input with all tensors detached from the computation graph.
    """
    if isinstance(vals, (list, tuple)):
        return tuple([detach_all_tensors(v) for v in vals])
    elif isinstance(vals, dict):
        return {k: detach_all_tensors(v) for k, v in vals.items()}
    elif isinstance(vals, torch.Tensor):
        return vals.detach()
    else:
        return vals


def record_inputs_pre_hook(self: nn.Module, args, kwargs):
    """Forward pre-hook to record input arguments and keyword arguments.
    
    This hook function is registered with PyTorch modules to record the inputs
    they receive. It stores the inputs in the module's _example_inputs and
    _example_kwargs attributes for later use.
    
    Args:
        self (nn.Module): The module whose inputs are being recorded.
        args: Positional arguments passed to the module's forward function.
        kwargs: Keyword arguments passed to the module's forward function.
    """
    # Append to existing lists or create new ones
    if hasattr(self, "_example_inputs"):
        self._example_inputs.append(detach_all_tensors(args))
        self._example_kwargs.append(detach_all_tensors(kwargs))
    else:
        self._example_inputs = []
        self._example_kwargs = []
        self._example_inputs.append(detach_all_tensors(args))
        self._example_kwargs.append(detach_all_tensors(kwargs))


def register_pre_hook_for_optimization(self: nn.Module):
    """Registers a pre-hook on a module to record input arguments during forward passes.
    
    This function adds a forward pre-hook to the given module that records the inputs
    provided to the module. This is useful for optimization processes that need
    examples of inputs that the module receives.
    
    Args:
        self (nn.Module): The module to register the pre-hook on.
        
    Returns:
        nn.Module: The module with the pre-hook registered.
    """
    if not hasattr(self, '_example_inputs'):
        self.__optimization_pre_hook = self.register_forward_pre_hook(record_inputs_pre_hook, prepend=True, with_kwargs=True)
    return self


def disable_pre_hook_for_optimization(self: nn.Module):
    """Disables and removes a previously registered optimization pre-hook.
    
    Args:
        self (nn.Module): The module with the pre-hook to disable.
        
    Returns:
        nn.Module: The module with the pre-hook removed.
    """
    if hasattr(self, '__optimization_pre_hook'):
        self.__optimization_pre_hook.remove()
        del self.__optimization_pre_hook
    return self


def add_example_args_kwargs(module, example_inputs, example_kwargs=None, transformation_dict=None):
    """Adds example inputs and keyword arguments to a module.
    
    This function registers a pre-hook on the module, runs a forward pass with the
    provided example inputs and keyword arguments to record them, and then disables
    the pre-hook.
    
    Args:
        module (nn.Module): The module to add example inputs to.
        example_inputs: Example inputs to record.
        example_kwargs (dict, optional): Example keyword arguments to record. Defaults to None.
        transformation_dict (dict, optional): A dictionary mapping submodule names to
            transformation functions. If provided, the function will apply the pre-hook
            transformations to specified submodules. Defaults to None.
            
    Note:
        If both example_inputs and example_kwargs are None or empty, this function
        does nothing.
    """
    # Skip if no examples are provided
    if (example_inputs is None or example_inputs in ([], [None], (), (None,))) and (example_kwargs is None or example_kwargs == {}):
        return

    # Prepare the inputs
    example_kwargs = example_kwargs or {}
    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)
    # Register hook, run forward pass, then disable hook
    wrapped_transformation_fn(register_pre_hook_for_optimization, module, transformation_dict=transformation_dict)
    # module.eval()
    module(*example_inputs, **example_kwargs)
    wrapped_transformation_fn(disable_pre_hook_for_optimization, module, transformation_dict=transformation_dict)
    