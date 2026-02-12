from torch import nn
import copy


def apply_transformation_to_submodules(model:nn.Module, transformation_dict: dict, *args, **kwargs):
    """Applies transformation functions to specified submodules of a model.
    
    This function takes a PyTorch model and a dictionary mapping submodule names to wrapper functions.
    It then applies the corresponding wrapper function to each specified submodule.
    
    Args:
        model (nn.Module): The PyTorch model containing the submodules to transform.
        transformation_dict (dict): A dictionary mapping submodule names to wrapper functions.
            The keys are the names of submodules, and the values are callables that will
            transform those submodules.
        *args: Additional positional arguments to pass to the wrapper functions.
        **kwargs: Additional keyword arguments to pass to the wrapper functions.
    
    Returns:
        nn.Module: The model with transformed submodules.
        
    Note:
        For ModuleList submodules, the function handles them specially by applying the 
        transformation to each element within the ModuleList.
    """
    # Create a dictionary of all named modules in the model
    module_dict = dict(model.named_modules())
    
    # Iterate through the transformation dictionary and apply transformations
    for name, wrapper_fn in transformation_dict.items() :
        if name not in module_dict:
            continue
        print(f'INFO: applying transformation to submodule: {name} with wrapper function: {wrapper_fn}')
        module = module_dict[name]
        
        # Get parent module and submodule name
        splits = name.rsplit('.',1)
        if len(splits) == 1:
            splits = '',splits[0]
        parent_module, sub_module_name = splits
        parent_module = model if parent_module == '' else module_dict[parent_module]
        
        # Special handling for ModuleList type modules
        if isinstance(module, nn.ModuleList):  # not well tested, example inputs might not be passed in correct manner
            for i, mod in enumerate(module):
                if isinstance(mod, nn.ModuleList): # taken care of 2 nested modulelist, need to generalise over many #TODO
                    for j, mo in enumerate(mod):
                        if mo is not None:
                            mo = wrapper_fn(mo, *args, **kwargs)
                        mod[j] = mo
                    #
                #
                else:
                    if mod is not None:
                        mod = wrapper_fn(mod, *args, **kwargs)
                module[i] = mod
                # example_inputs = kwargs.get("example_inputs", None)
                # if example_inputs is not None: #or kwargs.get("example_kwargs", None): TODO deal with kwargs passed to modulelist
                #     example_inputs = [mod(*example_inputs)]
                # mod = wrapper_fn(mod, *args, **kwargs)
                # if example_inputs is not None:
                #     kwargs["example_inputs"] = example_inputs
            #
        #
        else:
            # Apply transformation directly for regular modules
            module = wrapper_fn(module, *args, **kwargs)
            
        # Update the original module with the transformed one
        setattr(parent_module, sub_module_name, module)
    return model


class TransformationWrapper:
    """A wrapper class for transformation functions.
    
    This class is used to wrap a function with another function (wrapper).
    It allows for deferred binding of the inner function, which can be set later.
    
    Args:
        wrapper (callable): The outer function that will wrap the inner function.
        fn (callable, optional): The inner function to be wrapped. Can be None and set later.
    
    Raises:
        ValueError: If the inner function (fn) is None when the wrapper is called.
    """
    def __init__(self, wrapper:callable, fn:callable=None):
        self.wrapper = wrapper
        self.fn = fn
        
    def __call__(self, *args, **kwargs):
        """Calls the wrapper function with the inner function and provided arguments.
        
        Args:
            *args: Positional arguments to pass to the wrapper function.
            **kwargs: Keyword arguments to pass to the wrapper function.
            
        Returns:
            The result of the wrapper function called with the inner function and provided arguments.
            
        Raises:
            ValueError: If the inner function (fn) is None.
        """
        # Ensure the function to wrap is provided
        if self.fn is None:
            raise ValueError("the fn function to be wrapped should not be None")
        return self.wrapper(self.fn, *args, **kwargs)


def wrapped_transformation_fn(fn, model, *args, transformation_dict=None, **kwargs):
    """Wraps a transformation function and applies it to a model.
    
    This function takes a transformation function and a model, and either applies
    the function directly to the model or uses it with a transformation dictionary
    to apply transformations to specific submodules.
    
    Args:
        fn (callable): The transformation function to apply.
        model (nn.Module): The PyTorch model to transform.
        *args: Additional positional arguments to pass to the transformation function.
        transformation_dict (dict, optional): A dictionary mapping submodule names to 
            transformation functions. If provided, the function will apply transformations
            to the specified submodules. If None, the function will be applied directly to the model.
        **kwargs: Additional keyword arguments to pass to the transformation function.
        
    Returns:
        nn.Module: The transformed model.
        
    Note:
        If transformation_dict is provided, it is deep-copied before modification.
        For any key with a None value, the value is replaced with fn.
        For any key with a TransformationWrapper value whose fn is None, the wrapper's fn is set to fn.
    """
    if transformation_dict is not None:
        # Process the transformation dictionary and apply to submodules
        transformation_dict = copy.deepcopy(transformation_dict)
        for key, value in transformation_dict.items():
            if value is None:
                transformation_dict[key] = fn
            elif isinstance(value, TransformationWrapper) and value.fn is None:
                value.fn = fn
        return apply_transformation_to_submodules(model, transformation_dict, *args, **kwargs)
    else:
        # Apply transformation directly to the model
        return fn(model, *args, **kwargs)