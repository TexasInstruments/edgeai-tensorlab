
import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.utils.parametrize as parametrize
import types

from .... import xnn
from .utils import get_sparsity_nodes, register_n2m_filters, get_all_weights, register_n2m_weight_funcs, get_weights
from .parametrization import SPARSITY_CLASS_DICT
from ... import utils
from ...utils.helper_functions import get_parent_name, nested_getattr

def init(module, *args, example_inputs:list=None, example_kwargs:dict=None, sparsity_ratio=None, p=2.0, sparsity_global=False, copy_args=None,
            sparsity_type='n2m', sparsity_m=None, sparsity_start_epoch=0, sparsity_end_epoch=1,
            add_methods=True, copy_attrs=None, filter_func_register=None, weight_func_register=None, **kwargs):
    """Initializes a module for sparsity training.
    
    This function takes a PyTorch module and sets it up for sparsity training by adding
    necessary parameters, registering filter and weight functions, and adding methods
    for sparsity operations during training. It converts the module to a GraphModule if needed,
    and adds sparsity-specific methods and attributes to enable gradual sparsification
    during training.
    
    Args:
        module: The PyTorch module to initialize for sparsity training.
        *args: Additional arguments to pass to the sparsity functions.
        example_inputs (list, optional): Example inputs for model export. Defaults to None.
        example_kwargs (dict, optional): Example keyword arguments for model export. Defaults to None.
        sparsity_ratio (float, optional): Target sparsity ratio (e.g., 0.5 for 50% sparsity). Defaults to None.
        p (float, optional): Power parameter for sparsity calculation (controls sparsity schedule). Defaults to 2.0.
        sparsity_global (bool, optional): Whether to apply global sparsity across all layers. Defaults to False.
        copy_args (list, optional): List of arguments to copy from the original module. Defaults to None.
        sparsity_type (str, optional): Type of sparsity pattern ('n2m' or 'unstructured'). Defaults to 'n2m'.
        sparsity_m (int, optional): The m value in n:m sparsity pattern (e.g., 4 for 2:4 sparsity). Defaults to None.
        add_methods (bool, optional): Whether to add sparsity methods to the module. Defaults to True.
        copy_attrs (list, optional): List of attributes to copy from the original module. Defaults to None.
        filter_func_register (function, optional): Custom function to register sparsity filters. Defaults to None.
        weight_func_register (function, optional): Custom function to register weight functions. Defaults to None.
        sparsity_start_epoch: Epoch where incremental sparsification starts
        sparsity_end_epoch: Epoch where incremental sparsification end, reaching target sparsity
        **kwargs: Additional keyword arguments for sparsity initialization.
        
    Returns:
        fx.GraphModule: The initialized module with sparsity capabilities.
        
    Raises:
        RuntimeError: If required parameters (sparsity_ratio, total_epochs, or sparsity_m for n:m sparsity) are missing.
    """
    # Initialize default values for optional parameters
    copy_attrs = copy_attrs or []
    copy_args = copy_args or []
    example_inputs =[] if example_inputs is None else example_inputs
    example_kwargs = example_kwargs or {}
    mode = kwargs.get('mode', 'topk')  # Default sparsity mode is 'topk'
    
    # Ensure module has example inputs/kwargs for tracing/export
    if not (hasattr(module, '_example_inputs') and hasattr(module, '_example_kwargs')):
        # This should not get called unless this function is called separately, when called from wrapper model should have example inputs and kwargs
        # Add example inputs/kwargs to the module if not already present
        utils.add_example_args_kwargs(module, example_inputs=example_inputs, example_kwargs=example_kwargs)
    example_inputs = module._example_inputs.pop(0)
    example_kwargs = module._example_kwargs.pop(0)
    
    # Handle different module types: use as is if already a GraphModule, otherwise export it
    if isinstance(module,fx.GraphModule):
        #TODO Differnetiate between fx and pt2e graph modules
        # Assuming Default pt2e here
        gm_module = module
    else:
        # Convert PyTorch module to GraphModule using torch.export
        example_inputs = tuple(example_inputs)
        check_guards = kwargs.get('check_guards', True)
        gm_module = torch.export.export(module, example_inputs, kwargs=example_kwargs).module(check_guards=check_guards)
        # Add train() and eval() methods to the exported model
        from ...utils.helper_functions import allow_exported_model_train_eval
        allow_exported_model_train_eval(gm_module)
    
    # Initialize sparsity parameters attribute dictionary
    gm_module.__sparse_params__ = xnn.utils.AttrDict()
    
    # Configure sparsity training parameters

    gm_module.__sparse_params__.epoch_count = 0  # Track current training epoch
    gm_module.__sparse_params__.sparsity_ratio = sparsity_ratio  # Target sparsity level
    # gm_module.__sparse_params__.total_epochs = total_epochs  # Total training epochs
    gm_module.__sparse_params__.sparsity = 0  # Current sparsity level (will be updated)
    # gm_module.__sparse_params__.init_train_ep = sparsity_init_train_ep  # Initial training epochs
    gm_module.__sparse_params__.p = p  # Power parameter for mask calculation
    

    # Collect all the args used to initialize the parametrization class here. This would make REQUIRED_PARAMS unnecessary. 
    # It shouldn't matter if additional info is passed, as we use kwargs
    gm_module.__sparse_params__.parametrization_kwargs = {
        'current_epoch': 0,
        'sparsity_start_epoch': sparsity_start_epoch,
        'sparsity_end_epoch': sparsity_end_epoch, 
        'p': p,
    }
    # Allow kwargs to pass through to parametrization object
    gm_module.__sparse_params__.parametrization_kwargs.update(kwargs)
    
    # Validate required parameters
    if sparsity_ratio is None or sparsity_ratio==0:
        raise RuntimeError(f"sparsity ratio of {sparsity_ratio} is not supported.")
    # if sparsity_ratio==0:
    #     raise RuntimeError("sparsity ratio of 0 is not supported , try turning off sparsity and trying again")
    # if not(sparsity_ratio and total_epochs):
    #     raise RuntimeError("sparsity ratio and total epochs are necessary to be provided")
    # elif not(sparsity_ratio):
    #     raise RuntimeError("sparsity ratio should be provided")
    # elif not(total_epochs):
    #     raise RuntimeError("total epochs should be provided")
    
    # Set sparsity class from the dictionary based on sparsity type
    gm_module.__sparse_params__.sparsity_class = SPARSITY_CLASS_DICT[sparsity_type]

    # Initialize sparsity type flags
    gm_module.__sparse_params__.n2m_sparsity = False
    gm_module.__sparse_params__.unstructured = False
    gm_module.__sparse_params__.parametrized_params = set()  # Track which parameters are parametrized
    
    # Set specific sparsity type flag based on sparsity_type
    if sparsity_type=='n2m':
        gm_module.__sparse_params__.n2m_sparsity = True  # n:m structured sparsity
    elif sparsity_type=='unstructured':
        gm_module.__sparse_params__.unstructured = True  # Unstructured sparsity
        
    gm_module.__sparse_params__.global_sparsity = sparsity_global
    
    # Initialize filter arguments for identifying nodes to sparsify
    module.filter_args = []        
    
    # Configure n:m sparsity parameters if applicable
    if gm_module.__sparse_params__.n2m_sparsity:
        # n:m sparsity requires m value to be specified
        if sparsity_m is None:
            raise RuntimeError("The value of m should be provided in case of n:m sparsity")
        else:
            # Set m value and calculate n (number of non-zero elements in each block of size m)
            sparsity_n = round(sparsity_ratio*sparsity_m)
            # gm_module.__sparse_params__.m = sparsity_m
            # gm_module.__sparse_params__.n = sparsity_n = round(sparsity_ratio*sparsity_m)
            # gm_module.__sparse_params__.mode = mode  # Sparsity mode (e.g., 'topk')\
            # NOTE: parametrization_kwargs has already been partly set above
            gm_module.__sparse_params__.parametrization_kwargs.update({
                'm': sparsity_m,
                'n': sparsity_n,
                'mode': mode
            })
            
        # Add n and m values to filter arguments
        module.filter_args += [sparsity_n, sparsity_m]
        
        # Register filter and weight functions for n:m sparsity if not provided
        filter_func_register = filter_func_register or register_n2m_filters
        weight_func_register = weight_func_register or register_n2m_weight_funcs
        
        # Register n:m specific filters and weight functions with calculated n and m
        filter_func_register(sparsity_n, sparsity_m)
        weight_func_register(sparsity_n, sparsity_m)
    else:
        # For non-n:m sparsity, set m to None
        # gm_module.__sparse_params__.m = None
        gm_module.__sparse_params__.parametrization_kwargs.update({
                'm': None,
            })

    module.filter_args += [sparsity_type]
    gm_module.__sparse_params__.sparsity_nodes = get_sparsity_nodes(gm_module, *module.filter_args) # dict[tuple, list[list[Node]]]
    gm_module.__sparse_params__.weights = get_all_weights(gm_module, gm_module.__sparse_params__.sparsity_nodes, )
    
    if gm_module.__sparse_params__.n2m_sparsity and gm_module.__sparse_params__.global_sparsity:
        print("Cannot do both global sparsity along with n2m sparsity, it doesn't make sense! \n")
        raise NotImplementedError
    
    for copy_arg in copy_args:
        if hasattr(module, copy_arg):
            setattr(gm_module, copy_arg, getattr(module, copy_arg))
    
    insert_all_parametrizations(gm_module)

    # if gm_module.__sparse_params__.global_sparsity:
    #     gm_module.__sparse_params__.get_layer_sparsity_ratio(sparsity_ratio)
    #
    if add_methods:
        # Add sparsity-aware methods to the module
        
        # Add parametrization management method
        gm_module._insert_and_remove_parametrization_during_training = types.MethodType(insert_and_remove_parametrization_during_training, gm_module) 
        
        # Backup original train method and override with sparsity-aware versions
        gm_module.__sparsity_train_backup__ = types.MethodType(module.train.__func__, gm_module)
        gm_module.train = types.MethodType(train, gm_module)
        gm_module.eval = types.MethodType(train, gm_module)
        
        # Add core sparsity methods
        # gm_module.insert_all_parametrizations = types.MethodType(insert_all_parametrizations_2, gm_module)
        # gm_module.remove_parametrization = types.MethodType(remove_parametrization, gm_module)
        # gm_module.calculate_sparsity = types.MethodType(calculate_sparsity, gm_module)
    return gm_module

# TODO implement for sparsity from sparsity with pt2e
def get_layer_sparsity_ratio(module, sparsity_ratio=0.6):
    """Calculates the sparsity ratio for each layer in the module.
    
    This function is intended to determine appropriate sparsity ratios for individual 
    layers within the network, rather than applying a uniform ratio across all layers.
    This can help maintain performance while achieving the overall target sparsity.
    
    Args:
        module: The module to analyze for layer-wise sparsity assignment.
        sparsity_ratio (float, optional): The target global sparsity ratio. Defaults to 0.6.
    
    Returns:
        None: This is currently a placeholder function for future implementation.
    
    Note:
        This is a placeholder function for future implementation of layer-specific
        sparsity allocation strategies. When implemented, it will likely analyze layer
        importance and sensitivity to determine optimal per-layer sparsity ratios.
        
    TODO: Implement layer-wise sparsity ratio calculation based on layer sensitivity
    and importance metrics.
    """
    pass

def train(module, mode: bool = True): 
    """Sets the module to training mode with sparsity handling.
    
    This function overrides the default train method to handle sparsity parametrization
    during training. It first calls the original train method, then inserts or removes
    parametrization as needed for sparsity training.
    
    Args:
        module: The module to set to training mode.
        mode (bool, optional): Whether to set the module to training mode. Defaults to True.
        
    Returns:
        module: The module with updated training mode and sparsity parametrization.
        
    Note:
        This function is designed to work with modules that have been initialized for
        sparsity training using the init() function. It relies on the presence of a
        backup training method (__sparsity_train_backup__) and parametrization handling
        method (_insert_and_remove_parametrization_during_training).
    """
    # TODO: call step and finalize functions from here
    #       Every time it flips from eval to train, call step
    #       At 'last' eval, call finalize
    # Call the original train method if it exists, then handle sparsity parametrization
    if hasattr(module, "__sparsity_train_backup__"):
        # First execute the original train method
        module.__sparsity_train_backup__(mode=mode)
        
    # Then handle parametrization insertion/removal for sparsity
    module = module._insert_and_remove_parametrization_during_training(mode)
    return module

def eval(self, mode: bool = False):
    """Sets the module to evaluation mode with sparsity handling.
    
    This function sets the module to evaluation mode by calling the train function with mode=False.
    It ensures proper handling of sparsity parametrization during evaluation, including
    finalizing sparsity mask application when training is complete.
    
    Args:
        self: The module to set to evaluation mode.
        mode (bool, optional): Whether to set the module to training mode. Defaults to False.
        
    Returns:
        module: The module in evaluation mode with appropriate sparsity parametrization.
        
    Note:
        This is a convenience wrapper around the train() function. When training is
        complete (epoch_count == total_epochs), calling this function will finalize
        the sparsity patterns by applying hard binary masks to the weights.
    """
    # Simply call train with mode=False to handle evaluation mode
    return train(self, mode)

def step(module):
    module.__sparse_params__.epoch_count += 1
    update_all_parametrizations(module)
    # print(f'steppin {module.__sparse_params__.epoch_count}')
    return module 

def finalize(module):
    update_all_parametrizations(module, binary_mask=True) # note: this does increment an additional epoch, but should be fine
    sample_parametrization = module.__sparse_params__.parametrization_list[0][3] # (parent_module, parent_module_name, param_name, parametrization)
    print(f'Finalizing model by permanently sparsifying params.\n Current epoch: {sample_parametrization.current_epoch}')
    assert sample_parametrization.current_epoch >= sample_parametrization.sparsity_end_epoch, f'Prematurely finalized incremental sparsity'
    remove_all_parametrizations(module, leave_parametrized=True)  
    return module

def insert_and_remove_parametrization_during_training(module, mode: bool = True):
    """Manages sparsity parametrization during training and evaluation.
    
    This function handles the insertion and removal of parametrization for sparsity
    based on the current training mode and epoch. It ensures that weights are properly
    sparsified during training and finalized at the end of training.
    
    Args:
        module: The module to manage parametrization for.
        mode (bool, optional): Whether the module is in training mode. Defaults to True.
        
    Returns:
        module: The module with updated parametrization.
        
    Note:
        This function implements a key part of the sparsity training workflow:
        
        1. During training (mode=True):
           - Removes any existing parametrization
           - Increments the epoch counter
           - Inserts new parametrization with soft masks for gradual sparsification
           
        2. During final evaluation (mode=False and final epoch):
           - Applies hard binary masks to finalize weight sparsity
           - Calculates and reports the final sparsity level
    """
    # TODO: cleanup
    # if mode: # Training mode
    #     # Remove any existing parametrization without keeping the mask
    #     # This ensures weights are adjusted according to the new mask that will be created
    #     # remove_parametrization(module, leave_parameterized=False)
        
    #     # Increment epoch counter for sparsity scheduling
    #     module.__sparse_params__.epoch_count += 1
    #     update_all_parametrizations(module)
    #     # insert_parametrization(module, update=True)
    #     # Insert new parametrization (soft mask during training)
    #     # insert_all_parametrizations(module)
    #     # insert_all_parametrizations(module)
        
    # elif module.__sparse_params__.epoch_count==module.__sparse_params__.total_epochs: # evaluation in the final epoch, we would want to completely sparse out the weights
    #     # At the final epoch, finalize sparsity by applying hard binary masks
    #     update_all_parametrizations(module, binary_mask=True)
    #     remove_all_parametrizations(module, leave_parametrized=True) 
    #     # remove_parametrization(module, leave_parameterized=False) # we do not want to keep the old mask, rest of the weights are adjusted according to this one
    #     # insert_all_parametrizations(module, binary_mask=True) # binary_mask=True gives hard mask
    #     # remove_parametrization(module) # Apply the mask permanently to the weights
        
    #     # Calculate and report final sparsity level
    #     calculate_sparsity(module)
    #     print("The final sparsity of the network is {}".format(module.__sparse_params__.sparsity))
    
    return module

def calculate_sparsity(module):
    """Calculates the current sparsity level of the module.
    
    This function computes the overall sparsity ratio of the module by counting
    the number of zero elements in parametrized tensors and dividing by the
    total number of elements.
    
    Args:
        module: The module to calculate sparsity for.
        
    Returns:
        module: The module with updated sparsity value in __sparse_params__.
    
    Note:
        The calculated sparsity is stored in module.__sparse_params__.sparsity.
        This is a global sparsity calculation across all parametrized tensors.
        
    TODO: Implement layer-wise sparsity calculation for more detailed analysis.
    """
    num_zeros = 0     # Counter for zero-valued elements
    num_elements = 0  # Counter for total elements
    
    # Get all parameters and modules for reference
    params = dict(module.named_parameters())
    modules = dict(module.named_modules())
    
    # Count zeros across all parametrized parameters
    for param_name in module.__sparse_params__.parametrized_params:
        # Split parameter name into module path and parameter name
        # For example, "layer1.conv.weight" becomes ("layer1.conv", "weight")
        parent_module = param_name.rsplit('.',1)
        parent_module, param_name = parent_module if len(parent_module)>1 else ('', parent_module[0])
        
        # Get the module and parameter tensor
        parent_module = modules[parent_module]
        tensor = getattr(parent_module, param_name)
        
        # Count zeros and total elements in this tensor
        num_zeros += torch.sum(tensor==0).item()
        num_elements += torch.numel(tensor)

    # Calculate overall sparsity ratio as percentage of zero elements
    module.__sparse_params__.sparsity = num_zeros / num_elements
    return module

def remove_parametrization(module: fx.GraphModule, leave_parameterized=True):
    """Removes parametrization from the module's parameters.
    
    This function removes sparsity parametrization from the module's parameters,
    optionally keeping either the original or parametrized values. It traverses all
    parameters looking for those with parametrizations applied and removes them.
    
    Args:
        module (fx.GraphModule): The module to remove parametrization from.
        leave_parameterized (bool, optional): Whether to keep the parametrized values (True)
            or revert to the original values (False). Defaults to True.
            
    Returns:
        module: The module with parametrization removed.
    
    Note:
        When leave_parameterized=True (default), the sparsified weights are kept.
        When leave_parameterized=False, the original unsparsified weights are restored.
        
        This function identifies parametrized tensors by looking for parameters with
        the naming pattern: module_path.parametrizations.param_name.original
    """
    # Get all parameters and modules
    params = dict(module.named_parameters()) 
    modules = dict(module.named_modules())
    
    # Identify and process parametrized parameters
    for name, param in params.items():
        names = name.split('.')
        
        # Check if this parameter is an 'original' parameter inside a parametrization
        # Pattern to match: module_path.parametrizations.param_name.original
        if len(names) >= 3 and names[-1] == 'original' and names[-3] == 'parametrizations':
            # Extract module path and parameter name
            parent_module = '.'.join(names[:-3])
            param_name = names[-2]
            parent_module = modules[parent_module]
            
            # Remove parametrization if it exists
            if parametrize.is_parametrized(parent_module, param_name):
                # Track this parameter as having been parametrized
                module.__sparse_params__.parametrized_params.add('.'.join(names[:-3]+names[-2:-1]))
                
                # Remove parametrization, either keeping the sparsified tensor or original
                parametrize.remove_parametrizations(parent_module, param_name, leave_parametrized=leave_parameterized) 
    
    return module

def remove_all_parametrizations(module:fx.GraphModule, leave_parametrized:bool=False):
    """
    Remove all parametrizations added for sparsity.
    If leave_parametrized=True, permanently set the sparse weights. Otherwise, restore original.

    Assumes module.__sparse_params__.parametrization_list has been populated by insert_all_parametrizations
    Leaves module.__sparse_params__.parametrization_list empty at end of execution

    Args:
        module (fx.GraphModule): top level module
        leave_parametrized (bool, optional): Permanently sparsify weights if True. Defaults to False.
    """
    while module.__sparse_params__.parametrization_list:
        (parent_module, parent_module_name, param_name, parametrization) = module.__sparse_params__.parametrization_list.pop()
        if parametrize.is_parametrized(parent_module, param_name):
            module.__sparse_params__.parametrized_params.add(f'{parent_module_name}.{param_name}')
            parametrize.remove_parametrizations(parent_module, param_name, leave_parametrized=leave_parametrized)
        

def insert_all_parametrizations(module:fx.GraphModule, binary_mask:bool=False):
    """
    For each parameter to be sparsified, adds a parametrization.
    Assumes module.__sparse_params__.sparsity_nodes: dict[tuple, list[list[Node]]] has already been computed, e.g. using get_sparsity_nodes 
    Assumes module.__sparse_params__.sparsity_class has the class for parametrization
    Assumes module.__sparse_params__ has all keyword arguments to construct class, based on sparsity_class.REQUIRED_SPARSE_PARAMS

    Populates module.__sparse_params__.parametrization_list , clears old list TODO: perhaps change this behaviour. 
        This list contains (parent_module, param_name, parametrization) for each inserted parametrization

    Args:
        module (fx.GraphModule): Parent module
        binary_mask (bool, optional): Passed on to parametrization class init. Defaults to False.
    """

    parametrization_list = [] # (parent_module, parent_module_name, param_name, parametrization)
    modules = dict(module.named_modules())
    params = dict(module.named_parameters())
    sparsity_class = module.__sparse_params__.sparsity_class

    parametrization_kwargs = module.__sparse_params__.parametrization_kwargs
    parametrization_kwargs.update({'binary_mask': binary_mask})
    # dict(binary_mask=binary_mask)
    # for param in sparsity_class.REQUIRED_SPARSE_PARAMS:
    #     if hasattr(module.__sparse_params__, param):
    #         kwargs[param] = getattr(module.__sparse_params__, param)

    for key, nodes_list in module.__sparse_params__.sparsity_nodes.items(): # dict[tuple, list[list[Node]]]
        for nodes in nodes_list:
            weight_names = get_weights(module, nodes, key)
            main_weight_name = weight_names[0] # TODO: improve this once we define list[node] meaning per parametrization
            param_tensor = params[main_weight_name]

            for weight_name in weight_names:
                parent_module_name, param_name = get_parent_name(weight_name)
                parent_module = modules[parent_module_name]
                parametrization = sparsity_class(key, nodes, tensor=param_tensor, **parametrization_kwargs)
                parametrization_list.append((parent_module, parent_module_name, param_name, parametrization))

                parametrize.register_parametrization(parent_module, param_name, parametrization)
    module.__sparse_params__.parametrization_list = parametrization_list

def update_all_parametrizations(module:fx.GraphModule, binary_mask:bool =False) -> None:
    """
    Call 'update' on all parametrizations registered to the module IN module.__sparse_params__.parametrization_list
    NOTE: It is the responsibility of wherever parametrizations are added, to add to the above list.

    Args:
        module (fx.GraphModule): Parent module
        binary_mask (bool, optional): passed on to parametrization update function. Defaults to False.
    """
    flip_rate_list = []
    scale_list = []
    current_epoch = -1
    for (parent_module, parent_module_name, param_name, parametrization) in module.__sparse_params__.parametrization_list:
        if parametrize.is_parametrized(parent_module, param_name):
            param_tensor = nested_getattr(parent_module, f'parametrizations.{param_name}.original') # get original param
            old_mask = (parametrization.mask == 1) # also works for topk where masked element is nonzero
            new_mask = parametrization.update(tensor=param_tensor, binary_mask=binary_mask)
            new_mask = (new_mask==1)
            flip_rate = (old_mask!=new_mask).sum()/(old_mask.numel())
            flip_rate_list.append(flip_rate)
            if hasattr(parametrization, 'scale_weight_factor') and parametrization.scale_weight_factor is not None:
                scale_list.append(parametrization.scale_weight_factor)
            current_epoch = parametrization.current_epoch
    flip_rate_avg = torch.tensor(flip_rate_list).mean().item()
    scale_avg = torch.tensor(scale_list).mean().item()
    
    # TODO, check verbose mode somewhere..
    try:
        import mlflow
        if mlflow.active_run():
            if current_epoch > 1:
                mlflow.log_metric('avg_flip_rate', flip_rate_avg, step=current_epoch)
                mlflow.log_metric('avg_weight_scale', scale_avg, step=current_epoch)
    except Exception:
        pass
    # print(f'update_all_parametrizations, epoch {current_epoch}, avg flip rate%={flip_rate_avg*100}%')

# def insert_all_parametrizations(module:fx.GraphModule, binary_mask=False):
#     """Inserts sparsity parametrization into the module's parameters.
    
#     This function adds sparsity parametrization to the module's weight parameters
#     based on the identified sparsity nodes and configured sparsity class. The
#     parametrization applies masks to weights during forward passes to enforce sparsity.
    
#     Args:
#         module (fx.GraphModule): The module to add parametrization to.
#         binary_mask (bool, optional): Whether to use binary masks for sparsity.
#             When True, creates hard masks for final sparsification (0 or 1).
#             When False, uses soft masks during training (gradual sparsification). 
#             Defaults to False.
            
#     Returns:
#         None: The function modifies the module in-place by registering parametrizations.
        
#     Note:
#         The sparsity parametrization works by applying masks to weight tensors during
#         forward passes. This function creates appropriate mask generators based on the
#         sparsity class (e.g., N2MSparsityParametrization) and registers them with PyTorch's
#         parametrization system.
#     """
#     # Get parameters, modules, and sparsity configuration
#     params = dict(module.named_parameters())
#     modules = dict(module.named_modules())
#     weights_dict = module.__sparse_params__.weights
#     sparsity_class = module.__sparse_params__.sparsity_class

    
#     # Prepare arguments for the sparsity parametrization
#     kwargs = dict(binary_mask=binary_mask)

#     # Add all required parameters from sparse_params to kwargs
#     # This ensures the parametrization class has all the parameters it needs
#     for param in sparsity_class.REQUIRED_SPARSE_PARAMS:
#         if hasattr(module.__sparse_params__, param):
#             kwargs[param] = getattr(module.__sparse_params__, param)
    
#     # Process each sparsity node and its associated weights
#     for key, nodes_list in module.__sparse_params__.sparsity_nodes.items():
#         weights_list = weights_dict[key]
        
#         # Process each node and its weights
#         for nodes, weights in zip(nodes_list, weights_list):
#             # Normalize weights representation to a consistent format
#             # Convert set to list if needed
#             if isinstance(weights, set):
#                 weights = list(weights)
                
#             # Convert list/tuple to dict mapping each weight to itself
#             if isinstance(weights, (list, tuple)):
#                 weights = {x:[x] for x in weights}
                
#             # Ensure weights is now a dictionary
#             assert isinstance(weights, dict), 'Weight should be a dict by now!'
            
#             # Process each weight that needs parametrization
#             for main_weight, weights_to_be_sparsified in weights.items():

#                 # Get the tensor for the main weight
#                 tensor = params[main_weight]
                
#                 # Apply parametrization to all weights that should be sparsified
#                 for weight in weights_to_be_sparsified:
#                     # Parse the weight name to get parent module and parameter name
#                     # For example, "layer1.conv.weight" becomes ("layer1.conv", "weight")
#                     parent_module = weight.rsplit('.',1)
#                     parent_module, name = parent_module if len(parent_module)>1 else ('', parent_module[0])
#                     parent_module = modules[parent_module]

                    # # Create parametrization instance for this tensor
                    # # This will generate appropriate masks based on the sparsity type and configuration
                    # parametrization = sparsity_class(key, nodes, tensor=tensor, **kwargs)
                    # # Register the parametrization to be applied during forward passes
                    # parametrize.register_parametrization(parent_module, name, parametrization)
