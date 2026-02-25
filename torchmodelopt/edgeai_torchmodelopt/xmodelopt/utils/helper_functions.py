import torch
from torch import fx, nn
from typing import List,Dict,Type, Any
import types
from typing import List,Dict,Type, Any
import types
from torch.fx.passes.utils.source_matcher_utils import SourcePartition
import importlib
from functools import reduce


def get_module(m: torch.nn.Module, target:str):
    """Retrieves a module from a PyTorch model by its target name.
    
    This function traverses a PyTorch module hierarchy to find a submodule 
    specified by the target string. It can handle both direct module access 
    and nested module paths (e.g. 'layer1.0.conv').
    
    Args:
        m (torch.nn.Module): The parent module to search within.
        target (str): The target module name or path to retrieve.
        
    Returns:
        torch.nn.Module: The requested module if found.
    """
    try:
        modules = dict(m.named_modules())
        if target in modules:
            return modules[target]
    except:
        pass
    attr = target.split('.',1)
    if len(attr) == 1:
        attr = attr[0]    
        if hasattr(m, attr):
            return (getattr(m,attr))
    else:
        attr, target = attr
        try:
            modules = dict(m.named_modules())
            if attr in modules:
                m =  modules[attr]
        except:
            if hasattr(m, attr):
                m = getattr(m, attr)
        return get_module(m, target)


def get_attr(model, target):
    """Retrieves an attribute (parameter or buffer) from a PyTorch model by its target name.
    
    This function accesses parameters or buffers within a PyTorch model, handling both
    directly accessible attributes and nested attributes within submodules.
    
    Args:
        model (torch.nn.Module): The model to retrieve the attribute from.
        target (str): The target attribute name or path to retrieve.
        
    Returns:
        torch.Tensor: The requested parameter or buffer if found.
    """
    attrs = dict(model.named_parameters())
    attrs.update(model.named_buffers())
    if target in attrs:
        return attrs[target]
    split = target.rsplit('.',1)
    if len(split) == 1:
        return getattr(model, target)
    parent, target = split
    return getattr(get_module(model, parent), target)

def get_parent_name(target:str) -> tuple[str, str]:
    """gets the name of the parent module and attribute name of the module from the target of the module

    Args:
        target (str): parameter/submodule name, e.g. layer2.1.conv2.weight

    Returns:
        tuple[str, str]: Returns parent_name, target's name, e.g., 
    """
    ''''''
    *parent, name = target.rsplit('.', 1)
    return ( parent[0] if parent else ''), name

def get_class_string(cls):
    """Returns a fully qualified string representation of a class.
    
    Args:
        cls: The class to convert to a string representation.
        
    Returns:
        str: A string in the format 'module.class_name'.
    """
    return f'{cls.__module__}.{cls.__name__}'

def is_same_class(source: str|type, cls: type) ->  bool:
    """Checks if a source matches a given class type.
    
    This function compares a source (which can be either a string representation 
    or a class type) with a class type. It's useful when dealing with source partitions 
    where the source might be represented as either a string or a class object.
    
    Args:
        source (str|type): The source to compare, either as a string or a class type.
        cls (type): The class type to compare against.
        
    Returns:
        bool: True if the source matches the class, False otherwise.
        
    Example:
        is_same_class(partition.source, nn.Conv2d)
    """
    return (source == cls) or (source == get_class_string(cls))

def get_class(source):
    """Converts a source to its class type.
    
    This function handles sources that might be either a string representation 
    or a class type, and returns the corresponding class object. It's useful
    when working with source partitions where the source representation is variable.
    
    Args:
        source (str|type): The source to convert, either as a string or a class type.
        
    Returns:
        type: The class object represented by the source.
        
    Example:
        cls = get_class(partition.source)
        instance = cls(*args)
    """
    if isinstance(source, str):
        module_name, class_name = source.rsplit('.', 1)
        # Import the module dynamically
        module = importlib.import_module(module_name)
        # Get the class from the module
        cls = getattr(module, class_name)

        return cls
    else:
        return source # assume isinstance(source, type)

def nested_getattr(obj: Any, attr_path: str, default: Any=None):
    """Retrieves a nested attribute from an object using a dot-separated path.
    
    This function extends the built-in getattr to support accessing nested attributes
    through a dot-separated string path. It handles arbitrary levels of nesting.
    
    Args:
        obj (Any): The object to retrieve the attribute from.
        attr_path (str): A dot-separated path to the desired attribute.
        default (Any, optional): The value to return if the attribute is not found.
            Defaults to None.
            
    Returns:
        Any: The value of the nested attribute if found, otherwise the default value.
        
    Example:
        # Returns obj.layer1[0].weight
        weight = nested_getattr(obj, 'layer1.0.weight')
    """
    try:
        return reduce(getattr, attr_path.split('.'), obj)
    except AttributeError:
        return default

# Note: The source code is copied from pytorch github (https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/utils/source_matcher_utils.py#L51)
# and modified as per requirement 
def get_source_partitions(graph:fx.Graph, wanted_sources:list, filter_fn = None):
    '''
    A custom made get_source_partitions that can handle any type of modules and functions that are wrapped for fx.
    
    This function identifies and partitions nodes in a PyTorch FX graph based on the source modules or functions
    they originate from. It's particularly useful for isolating specific parts of a model for targeted operations.
    
    Args:
        graph (fx.Graph): The PyTorch FX graph to analyze.
        wanted_sources (list): List of source modules or functions to look for in the graph.
            Can include both class objects and string representations of classes.
        filter_fn (callable, optional): Function to filter nodes based on custom criteria.
        
    Returns:
        Dict[Type[Any], List[SourcePartition]]: Dictionary mapping source types to lists of 
        source partitions. Each source partition contains nodes from the graph that
        originate from the corresponding source.
    '''
    
    # Convert class objects to string representations for comparison
    class_names = [get_class_string(s) for s in wanted_sources if isinstance(s, type)]
    class_names = [get_class_string(s) for s in wanted_sources if isinstance(s, type)]
    name_2_class_dict = dict(zip(class_names,[s for s in wanted_sources if isinstance(s, type)]))
    wanted_sources += class_names
    
    # Dictionary to store modules and their corresponding nodes
    modules: Dict[Type, Dict[str, List[fx.Node]]] = {}
    
    def get_all_args(args:list):
        """Recursively extracts all fx.Node objects from a list of arguments."""
        result = []
        for arg in args:
            if isinstance(arg,fx.Node):
                result.append(arg)
            elif isinstance(arg,(list,tuple)):
                result.extend(get_all_args(arg))
        return result
    
    def add_node_to_partition(source_fn:tuple,node:fx.Node):
        """Adds a node to the appropriate source partition."""
        diff_modules = modules.setdefault(source_fn[1], {})
        partition = diff_modules.setdefault(source_fn[0], [])
        partition.append(node) if node not in partition else None
    
    # Traverse the graph and collect nodes into partitions
    for node in graph.nodes:
        found = False
        if (nn_module_stack:= node.meta.get("nn_module_stack", None)):
            for k,v in nn_module_stack.items():
                if v[1] in wanted_sources:
                    key = k
                    source_fn = nn_module_stack[key]
                    add_node_to_partition(source_fn, node)
                    found = True
                    if not ((isinstance(v[1], type) and issubclass(v[1],nn.Sequential) )or v[1] == '.'.join([str(nn.Sequential.__module__), str(nn.Sequential.__name__)])):
                        break 

        if not found and (source_fn_st := node.meta.get("source_fn_stack", None)):
            source_fn = source_fn_st[-1]
            if source_fn[1] not in wanted_sources:
                continue
            add_node_to_partition(source_fn, node)
        elif not found and (torch_fn := node.meta.get("torch_fn", None)):
            node_fqn, source_fn = torch_fn
            source_fn_name = source_fn.split(".")[1]
            if source_fn_name not in wanted_sources:
                continue
            add_node_to_partition((node_fqn,source_fn_name), node)
    
    def make_partition(nodes: List[fx.Node], module_type: Type) -> SourcePartition:
        """Creates a SourcePartition from a list of nodes."""
        input_nodes = set()
        output_nodes = set()
        params = set()
        for i, node in enumerate(nodes):
            if node.op != 'call_module':
                continue
            meta_keys = list(node.meta)
            #TODO make sure this check is correct
            if len(meta_keys) == 1 and meta_keys[0] == 'nn_module_stack' and '_guards_fn' in node.target:
                nodes.remove(node)
            
        # Identify input nodes, output nodes, and parameter nodes
        for node in nodes:
            for arg in get_all_args(node.args):
                if isinstance(arg, fx.Node) and arg not in nodes and arg.op != "get_attr":
                    input_nodes.add(arg)

            if node.op == "get_attr":
                params.add(node)
                continue

            for user in node.users.keys():
                if user not in nodes:
                    output_nodes.add(node)
        
        # Special handling for PyTorch 2.9 compatibility
        # happens in torch==2.9 as per tests, but does not happen in torch==2.4
        if len(params) == 0:
            param_pos = dict()
            for i, node in enumerate(nodes):
                for arg in get_all_args(node.args):
                    if isinstance(arg, fx.Node) and arg not in nodes and arg.op == "get_attr":
                        params.add(arg)
                        if arg not in param_pos:
                            param_pos[arg] = i 
            for i, (p, idx) in enumerate(param_pos.items()):
                nodes.insert((i+idx), p, )
                
        # Sort nodes for consistency
        input_nodes = sorted(input_nodes, key= lambda node : min([nodes.index(user) for user in node.users.keys() if user in nodes])) 
        params = sorted(params, key = lambda node: nodes.index(node))
        output_nodes = sorted(output_nodes, key = lambda node: nodes.index(node))
        
        # Create and return the partition
        return SourcePartition(
            nodes,
            module_type,
            list(input_nodes),
            list(output_nodes),
            list(params),  # type: ignore[arg-type]
        )
    
    # Apply filter function if provided
    ret: Dict[Type[Any], List[SourcePartition]] = {}
    filter_fn = None
    if filter_fn:
        # for each partition, we apply filter_fn to filter out all partitions that doesn't satisfy the
        # filter condition
        filtered_modules = {}
        for tp, name_to_partition in modules.items():
            filtered_name_to_partition = {
                name: partition
                for name, partition in name_to_partition.items()
                if all(map(filter_fn, partition))
            }
            filtered_modules[tp] = filtered_name_to_partition
        modules = filtered_modules

    def separate_partitions(partitions:list[fx.Node]):
        """Separates a list of nodes into distinct partitions."""
        reuslt = []
        temp = []
        for node in partitions:
            temp.append(node)
            if all(user not in partitions for user in node.users) and node.next not in partitions:
                reuslt.append(temp)
                temp = []
        return reuslt
    
    # Create SourcePartition objects for each module and partition
    for k, v in modules.items():
        ret[k] = []
        for key,partitions in v.items():
            for partition in separate_partitions(partitions):
                ret[k].append(make_partition(partition, name_2_class_dict.get(k,k)))

    # Convert class string keys back to actual class objects
    for cls in class_names:
        if cls in ret:
            ret[name_2_class_dict[cls]] = ret.pop(cls)
    
    return ret

_EXPORTED_TRAINING_ATTR = "_exported_training"

class WrapperModule(torch.nn.Module):
    """Class to wrap a callable in an :class:`torch.nn.Module`. Use this if you
    are trying to export a callable.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        """Simple forward that just calls the ``fn`` provided to :meth:`WrapperModule.__init__`.
        
        Args:
            *args: Positional arguments to pass to the wrapped function.
            **kwargs: Keyword arguments to pass to the wrapped function.
            
        Returns:
            The result of calling the wrapped function with the provided arguments.
        """
        return self.fn(*args, **kwargs)

def _change_arg_or_kwargs_(m:fx.GraphModule, fn_1, fn_2, example_inputs, example_kwargs={}):
    """Changes arguments or keyword arguments in function calls within a GraphModule.
    
    This function identifies differences between two wrapper functions and updates 
    the corresponding function calls in a GraphModule. It's primarily used to switch
    between training and evaluation modes for operations like batch normalization and dropout.
    
    Args:
        m (fx.GraphModule): The GraphModule to modify.
        fn_1 (callable): The first function to compare.
        fn_2 (callable): The second function to compare, which provides the replacement values.
        example_inputs (tuple): Example inputs to use when exporting the functions.
        example_kwargs (dict, optional): Example keyword arguments to use when exporting the functions.
            Defaults to an empty dict.
    
    Raises:
        ValueError: If the functions being compared have incompatible operations or structures.
    """
    fn_1 = WrapperModule(fn_1)
    fn_2 = WrapperModule(fn_2)
    fn_1 = torch.export.export(
        fn_1,  # type: ignore[arg-type]
        example_inputs,
        example_kwargs,
        strict=True,
    ).module()
    
    fn_2 = torch.export.export(
        fn_2,  # type: ignore[arg-type]
        example_inputs,
        example_kwargs,
        strict=True,
    ).module()
    

    bool_dict = {}
    
    fn = None
    keys = []
    args = []
    for n1, n2 in zip(list(fn_1.graph.nodes), list(fn_2.graph.nodes)):
        n1.name += '_train'
        n2.name += '_eval'
        if n1.op != n2.op :
            raise ValueError (f'{n1}.op ({n1.op}) should be equal to {n2}.op ({n2.op})')
        if n1.op != 'call_function':
            continue
        if n1.target != n2.target :
            raise ValueError (f'{n1}.target ({n1.target}) should be equal to {n2}.target ({n2.target})')
        if len(n1.args) != len(n2.args) or len(n1.kwargs) != len(n2.kwargs):
            raise ValueError (f'length of {n1}\'s args ({len(n1.args)}) or kwargs ({len(n1.kwargs)}) should be equal to length of {n2}\'s args ({len(n2.args)}) or kwargs ({len(n2.kwargs)})')
        if not all(k in n2.kwargs for k in n1.kwargs):
            raise ValueError(f'kwargs of {n1} ({list(n1.kwargs.keys())}) does not match with ({n2}) ({list(n2.kwargs.keys())})')
        for k in n1.kwargs:
            v1 = n1.kwargs[k]
            v2 = n2.kwargs[k]
            if v1.__class__ != v1.__class__:
                keys.append(k)
                continue
            if isinstance(v1, fx.Node) and bool_dict.get((v1.name, v2.name), False):
                keys.append(k)
                continue
            if v1 != v2:
                keys.append(k)
        for i, (v1, v2) in enumerate(zip(n1.args, n2.args)):
            if v1.__class__ != v1.__class__:
                args.append(i)
                continue
            if isinstance(v1, fx.Node) and bool_dict.get((v1.name, v2.name), False):
                args.append(i)
                continue
            if v1 != v2:
                args.append(i)
        if len(args+keys) == 0:
            bool_dict[(n1.name, n2.name)] = True
            continue
        fn = n1.target
        break
    
    if fn is None:
        return
    
    # Update the training mode for batch normalization nodes
    for node in m.graph.nodes:
        if node.op == 'call_function' and node.target == fn:
            for i in args:
                if isinstance(node.args[i] , fx.Node):
                    raise ValueError(f'{node}\'s args at {i} is a node which should not change')
                if n1.args[i] == node.args[i]:
                    node.update_arg(i, n2.args[i])
            for k in keys:
                if isinstance(node.kwargs[k] , fx.Node):
                    raise ValueError(f'{node}\'s kwargs with key {k} is a node which should not change')
                if n1.kwargs[k] == node.kwargs[k]:
                    node.update_kwarg(k, n2.kwargs[k])
    m.recompile()


def _replace_batchnorm(m: torch.fx.GraphModule, train_to_eval: bool):
    """Replaces batch normalization nodes in a GraphModule to set their training mode.
    
    Args:
        m (torch.fx.GraphModule): The GraphModule to modify.
        train_to_eval (bool): If True, sets the training mode to False (eval mode).
            If False, sets the training mode to True (train mode).
    """
    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()
    from torch.nn import functional as F
    def bn_train(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True
        )

    def bn_eval(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False
        )

    example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    devices = {p.device for p in m.parameters()} | {
        p.device for p in m.buffers()
    }

    assert len(devices) <= 1, (
        "prepare only works with cpu or single-device CUDA modules, "
        f"but got devices {devices}"
    )
    device = next(iter(devices)) if len(devices) > 0 else None
    is_cuda = device is not None and device.type == "cuda"
    if is_cuda:
        example_inputs = tuple(
            [x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs]
        )

    if train_to_eval:
        _change_arg_or_kwargs_(m, bn_train, bn_eval, example_inputs)
    else:
        _change_arg_or_kwargs_(m, bn_eval, bn_train, example_inputs)
    
    return


def _replace_dropout(m: torch.fx.GraphModule, train_to_eval: bool):
    """Replaces dropout nodes in a GraphModule to set their training mode.
    
    Args:
        m (torch.fx.GraphModule): The GraphModule to modify.
        train_to_eval (bool): If True, sets the training mode to False (eval mode).
            If False, sets the training mode to True (train mode).
    """
    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()
    from torch.nn import functional as F
    def dropout_train(x):
        return F.dropout(x, p=0.5, training=True, )

    def dropout_eval(x):
        return F.dropout(x, p=0.5, training=False,)
    
    example_inputs = (torch.randn(1),)
    devices = {p.device for p in m.parameters()} | {
        p.device for p in m.buffers()
    }

    assert len(devices) <= 1, (
        "prepare only works with cpu or single-device CUDA modules, "
        f"but got devices {devices}"
    )
    device = next(iter(devices)) if len(devices) > 0 else None
    is_cuda = device is not None and device.type == "cuda"
    if is_cuda:
        example_inputs = tuple(
            [x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs]
        )
    
    # Update the training mode for dropout nodes
    if train_to_eval:
        _change_arg_or_kwargs_(m, dropout_train, dropout_eval, example_inputs)
    else:
        _change_arg_or_kwargs_(m, dropout_eval, dropout_train, example_inputs)
    
    return


def _move_exported_model_to_train(model, mode: bool=True):
    """Moves an exported model to training mode.
    
    Args:
        model (torch.fx.GraphModule): The exported model to modify.
        mode (bool, optional): The training mode to set. Defaults to True.
        
    Returns:
        torch.fx.GraphModule: The modified model.
    """
    # Check if the model is already in the desired mode
    if hasattr(model, 'training'):
        setattr(model, _EXPORTED_TRAINING_ATTR, model.training)
        
    is_training = getattr(model, _EXPORTED_TRAINING_ATTR, not mode)
    if is_training==mode:
        return model
        
    # Set the training attribute and update batchnorm and dropout nodes
    setattr(model, _EXPORTED_TRAINING_ATTR, mode)
    _replace_dropout(model, not mode)        
    _replace_batchnorm(model, not mode)        
    return model


def _move_exported_model_to_eval(model):
    """Moves an exported model to evaluation mode.
    
    Args:
        model (torch.fx.GraphModule): The exported model to modify.
        
    Returns:
        torch.fx.GraphModule: The modified model in evaluation mode.
    """
    return _move_exported_model_to_train(model, False)


def allow_exported_model_train_eval(model: fx.GraphModule):
    """Adds train() and eval() methods to an exported model.
    
    This function adds train() and eval() methods to an exported model, allowing
    it to behave like a regular PyTorch module with respect to training mode.
    
    Args:
        model (fx.GraphModule): The exported model to modify.
        
    Returns:
        fx.GraphModule: The model with train() and eval() methods added.
    """
    # Define the train method
    def _train(self, mode: bool = True):
        if mode:
            _move_exported_model_to_train(self)
        else:
            _move_exported_model_to_eval(self)

    # Define the eval method
    def _eval(self):
        _move_exported_model_to_eval(self)

    # Attach the methods to the model
    model.train = types.MethodType(_train, model)  # type: ignore[method-assign]
    model.eval = types.MethodType(_eval, model)  # type: ignore[method-assign]
    return model


def get_tensors_to_device(entries, device):
    """Recursively moves tensors to the specified device.
    
    Args:
        entries: An object, which can be a tensor, a collection of tensors,
            or any other type of object.
        device: The device to move tensors to (e.g., 'cuda', 'cpu').
        
    Returns:
        The input object with any contained tensors moved to the specified device.
    """
    # Handle dictionary inputs
    if isinstance(entries, dict):
        for k,v in entries.items():
            entries[k] = get_tensors_to_device(v, device)
    # Handle collection inputs
    elif isinstance(entries, (list, tuple, set)):
        entries = entries.__class__([get_tensors_to_device(v, device) for v in entries])
    # Handle tensor inputs
    elif isinstance(entries, torch.Tensor):
        entries= entries.to(device)
    # Leave other types unchanged
    else:
        entries = entries
    return entries


def _model_to_device(model, device):
    """Moves a PyTorch model to the specified device with special handling for GraphModules.
    
    This function moves a PyTorch model to a specified device and ensures that any device
    references within FX GraphModule nodes are also updated to maintain consistency.
    
    Args:
        model: The PyTorch model to move to the device.
        device: The target device to move the model to (e.g., 'cuda', 'cpu').
        
    Returns:
        The model after moving it to the specified device.
    """
    if device:
        # Convert string device specification to torch.device object if needed
        if not isinstance(device, torch.device):
            device = torch.device(device)
        
        # Move the model parameters and buffers to the specified device
        model.to(device)
        
        # TODO to handle cases where device are not used for tensors or in some edge cases 
        if isinstance(model, torch.fx.GraphModule):
            # For GraphModules, we also need to update any torch.device objects 
            # referenced in the computational graph
            for node in list(model.graph.nodes):
                # Update any device objects in the node's positional arguments
                for i, a in enumerate(node.args):
                    if isinstance(a, torch.device):
                        node.update_arg(i, device)
                
                # Update any device objects in the node's keyword arguments
                for k,v in node.kwargs.items():
                    if isinstance(v, torch.device):
                        node.update_kwarg(k, device)
            
            # Recompile the graph to ensure changes take effect
            model.recompile()
        #
    #
    return model
