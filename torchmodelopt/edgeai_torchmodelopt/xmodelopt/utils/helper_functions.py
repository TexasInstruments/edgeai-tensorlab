import importlib
import torch
from torch import fx, nn
from typing import List,Dict,Type, Any
import types
from torch.fx.passes.utils.source_matcher_utils import SourcePartition
from functools import reduce

def get_class_string(cls):
    return f'{cls.__module__}.{cls.__name__}'

def is_same_class(source: str|type, cls: type) ->  bool:
    '''
        use to compare when not sure if source is string or class
        Makes parition.source backwards compatible, if get_source_paritition behavior changes
        Intented usage example: is_same_class(partition.source, nn.Conv2d)
    '''
    return (source == cls) or (source == get_class_string(cls))

def get_class(source):
    '''
        use to get class object when not sure if source is string or class
        Intented usage example: get_class(parition.source)(args...)
    '''
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
    '''
        Like getattr, but works for nested attribute paths. 
        e.g.: If attr_path = 'layer1.0.weight', it will return obj.layer1.0.weight
    '''
    try:
        return reduce(getattr, attr_path.split('.'), obj)
    except AttributeError:
        return default


def get_class_string(cls):
    return f'{cls.__module__}.{cls.__name__}'

# Note: The source code is copied from pytorch github (https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/utils/source_matcher_utils.py#L51)
# and modified  as per requirement 
def get_source_partitions(graph:fx.Graph, wanted_sources:list, filter_fn = None):
    '''
    a custom made get_source_partitions that can handle any type of modules and functions that are wrapped for fx
    
    Note: This function is also defined on pruning.v3.utils. If this is modified later on, same changes have to be made on that function definition 
    
    '''
    
    class_names = [get_class_string(s) for s in wanted_sources if isinstance(s, type)]
    name_2_class_dict = dict(zip(class_names,[s for s in wanted_sources if isinstance(s, type)]))
    wanted_sources += class_names
    
    
    modules: Dict[Type, Dict[str, List[fx.Node]]] = {}
    def get_all_args(args:list):
        result = []
        for arg in args:
            if isinstance(arg,fx.Node):
                result.append(arg)
            elif isinstance(arg,(list,tuple)):
                result.extend(get_all_args(arg))
        return result
    
    def add_node_to_partition(source_fn:tuple,node:fx.Node):
        diff_modules = modules.setdefault(source_fn[1], {})
        partition = diff_modules.setdefault(source_fn[0], [])
        partition.append(node) if node not in partition else None
    
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
        input_nodes = sorted(input_nodes, key= lambda node : min([nodes.index(user) for user in node.users.keys() if user in nodes])) 
        params = sorted(params, key = lambda node: nodes.index(node))
        output_nodes = sorted(output_nodes, key = lambda node: nodes.index(node))
        return SourcePartition(
            nodes,
            module_type,
            list(input_nodes),
            list(output_nodes),
            list(params),  # type: ignore[arg-type]
        )
    
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
        reuslt = []
        temp = []
        for node in partitions:
            temp.append(node)
            if all(user not in partitions for user in node.users) and node.next not in partitions:
                reuslt.append(temp)
                temp = []
        return reuslt
    
    for k, v in modules.items():
        ret[k] = []
        for key,partitions in v.items():
            for partition in separate_partitions(partitions):
                ret[k].append(make_partition(partition, k))

    for cls in class_names:
        if cls in ret:
            ret[name_2_class_dict[cls]] = ret.pop(cls)
    
    return ret

_EXPORTED_TRAINING_ATTR = "_exported_training"

def _replace_batchnorm(m: torch.fx.GraphModule, train_to_eval: bool):
    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()
    
    for node in m.graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.aten.batch_norm.default:
            if 'training' in node.kwargs:
                node.update_kwarg('training', not train_to_eval)
            else:
                node.update_arg(5, not train_to_eval)
    
    
    m.recompile()
    return


def _replace_dropout(m: torch.fx.GraphModule, train_to_eval: bool):
    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()
    for node in m.graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.aten.dropout.default:
            if 'train' in node.kwargs:
                node.update_kwarg('train', not train_to_eval)
            else:
                node.update_arg(2, not train_to_eval)
    
    m.recompile()
    return


def _move_exported_model_to_train(model, mode: bool=True):
    is_training = getattr(model, _EXPORTED_TRAINING_ATTR, not mode)
    if is_training==mode:
        return model
    setattr(model, _EXPORTED_TRAINING_ATTR, mode)
    _replace_dropout(model, not mode)        
    _replace_batchnorm(model, not mode)        
    return model
def _move_exported_model_to_eval(model):
    return _move_exported_model_to_train(model, False)

def allow_exported_model_train_eval(model: fx.GraphModule):
    def _train(self, mode: bool = True):
        if mode:
            _move_exported_model_to_train(self)
        else:
            _move_exported_model_to_eval(self)

    def _eval(self):
        _move_exported_model_to_eval(self)

    model.train = types.MethodType(_train, model)  # type: ignore[method-assign]
    model.eval = types.MethodType(_eval, model)  # type: ignore[method-assign]
    return model