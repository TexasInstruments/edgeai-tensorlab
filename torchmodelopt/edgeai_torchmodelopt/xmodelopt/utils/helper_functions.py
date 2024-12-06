
from torch import fx, nn
from typing import List,Dict,Type, Any, Iterable
from torch.fx.passes.utils.source_matcher_utils import SourcePartition

# Note: The source code is copied from pytorch github (https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/utils/source_matcher_utils.py#L51)
# and modified  as per requirement 
def get_source_partitions(graph:fx.Graph, wanted_sources:list, filter_fn = None):
    '''
    a custom made get_source_partitions that can handle any type of modules and functions that are wrapped for fx
    
    Note: This function is also defined on pruning.v3.utils. If this is modified later on, same changes have to be made on that function definition 
    
    '''
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
                    if not issubclass(v[1],nn.Sequential) :
                        break 

        if not found and (source_fn_st := node.meta.get("source_fn_stack", None)):
            source_fn = source_fn_st[-1]
            if source_fn[1] not in wanted_sources:
                continue
            add_node_to_partition(source_fn, node)
        else:
            continue

    
    def make_partition(nodes: List[fx.Node], module_type: Type) -> SourcePartition:
        input_nodes = set()
        output_nodes = set()
        params = set()
        for node in nodes:
            for arg in get_all_args(node.args):
                if arg not in nodes:
                    input_nodes.add(arg)

            if node.op == "get_attr":
                params.add(node)

            for user in node.users.keys():
                if user not in nodes:
                    output_nodes.add(node)

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

    return ret
