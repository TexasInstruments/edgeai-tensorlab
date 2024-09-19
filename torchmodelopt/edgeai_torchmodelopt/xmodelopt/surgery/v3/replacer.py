#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################

import warnings
import torch
import types
from torch import nn,fx,Tensor
from torch.fx import GraphModule
from typing import Iterable
from torch.fx.passes.utils.source_matcher_utils import SourcePartition
from typing import Any
import copy
from .custom_modules import ReplacedModule
from . import utils

# from .custom_symbolic_trace import custom_symbolic_trace
'''
this module's function are implemented to change nodes only.
no change is made on the incoming arguments and keyword arguments.
to do so self-made surgery functions are required.
for example, custom_surgery_functions module can be checked
'''


__net_module_replaced = None


def _remove_hanging_nodes(main_module:GraphModule):
    ''' 
    removes all hanging nodes so that the graph does have any node with 0 users
    '''
    def find_hanging_nodes(main_module:GraphModule):
        count =[]
        for node in main_module.graph.nodes:
            if (node.op not in ('output','placeholder') and len(node.users)==0):
                count.append(node)
        return count
    h_nodes=find_hanging_nodes(main_module)
    while len(h_nodes)>0:
        for node in h_nodes:
            main_module.graph.erase_node(node)
        h_nodes=find_hanging_nodes(main_module)
    main_module.delete_all_unused_submodules()
    main_module.graph.lint()
    main_module.recompile()


def adjust_meta(new_node:fx.Node, old_node:fx.Node, replace_node:fx.Node,partition: SourcePartition):
    '''
    adjusts meta for the new nodes of the main graph from old nodes from partition
    and the replace nodes from the replacement graph
    '''
    for meta_name in replace_node.meta.keys():
        if meta_name in ('source_fn_stack','seq_nr','example_value','val'):
            new_node.meta[meta_name] = replace_node.meta[meta_name]
        elif meta_name == 'from_node':
            new_node.meta[meta_name] = [(new_node.name,new_node.name)]
        elif meta_name == 'stack_trace':
            new_node.meta[meta_name] = old_node.meta[meta_name] + replace_node.meta[meta_name]
        elif meta_name == 'nn_module_stack':
            old_meta_val = list(old_node.meta[meta_name].items())
            index = 0
            while index<len(old_meta_val):
                if old_meta_val[index][1][1] == partition.source:
                    break
                index += 1 
            replace_meta_val = list(replace_node.meta[meta_name].items())
            new_node.meta[meta_name] = dict(old_meta_val[:index]+replace_meta_val)


def _get_partition_string(main_module:fx.GraphModule, partition:SourcePartition):
    modules = dict(main_module.named_modules())
    string = ''
    for i,node in enumerate(partition.nodes):
        string += f'{node.target}'
        if node.op == 'call_module':
            string += f'_{modules[node.target]}'
        if i < (len(partition.nodes) -1):
            string += '_'
    return string 

# replaces a pattern fom start to end in graph module to a call_module node
def _replace_pattern(main_module:GraphModule, partition:SourcePartition, replace_module:ReplacedModule, no_of_module_replaced:int=0, partition_module_map:dict[str,dict[str,tuple[str,nn.Module|nn.Parameter|torch.Tensor|Any]]] = {}):
    '''
    replaces nodes from a partition with the nodes of replacement module if it is graph module
    if it's a module other than a graph module it will add that to the main module and add it as call module to the graph
    '''
    inputs = [node for node in replace_module.module.graph.nodes if node.op == 'placeholder']
    node_mapping: dict[fx.Node,fx.Node] = {}
    node_mapping = dict(zip(inputs,partition.input_nodes)) # TODO create a proper mapping
    if len(partition.output_nodes) ==0:
        return
    main_output_nodes = list(partition.output_nodes[0].users.keys())
    
    total_nodes = list(main_module.graph.nodes)
    main_output_nodes.sort(key=lambda node:total_nodes.index(node))
    replace_modules = dict(replace_module.module.named_modules())
    
    partition_string = _get_partition_string(main_module, partition)
    partition_module = partition_module_map.setdefault(partition_string,{})
    
    if isinstance(replace_module.module,GraphModule):
        num_param_added = num_attr_added = num_module_added = 0
        node_mapping = node_mapping or {}
        for node in replace_module.module.graph.nodes:
            if node.op in ('placeholder','output'):
                continue
            args = []
            for arg in node.args:
                if isinstance(arg,fx.Node):
                    args.append(node_mapping.get(arg) )
                elif isinstance(arg,Iterable) and not isinstance(arg,str):
                    l = []
                    for a in arg:
                        if isinstance(a,fx.Node):
                            l.append(node_mapping.get(a))
                        else:
                            l.append(a)
                    args.append(l)
                else:
                    args.append(arg)
            args= tuple(args)
            preceeding_node = main_output_nodes[0] if len(main_output_nodes) else total_nodes[-1]
            with main_module.graph.inserting_before(preceeding_node):
                if node.op == 'call_function':
                    new_node = main_module.graph.call_function(node.target,args,node.kwargs)
                
                elif node.op == 'get_attr':
                    orig_attr = getattr(replace_module.module, node.target)
                    if isinstance(orig_attr, nn.Parameter):
                        key = f'_param_constant{num_param_added}'
                        is_key_new = key not in partition_module
                        attr_name = f'{key}_{no_of_module_replaced}'
                        attr_name, attr = partition_module.setdefault(key, (attr_name,orig_attr))
                        if is_key_new:
                            main_module.register_parameter(attr_name,copy.deepcopy(attr))
                            num_param_added += 1
                    else:
                        key = f'_attr_constant{num_attr_added}'
                        attr_name = f'{key}_{no_of_module_replaced}'
                        is_key_new = key not in partition_module
                        attr_name, attr = partition_module.setdefault(key, (attr_name,orig_attr))
                        if is_key_new:
                            if isinstance(attr, torch.Tensor):
                                main_module.register_buffer(attr_name,attr)
                            else:
                                setattr(main_module,attr_name,attr)
                            num_attr_added += 1
                    new_node = main_module.graph.get_attr(attr_name)
                
                elif node.op == 'call_module':
                    orig_module = copy.deepcopy(replace_modules[node.target])
                    source = partition.source
                    key = f'replaced_{source if isinstance(source,str) else source.__name__}_{num_module_added}'
                    is_key_new = key not in partition_module
                    module_name = f'{key}_{no_of_module_replaced}'
                    module_name, module = partition_module.setdefault(key,(module_name,orig_module))
                    if is_key_new:
                        main_module.add_module(module_name,module)
                        num_module_added += 1
                    new_node = main_module.graph.call_module(module_name,args,node.kwargs)
                
                node_mapping[node] = new_node
                adjust_meta(new_node,partition.output_nodes[0],node,partition)
        
        output_nodes= [node for node in replace_module.module.graph.nodes if node.op == 'output'][0].args[0]
        for i,output in enumerate(partition.output_nodes):
            output.replace_all_uses_with(node_mapping[output_nodes[i]]) 
    
    else:
        with main_module.graph.inserting_before(output_nodes[0]):
            replace_module_name = f'replaced_{replace_module.partition.source.__name__}_{no_of_module_replaced}'
            main_module.add_module(replace_module_name,replace_module)
            new_node = main_module.graph.call_module(replace_module_name,tuple(partition.input_nodes),{})
            partition.output_nodes[0].replace_all_uses_with(new_node)
    
    main_module.graph.lint()
    main_module.recompile()


# replaces all matches with call_module node
def _replace_all_matches(main_module:GraphModule, pattern_partitions:list[SourcePartition], replacements:list[tuple[nn.Module|type|types.FunctionType,types.FunctionType|types.NoneType]],aten_graph = False):
    '''
    replace all pattern partitions from the graph with a copy of replacement module
    if it gets None it will not replace that partition
    '''
    
    def default_module_gen_func(partiion:SourcePartition, main_model:GraphModule, aten_graph:bool = False):
        # assert isinstance(replace_module,nn.Module)
        return copy.deepcopy(replace_module)

    def default_input_adjustment_func(partition:SourcePartition, inputs:dict[fx.Node,Any]):
        inputs = [inputs[node] for node in partition.input_nodes]
        return inputs,{}
    
    global __net_module_replaced
    partition_module_map = {}
    for partition in pattern_partitions:
        partition_string = _get_partition_string(main_module, partition)
        is_new = partition_string not in partition_module_map
        source = partition.source
        for i,replacement in enumerate(replacements):
            replace_module, input_adjustment_func =replacement
            replace_module_copy = ReplacedModule(main_module, partition, 
                                                gen_func=default_module_gen_func if isinstance(replace_module,nn.Module) else replace_module,
                                                input_adjustment_func= input_adjustment_func or default_input_adjustment_func,
                                                aten_graph = aten_graph
                                                )
            if replace_module_copy.module is not None:
                break
        temp_partition_module_map = {}
        temp_net_module_replaced = 0
        for replacement in replacements[(i+1):]:
            replace_module, input_adjustment_func =replacement
            temp_partitions = utils.get_source_partition(replace_module_copy.module.graph,[source])[source]
            for temp_partition in temp_partitions:
                temp_partition_string = _get_partition_string(replace_module_copy.module, temp_partition)
                is_temp_new = temp_partition_string not in temp_partition_module_map
                temp_replace_module_copy = ReplacedModule(replace_module_copy.module, temp_partition, 
                                                    gen_func=default_module_gen_func if isinstance(replace_module,nn.Module) else replace_module,
                                                    input_adjustment_func= input_adjustment_func or default_input_adjustment_func,
                                                    aten_graph = aten_graph
                                                    )
                if temp_replace_module_copy.module is None:
                    continue
                _replace_pattern(replace_module_copy.module,temp_partition,temp_replace_module_copy,temp_net_module_replaced,temp_partition_module_map)
                if is_temp_new:
                    temp_net_module_replaced += 1
        if replace_module_copy.module is None:
            continue 
        _remove_hanging_nodes(replace_module_copy.module)
        # replace_module_copy = replace_module_copy.to(main_module)
            
        _replace_pattern(main_module, partition, replace_module_copy, __net_module_replaced,partition_module_map)
        if is_new:
            __net_module_replaced = __net_module_replaced + 1
    
    _remove_hanging_nodes(main_module)
    main_module.graph.lint()
    main_module.recompile()



# replace nodes if they don't need any change with their keyword arguments and arguements
def graph_pattern_replacer(main_module:GraphModule, pattern_partions:list[SourcePartition], replacement:tuple[nn.Module|type|types.FunctionType,types.FunctionType|types.NoneType], aten_graph = False, verbose_mode=False):
    '''
    replaces all matched partitions in the graph  with replacement module (wrapper call)
    '''
    # replacement = (replacement[0]() if isinstance(replacement[0], type) else replacement[0],replacement[1])
    global __net_module_replaced
    if __net_module_replaced is None:
        __net_module_replaced = 0
    _replace_all_matches(main_module, pattern_partions, replacement,aten_graph= aten_graph)
    if verbose_mode:
        print(type(pattern_partions).__name__, len(pattern_partions))
    
    return main_module
