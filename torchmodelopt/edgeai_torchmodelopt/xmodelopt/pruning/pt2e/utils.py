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


from copy import deepcopy
from curses.ascii import SO
from typing import Any, Iterable, List
from edgeai_torchmodelopt.xmodelopt import pruning

from numpy import partition
import torch
import torch.fx as fx
import torch.nn as nn
from torch import _dynamo as torch_dynamo
from torchvision import models as tvmodels
from timm import models as tmmodels
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions, SourcePartition
from torch.fx.passes.utils.matcher_utils import InternalMatch,SubgraphMatcher

import operator

_call_functions_to_look =[
    tvmodels.swin_transformer.shifted_window_attention,
    tvmodels.swin_transformer._get_relative_position_bias,
]


def remove_duplicates(items:list):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def _is_a_proper_input_node(node:fx.Node,model:fx.graph_module.GraphModule,old_results:dict[str:bool]={}):
    if node.name in old_results:
        return old_results[node.name]

    if node.op == 'placeholder':
        old_results[node.name] = True
        return True

    if node.op == 'call_function' and node.target == getattr:
        old_results[node.name] = False
        return False

    if node.op == 'get_attr':
        attr_names = node.target.split('.')
        attr = model
        for name in attr_names:
            attr = getattr(attr,name)
        old_results[node.name] = isinstance(attr,torch.nn.Parameter)
        return old_results[node.name]
    
    args = []
    for arg in node.args:
        if isinstance(arg,fx.Node):
            args.append(arg)
        elif isinstance(arg,Iterable) and not isinstance(arg,str):
            for a in arg:
                if isinstance(a,fx.Node):
                    args.append(a)
    args = remove_duplicates(args)

    old_results[node.name]=any([_is_a_proper_input_node(arg,model,old_results) for arg in args])
    return old_results[node.name]

def _get_proper_input_args(curr_partition:fx.Node|SourcePartition|InternalMatch,fx_model:fx.GraphModule,):
    args:List[fx.Node] = []
    if isinstance(curr_partition,fx.Node):
        node = curr_partition
        for arg in node.args:
            if isinstance(arg,fx.Node) and _is_a_proper_input_node(arg,fx_model):
                args.append(arg)
            elif isinstance(arg,Iterable) and not isinstance(arg,str):
                for a in arg:
                    if isinstance(a,fx.Node) and _is_a_proper_input_node(a,fx_model):
                        args.append(a)
    elif isinstance(curr_partition,SourcePartition):
        for arg in curr_partition.input_nodes:
            if _is_a_proper_input_node(arg,fx_model):
                args.append(arg)
    elif isinstance(curr_partition,InternalMatch):
        for arg in curr_partition.placeholder_nodes:
            if _is_a_proper_input_node(arg,fx_model):
                args.append(arg)
    args = remove_duplicates(args)
    return args


def get_pruning_partitions(module:fx.GraphModule):
    # some modules can contain other other module as submodule we have to go from larger modules to their submodule if they are required like Linear in MHA(Attention)
    #backup for get_source_partition 
    orig_module = deepcopy(module)
    orig_module, module = module, orig_module
    with module.graph.inserting_after():
        for node in module.graph.nodes:
            if node.op != 'output' and len(node.users) == 0:
                module.graph.erase_node(node)
    module.graph.lint(), module.recompile()
    other_patterns:list[tuple[Any,fx.GraphModule]] = []
    visited_node_names = []
    
    result:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]] = {}
    mod= tmmodels.vision_transformer.Attention(192,3,True)
    mod,_ = torch_dynamo.export(mod,aten_graph=True)(torch.ones(1,197,192))
    other_patterns.append((tmmodels.vision_transformer.Attention,mod))
    
    for cls,mod in (other_patterns):
        
        with mod.graph.inserting_after():
            for node in mod.graph.nodes:
                if len(node.users)  == 0 and node.op != 'output':
                    mod.graph.erase_node(node)
        mod.graph.lint(),mod.recompile()
        subgraph_matcher = SubgraphMatcher(mod.graph,ignore_literals=True)
        matches = subgraph_matcher.match(module.graph)
        unique_matches:List[InternalMatch] = []
        for match in matches:
            nodes = [node for p,node in match.nodes_map.items()]
            # if none of the match nodes is already visited it will be added
            if not any(node.name in visited_node_names for node in nodes):
                unique_matches.append(match)
                visited_node_names.extend([n.name for n in nodes])
            
        result[cls] = (mod,unique_matches)  
    
    nn_modules = [
        nn.MultiheadAttention,
        nn.Conv2d,
        nn.LayerNorm,
        nn.BatchNorm2d,
        nn.Linear,
    ]
    for layer in nn_modules:
        res = get_source_partitions(orig_module.graph,[layer])
        unique_matches = []
        if layer in res:
            matches = res[layer]
            for match in matches:
                nodes = match.nodes
                if not any(node.name in visited_node_names for node in nodes):
                    unique_matches.append(match)
                    visited_node_names.extend([n.name for n in nodes])
            result[layer] = unique_matches
    
    
    return result

# def get_pruning_class

def create_bn_conv_mapping(module:fx.GraphModule,pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]]):
    # need to check all the layers connected to conv2d and linear and we would have to include them all in the mapping list
    next_bn_partitions:dict[Any,SourcePartition] = dict()
    
    # the QAT module will already be merged and thus we would not have to calculate all this.
        
    if all(layer in pattern_partitions for layer in (nn.Conv2d,nn.BatchNorm2d)):
        batch_norm_partitions = pattern_partitions[nn.BatchNorm2d]
        conv_partitions= pattern_partitions[nn.Conv2d]
        for bn_match in batch_norm_partitions:
            input_node = bn_match.input_nodes[0]
            # to check if input node in any conv partition
            for conv_match in conv_partitions:
                if input_node in conv_match.nodes:
                    next_bn_partitions[conv_match.output_nodes[0].name] = bn_match
                    break

                    
    return next_bn_partitions

def find_in_node(module:fx.GraphModule,orig_partition:SourcePartition, curr_partition:SourcePartition|fx.Node, pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]], next_conv_node_list:dict[Any,List[SourcePartition]]):
    # recursive call to find the related conv layers to the orig conv layer in its users (below layers)
    if isinstance(curr_partition,fx.Node) and  curr_partition.op == 'output':
        return
    if nn.Conv2d in pattern_partitions:
        conv_partitions = pattern_partitions[nn.Conv2d]
    else:
        return
    
    if orig_partition not in conv_partitions:
        return
    

    # For transformers - may needed to be enhanced #TODO
    if isinstance(curr_partition,fx.Node) :
        if curr_partition.op == 'call_function' :
            if curr_partition.target == torch.ops.aten.transpose.int and  1 in list(curr_partition.args)[1:] :
                return
            if curr_partition.target == torch.ops.aten.permute.default and list(curr_partition.args)[-1]==1 :
                return

        for sub_node in curr_partition.users:
            found = False
            for conv_match in conv_partitions:
                if sub_node in conv_match.nodes:
                    find_in_node(module,orig_partition,conv_match,pattern_partitions,next_conv_node_list)
                    found = True
                    break
            if found : continue
            find_in_node(module, orig_partition, sub_node, pattern_partitions, next_conv_node_list)
            
    if isinstance(curr_partition ,SourcePartition) :
        params = dict(module.named_parameters())
        if curr_partition in conv_partitions:
            if curr_partition != orig_partition :
                if params[curr_partition.nodes[0].target].shape[1]== params[orig_partition.nodes[0].target].shape[0]:
                    next_conv_node_list[orig_partition.output_nodes[0].name].append(curr_partition)
                return

        for out in curr_partition.output_nodes:
            for user in out.users:
                found = False
                for conv_match in conv_partitions:
                    if user in conv_match.nodes and conv_match != curr_partition:
                        find_in_node(module,orig_partition,conv_match,pattern_partitions,next_conv_node_list)
                        break
                if found : continue
                find_in_node(module, orig_partition, user, pattern_partitions, next_conv_node_list)
    return

                        
def create_next_conv_node_list(module:fx.GraphModule,pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]]):
    # returns list of all nodes which are connected to the current node 
    next_conv_node_list:dict[Any,List[SourcePartition]] = dict()
    
    if nn.Conv2d in pattern_partitions:
        conv_partitions = pattern_partitions[nn.Conv2d]
    else:
        return next_conv_node_list
    for match in conv_partitions:
        next_conv_node_list[match.output_nodes[0].name] = []
        find_in_node(module,match,match,pattern_partitions,next_conv_node_list)
    return next_conv_node_list

def get_bn_adjusted_weight(module:fx.GraphModule,conv_partition:SourcePartition, next_bn_partitions:dict[Any,SourcePartition]):
    params = dict(module.named_parameters())
    if conv_partition.output_nodes[0].name in next_bn_partitions:
        bn_partition = next_bn_partitions[conv_partition.output_nodes[0].name]
        target =params[conv_partition.nodes[0].target].shape
        bn_weight = params[bn_partition.nodes[3].target].detach().clone()[:, None, None, None].expand(target)
        bn_var= torch.sqrt((getattr(module,bn_partition.nodes[5].target) + bn_partition.nodes[7].args[-1])[:, None, None, None].expand(target)) # need to see this line, what to detach and clone #TODO
        net_weight = torch.div(torch.mul(conv_partition.nodes[0].target, bn_weight), bn_var)
    else:
        net_weight = params[conv_partition.nodes[0].target]
    return net_weight


def remove_channels_conv_next(module:fx.GraphModule, conv_partition:SourcePartition, next_bn_partitions:dict[Any,SourcePartition], next_conv_node_list:dict[Any,List[SourcePartition]], nonzero_idx:torch.Tensor):
    # removes the output channels from current conv node, along with is BN, further removes all the input channels from the further connected conv nodes
    params = dict(module.named_parameters())
    params[conv_partition.nodes[0].target].data = params[conv_partition.nodes[0].target].data[nonzero_idx].contiguous()
    # nn.Parameter can also be used over here
    if len(conv_partition.nodes) == 3: # not going here for some reason, need to see why, no bias seen
        params[conv_partition.nodes[1].target].data = params[conv_partition.nodes[1].target].data[nonzero_idx].contiguous()
    # modules[node.target].out_channels = modules[node.target].weight.shape[0]
    if conv_partition.output_nodes[0].name in next_bn_partitions:
        bn_partition = next_bn_partitions[conv_partition.output_nodes[0].name]
        params[bn_partition.nodes[3].target].data =  params[bn_partition.nodes[3].target].data[nonzero_idx].contiguous()
        params[bn_partition.nodes[4].target].data =  params[bn_partition.nodes[4].target].data[nonzero_idx].contiguous()
        # modules[next_bn_partitions[node.target].target].track_running_stats = False
        running_mean = getattr(module,bn_partition.nodes[5].target)
        running_var = getattr(module,bn_partition.nodes[6].target)
        running_mean.data =  running_mean.data[nonzero_idx].contiguous()
        running_var.data =  running_var.data[nonzero_idx].contiguous()
        # modules[next_bn_partitions[node.target].target].num_features =  modules[node.target].weight.shape[0]
    
    for n_conv_partition in next_conv_node_list[conv_partition.output_nodes[0].name]:
        if params[n_conv_partition.nodes[0].target].shape[1]==1: #dwconv 
            # modules[n_id.target].in_channels = modules[node.target].weight.shape[0]
            # modules[n_id.target].groups = modules[node.target].weight.shape[0]
            continue
        if params[n_conv_partition.nodes[0].target].shape[1]!=nonzero_idx.shape[0]: 
            # the input channels have already been changed once, need not be changed again
            # however there could be some concat/etc and it needs to be accomodated #TODO
            continue
        params[n_conv_partition.nodes[0].target].data = params[n_conv_partition.nodes[0].target].data[:,nonzero_idx,:,:].contiguous()
        # modules[n_id.target].in_channels = params[n_conv_partition.nodes[0].target].shape[1]
        #BN parameters may not be removed, because we are just removing from input 
    return module 

def find_next_prunned_nodes(node:fx.Node,model:fx.GraphModule,old_result:dict=None):
    if isinstance(model,fx.GraphModule):
        fx_model = model
    else:
        fx_model = fx.symbolic_trace(model)
    
    if old_result is None:
        old_result = {}
    if node.name in old_result:
        return old_result[node.name]
    modules = dict(fx_model.named_modules())
    result = []
    for n_id in node.users:
        if n_id.op == 'output':
            continue
        elif n_id.op == 'call_function':
            if n_id.target == tvmodels.swin_transformer.shifted_window_attention:
                result.append(n_id)
        elif n_id.op == 'call_module':
            module = modules[n_id.target]
            if isinstance(module,(nn.Conv2d,nn.LayerNorm,nn.MultiheadAttention,nn.Linear)):
                #nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,
                result.append(n_id)
        else:
            result.extend(find_next_prunned_nodes(n_id,fx_model,old_result))
    old_result[node.name] = result 
    return result

def find_prev_pruned_nodes(node:fx.Node,model:fx.GraphModule,old_result:dict = {}):
    if isinstance(model,fx.GraphModule):
        fx_model = model
    else:
        fx_model = fx.symbolic_trace(model)
    
    if old_result is None:
        old_result = {}
    if node.name in old_result:
        return old_result[node.name]
    modules = dict(fx_model.named_modules())
    result = []
    args =_get_proper_input_args(node,fx_model)
    for n_id in args :
        if n_id.op == 'placeholder':
            continue
        if n_id.op == 'get_attr':
            continue
        elif n_id.op == 'call_function':
            if n_id.target == tvmodels.swin_transformer.shifted_window_attention:
                result.append(n_id)
        elif n_id.op == 'call_module':
            module = modules[n_id.target]
            if isinstance(module,(nn.Conv2d,nn.LayerNorm,nn.MultiheadAttention,nn.Linear)):
                #nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,
                result.append(n_id)
        else:
            result.extend(find_next_prunned_nodes(n_id,fx_model,old_result))
    old_result[node.name] = result 
    return result

def create_channel_pruned_model2(model):
    model.eval()
    # the QAT module will already be merged and thus we would not have to calculate all this.
    if isinstance(model, fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = model
    else:
        fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    model_graph = fx_model.graph

    next_bn_partitions = create_bn_conv_mapping(model)
    next_conv_node_list = create_next_conv_node_list(model)

    all_connected_nodes= find_all_connected_nodes(fx_model)
    #pruning dimension set up
    pruning_dim = {}
    for sublist in all_connected_nodes:
        for node,dim in sublist:
            pruning_dim[node.name] = dim
    
    for node in fx_model.graph.nodes:
        if node.name in pruning_dim:
            continue
        if node.op == 'call_module':
            module = modules[node.target]
            if isinstance(module,(nn.Conv2d,nn.LayerNorm,nn.MultiheadAttention,nn.Linear)):                
                #nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,
                pruning_dim[node.name] = 0

        elif node.op == 'call_function':
            if node.target == tvmodels.swin_transformer.shifted_window_attention:
                pruning_dim[node.name] = 0
        elif node.op == 'get_attr':
            if  any(f in [n.target for n in node.users] for f in _call_functions_to_look):
                    continue
            attr = fx_model
            for attr_name in node.target.split('.'):
                attr = getattr(attr,attr_name)
            if isinstance(attr,nn.Parameter):
                pruning_dim[node.name] = 0
    
    next_pruning_nodes_dict  = {}
    
    reshape_changes = {}
    for node in model_graph.nodes:
        if node.op  == 'call_method':
            if node.target == 'reshape':
                prev_nodes = find_prev_pruned_nodes(node,fx_model)
                if len(prev_nodes):
                    n_id = prev_nodes[0]
                    if n_id.op == 'call_function':
                        if n_id.target == tvmodels.swin_transformer.shifted_window_attention:
                            qkv_weight_node  = node.args[1]
                            proj_weight_node  = node.args[2]
                            qkv_module,_ =qkv_weight_node.target.rsplit('.',1)
                            qkv_module = modules[qkv_module]
                            proj_module,_ =proj_weight_node.target.rsplit('.',1)
                            proj_module = modules[proj_module]
                    elif n_id.op == 'call_module':
                        module = modules[n_id.target]
                        if isinstance(module,(nn.Conv2d,nn.Linear,nn.LayerNorm)):
                            module = module
                        elif isinstance(module,nn.MultiheadAttention):
                            module = module.out_proj
                        args = list(node.args)
                        for i,arg in enumerate(args):
                            if arg == module.weight.shape[0]:
                                reshape_changes[node.name] = (i,module) 
    
    for node in model_graph.nodes:
        if node.op in ('output','placeholder'):
            continue
        
        elif node.op  == 'call_method':
            if node.target == 'reshape':
                index,module = reshape_changes[node.name]
                args = list(node.args)
                args[index] = module.weight.shape[0]
                node.args = tuple(args)
                                
        nonzero_idx = None
        
        if node.op == 'get_attr':
            if  any(f in [n.target for n in node.users] for f in _call_functions_to_look):
                    continue
            attr = fx_model
            attr_names = node.target.split('.')
            module =modules['.'.join(attr_names[:-1])]
            for attr_name in attr_names:
                attr = getattr(attr,attr_name)
            if isinstance(attr,nn.Parameter):
                dim = pruning_dim.get(node.name,0)
                shape = list(range(len(attr.shape)))
                shape[0],shape[dim] = shape[dim],shape[0]
                attr = attr.permute(shape)
                nonzero_idx = ~(attr.view(attr.shape[0], -1).sum(dim=1) == 0)
                attr.data = attr.data[nonzero_idx].contiguous()
                attr =torch.nn.Parameter( attr.permute(shape))
                setattr(module,attr_names[-1],attr)
                
                
        elif node.args and node.op == 'call_module':
            module = modules[node.target]
            
            if isinstance(module, torch.nn.Conv2d):
                nonzero_idx = ~(module.weight.view(module.weight.shape[0], -1).sum(dim=1) == 0)
                modules = remove_channels_conv_next(modules, node, next_bn_partitions, next_conv_node_list, nonzero_idx) 
                
                # it is possible that next layer is dwconv and thus output and input would change accordingly, accomodate that
                for n_id in next_conv_node_list[node.target]:
                    if modules[n_id.target].weight.shape[1]==1:
                        modules = remove_channels_conv_next(modules, n_id, next_bn_partitions, next_conv_node_list, nonzero_idx)
                
            elif isinstance(module,nn.LayerNorm):
                nonzero_idx = ~(module.weight.view(module.weight.shape[0], -1).sum(dim=1) == 0)
                module.weight.data = module.weight.data[nonzero_idx].contiguous()
                if module.bias is not None:
                    module.bias.data = module.bias.data[nonzero_idx].contiguous()
                shape = list(module.normalized_shape)
                shape[0] = module.weight.shape[0]
                module.normalized_shape = tuple(shape)
            
            elif isinstance(module,nn.Linear) and 'output' not in [n.op for n in node.args]:
                nonzero_idx = ~(module.weight.view(module.weight.shape[0], -1).sum(dim=1) == 0)
                module.weight.data = module.weight.data[nonzero_idx].contiguous()
                if module.bias is not None:
                    module.bias.data = module.bias.data[nonzero_idx].contiguous()
                module.out_features = module.weight.shape[0]
            
            elif isinstance(module,nn.MultiheadAttention):
                embed_dim = module.embed_dim
                head_dim = module.head_dim
                num_heads = module.num_heads
                in_proj_weight = module.in_proj_weight

                in_proj_weight = in_proj_weight.reshape(3,num_heads,head_dim,in_proj_weight.shape[-1])
        
                in_proj_weight = in_proj_weight.permute(1,0,2,3)
                inproj_head_nonzero_idx = ~(in_proj_weight.reshape(in_proj_weight.shape[0], -1).sum(dim=1) == 0)
                module.num_heads = len(torch.where(inproj_head_nonzero_idx)[0])
                in_proj_weight = in_proj_weight.permute(2,1,0,3)
                inproj_channel_nonzero_idx = ~(in_proj_weight.reshape(in_proj_weight.shape[0], -1).sum(dim=1) == 0)
                in_proj_weight = in_proj_weight.permute(2,0,1,3)
                in_proj_weight = in_proj_weight.reshape(in_proj_weight.shape[0]*in_proj_weight.shape[1],in_proj_weight.shape[2],in_proj_weight.shape[3])
                inproj_head_channel_nonzero_idx = ~(in_proj_weight.reshape(in_proj_weight.shape[0], -1).sum(dim=1) == 0)
                module.head_dim = len(torch.where(inproj_channel_nonzero_idx)[0])
                module.embed_dim = module.head_dim*module.num_heads
                nonzero_idx = ~(module.in_proj_weight.reshape(module.in_proj_weight.shape[0], -1).sum(dim=1) == 0)
                module.in_proj_weight.data = module.in_proj_weight.data[nonzero_idx].contiguous()
                if module.in_proj_bias is not None:
                    module.in_proj_bias.data = module.in_proj_bias.data[nonzero_idx].contiguous()
                if module.out_proj.in_features == nonzero_idx.shape[0]:
                    module.out_proj.in_features= in_features
                
                module.out_proj.weight = torch.nn.Parameter(module.out_proj.weight[:,inproj_head_channel_nonzero_idx])
                module.embed_dim = module.out_proj.weight.shape[1]
                nonzero_idx = ~(module.out_proj.weight.view(module.out_proj.weight.shape[0], -1).sum(dim=1) == 0)
                module.out_proj.weight.data = module.out_proj.weight.data[nonzero_idx].contiguous()
                if module.out_proj.bias is not None:
                    module.out_proj.bias.data = module.out_proj.bias.data[nonzero_idx].contiguous()
                module.out_proj.out_features = module.out_proj.weight.shape[0]
        
        if nonzero_idx is not None:
            in_features = len(torch.where(nonzero_idx)[0])
            if node.name in list(next_pruning_nodes_dict.keys()):
                next_pruning_nodes = next_pruning_nodes_dict[node.name]
            else: 
                next_pruning_nodes = find_next_prunned_nodes(node,fx_model,next_pruning_nodes_dict)
            for n in next_pruning_nodes:
                if n.op == 'call_module':
                    module = modules[n.target]
                    if isinstance(module,nn.Conv2d):
                        if module.in_channels == nonzero_idx.shape[0]:
                            module.in_channels= in_features
                        if module.weight.shape[1] == nonzero_idx.shape[0]:
                            module.weight = torch.nn.Parameter(module.weight[:,nonzero_idx])
                        
                    # elif isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
                    #     pass
                    elif isinstance(module,nn.LayerNorm):
                        if module.weight.shape[1] == nonzero_idx.shape[0]:
                            module.weight = torch.nn.Parameter(module.weight[:,nonzero_idx])
                    elif isinstance(module,nn.Linear):
                        if module.in_features == nonzero_idx.shape[0]:
                            module.in_features= in_features
                        if module.weight.shape[1] == nonzero_idx.shape[0]:
                            module.weight = torch.nn.Parameter(module.weight[:,nonzero_idx])
                    elif isinstance(module,nn.MultiheadAttention):
                        if module.in_proj_weight.shape[1] == nonzero_idx.shape[0]:
                            module.in_proj_weight = torch.nn.Parameter(module.in_proj_weight[:,nonzero_idx])
                elif n.op == 'call_function':
                    if n.target == tvmodels.swin_transformer.shifted_window_attention:
                        pass
    fx_model.graph.lint()
    fx_model.recompile()                                      
    return fx_model

                
def create_channel_pruned_model(model):
    model.eval()
    # the QAT module will already be merged and thus we would not have to calculate all this.
    if isinstance(model, torch.fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = model
    else:
        fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    model_graph = fx_model.graph

    next_bn_partitions = create_bn_conv_mapping(model)
    next_conv_node_list = create_next_conv_node_list(model)
   
    # it will still fail in case of concat, similar, need to be thought of a way of doing that
    
    # we have to change the output of the current channel, as well as the input of the other connected channels.
    for node in model_graph.nodes:
        if node.target=='output':
            continue
        
        if node.args and isinstance(node.target, str):
            if isinstance(modules[node.target], torch.nn.Conv2d):
                # continue
                if modules[node.target].weight.shape[1]==1:
                    continue
                # rn we only prune the conv layers, along with the corresponding bn layer. Even the conv layer should not be depthwise that we prune
                nonzero_idx = ~(modules[node.target].weight.view(modules[node.target].weight.shape[0], -1).sum(dim=1) == 0)
                
                # deleting the zero_idx channels from the current's output as well as all the next conv's inputs
                modules = remove_channels_conv_next(modules, node, next_bn_partitions, next_conv_node_list, nonzero_idx) 
                
                # it is possible that next layer is dwconv and thus output and input would change accordingly, accomodate that
                for n_id in next_conv_node_list[node.target]:
                    if modules[n_id.target].weight.shape[1]==1:
                        modules = remove_channels_conv_next(modules, n_id, next_bn_partitions, next_conv_node_list, nonzero_idx)
                # we might even have to recurse through this, however would need a good exit statemnet #TODO   
                
                if len(next_conv_node_list[node.target])==0:
                    final_out_ch = nonzero_idx.shape[0]
                    final_nonzero_idx = nonzero_idx 
                
            if isinstance(modules[node.target], torch.nn.Linear):
                modules[node.target].weight = torch.nn.Parameter(modules[node.target].weight[:,final_nonzero_idx])
                modules[node.target].in_features = final_out_ch
                    
    return fx_model


# remove print statements #TODO
#TODO for InternalMatch thing
def find_layers_in_prev(node:fx.Node, connected_list:List[fx.Node|SourcePartition|InternalMatch],fx_model:fx.GraphModule,pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]],visited_nodes:list=None):
    # find all the connected nodes in the args(parents) of the current node, whenever a conv is found, we can stop our searching
    params = dict(fx_model.named_parameters())
    if visited_nodes is None:
        visited_nodes = list()
    if node.name in visited_nodes:
        return 
    temp_args = _get_proper_input_args(node,fx_model)

    all_get_attr_nodes_in_partitions = []
    for cls,partitions in pattern_partitions.items():
        for partition in partitions:
            nodes = []
            if isinstance(partition,SourcePartition): nodes.extend(partition.nodes)
            
            # if isinstance(partition,InternalMatch): nodes.extend(list(partition.nodes_map.values()))
            all_get_attr_nodes_in_partitions.extend([node.name for node in nodes if node.op == 'get_attr' ])
    args = [n for n in temp_args if n.op != 'get_attr']
    args.extend([n for n in temp_args if n.op == 'get_attr'])
    for n_id in args:
        # if isinstance(n_id, torch.fx.Node): # removing if the arg is not a node, but constants
        n_id
        if n_id.op == 'placeholder':
            visited_nodes.append(n_id.name)
        
        added=False
        if nn.Conv2d in pattern_partitions :
            found = False
            for conv_partition in pattern_partitions[nn.Conv2d]:
                if n_id in conv_partition.nodes:
                    found = True
                    break 
            if found:
            #for conv2d and depthwise conv
                connected_list.append(conv_partition)
                visited_nodes.append(conv_partition.output_nodes[0].name)
                added = True
                # if it is a depthwise layer, then we need to go even further into it
                if params[conv_partition.nodes[0].target].shape[1]==1:
                    find_layers_in_prev(conv_partition.output_nodes[0], connected_list,  fx_model,visited_nodes)
        if not added and nn.LayerNorm in pattern_partitions :
            found = False
            for ln_partition in pattern_partitions[nn.LayerNorm]:
                if n_id in ln_partition.nodes:
                    found = True
                    break 
            if found:
                connected_list.append(ln_partition)
                visited_nodes.append(ln_partition.nodes[2].name)
                added = True
        if not added and nn.BatchNorm2d in pattern_partitions :
            found = False
            for bn_partition in pattern_partitions[nn.BatchNorm2d]:
                if n_id in bn_partition.nodes:
                    found = True
                    break 
            if found:
            #for conv2d and depthwise conv
                connected_list.append(bn_partition)
                visited_nodes.append(bn_partition.nodes[7].name)
                added = True
        if not added and nn.Linear in pattern_partitions :
            found = False
            for fc_partition in pattern_partitions[nn.Linear]:
                if n_id in fc_partition.nodes:
                    found = True
                    break 
            if found:
            #for conv2d and depthwise conv
                connected_list.append(fc_partition)
                visited_nodes.append(fc_partition.nodes[4].name)
                added = True
        if not added and nn.MultiheadAttention in pattern_partitions :
            found = False
            for ln_partition in pattern_partitions[nn.MultiheadAttention]:
                if n_id in ln_partition.nodes:
                    found = True
                    break 
            if found:
            #for conv2d and depthwise conv
                connected_list.append(ln_partition)
                visited_nodes.append(ln_partition.nodes[2].name)
                added = True

        
        if not added and n_id.op == 'get_attr' and n_id.name not in all_get_attr_nodes_in_partitions:
            attr = getattr(fx_model,n_id.target)
            if isinstance(attr,torch.nn.Parameter):
                connected_list.append(n_id)
                visited_nodes.append(n_id.name)
        if not added:
            find_layers_in_prev(n_id,connected_list,fx_model,pattern_partitions,visited_nodes)
                    
    visited_nodes.append(node.name)
    return 

# TODO for other modules LayerNorm, MultiHeadAttention, Linear       
def find_all_connected_nodes(model:fx.GraphModule,pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]]):
    # returns the list of all conv that share the same output
    fx_model = model
    params = dict(fx_model.named_parameters())
    model_graph = fx_model.graph
    connected_list_prev = ['init']
    all_connected_list = []
    
    def extract_dims(connected_list_prev:list[fx.Node|str|SourcePartition|InternalMatch],pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]]):
        src_ptns = [match for match in connected_list_prev if isinstance(match,SourcePartition)]
        pruning_channels = None
        if src_ptns:
            src_ptn = src_ptns[0]
            if src_ptn.source == nn.Conv2d:
                weight = params[src_ptn.nodes[0].target]
            if src_ptn.source == nn.BatchNorm2d:
                weight = params[src_ptn.nodes[3].target]
            if src_ptn.source == nn.LayerNorm:
                weight = params[src_ptn.nodes[0].target]
            if src_ptn.source == nn.Linear:
                if len(src_ptn.nodes) == 4:
                    i= 0
                elif len(src_ptn.nodes) == 6:
                    i= 1
                elif len(src_ptn.nodes) == 8:
                    i= 3
                elif len(src_ptn.nodes) == 7:
                    i= 2
                weight = params[src_ptn.nodes[i].target]
            if src_ptn.source == nn.MultiheadAttention:
                if len(src_ptn.nodes) ==43:
                    i= 37
                elif len(src_ptn.nodes) == 52:
                    i = 46
                elif len(src_ptn.nodes) == 60:
                    i = 54
                weight = params[src_ptn.nodes[i].target]
            pruning_channels =weight.size(0)
        partitions = src_ptns
        for index,item in enumerate(connected_list_prev):
            if pruning_channels is None:
                connected_list_prev[index] = (item,0)
            else:
                if item in partitions :
                    connected_list_prev[index] = (item,0)
                else:
                    attr = fx_model
                    attr_names = item.target.split('.')
                    for name in attr_names:
                        attr = getattr(attr,name)
                    dim = list(attr.shape).index(pruning_channels)
                    connected_list_prev[index] = (item,dim)
        return connected_list_prev
    all_partition_nodes =[]
    for cls,partitions in pattern_partitions.items():
        for partition in partitions:
            nodes = []
            if isinstance(partition,SourcePartition): nodes.extend(partition.nodes)
            # if isinstance(partition,InternalMatch): nodes.extend(list(partition.nodes_map.values()))
            all_partition_nodes.extend([node.name for node in nodes])
    for node in model_graph.nodes:
        args = _get_proper_input_args(node,fx_model)
        # if (len(node.args)>1) and not(node.target in (torch.mul,operator.mul)): # to find layers like add, removing mul layer
        #     if all(isinstance(n_id, torch.fx.Node) for n_id in node.args): # to remove nodes like gemm, which has node, int as two inputs
                # this might be a problem for layers like concat, as they have axis as input and thus is a problem #TODO
                # also if there is concat layer, it wont affect us because any number of channels can be concat to anything #TODO
                # problem with mul layer as well
        is_mul = node.op == 'call_function' and node.target in (operator.mul,torch.mul,)
        if len(args)>1 and not is_mul and node.name not in all_partition_nodes:
            connected_list = []
            find_layers_in_prev(node, connected_list,  fx_model,pattern_partitions)
            if connected_list:
                if (connected_list_prev[-1] != connected_list[-1]) and (connected_list_prev[0] != connected_list[0]):
                    connected_list_prev = extract_dims(connected_list_prev,pattern_partitions)
                    # print(node,connected_list_prev)           
                    all_connected_list.append(connected_list_prev)
                connected_list_prev = connected_list
    connected_list_prev = extract_dims(connected_list_prev,pattern_partitions)
    # print(node,connected_list_prev)           
    all_connected_list.append(connected_list_prev)          
    return all_connected_list[1:]

def get_sum_of_all_conv_below(curr_node:fx.Node, module:fx.GraphModule,pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]], next_bn_partitions:dict[Any,SourcePartition], next_conv_node_list:dict[Any,List[SourcePartition]], node_weight_transposed:torch.Tensor, out_channels:int, ignore_node_name_list=[], global_pruning=False):
    if nn.Conv2d not in pattern_partitions:
        return None
    # gets the inner dimension concat of all the nodes conneected to the curr_node
    params = dict(module.named_parameters())
    for conv_match in next_conv_node_list[curr_node.name]:
        if conv_match.output_nodes[0].name not in ignore_node_name_list:
            is_depthwise_node = (params[conv_match.nodes[0].target].shape[1] == 1)
            wt_node = abs(get_bn_adjusted_weight(module,conv_match,next_bn_partitions).transpose(0,1).reshape(out_channels, -1).detach())
            wt_node = wt_node.mean(axis=1).unsqueeze if global_pruning else wt_node
            if is_depthwise_node:
                node_weight_transposed = torch.concat([node_weight_transposed, wt_node], axis=1)
                node_weight_transposed = torch.concat([node_weight_transposed, get_sum_of_all_conv_below(conv_match.output_nodes[0], module, pattern_partitions, next_bn_partitions, next_conv_node_list, node_weight_transposed, out_channels, ignore_node_name_list, global_pruning)], axis=1)
            else:
                node_weight_transposed = torch.concat([node_weight_transposed, wt_node], axis=1)
            ignore_node_name_list.append(conv_match.output_nodes.name)
    return node_weight_transposed.mean(axis=1).unsqueeze(1) if (global_pruning and node_weight_transposed.shape[1]) else node_weight_transposed

def search_for_first_conv(curr_node:fx.Node,module:fx.GraphModule,pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]]):
    if nn.Conv2d in pattern_partitions:
        conv_partitions = pattern_partitions[nn.Conv2d]
    else: 
        return None
    params = dict(module.named_parameters())
    for conv_partition in conv_partitions:
        if curr_node in conv_partition:
            if params[conv_partition.nodes[0].target].shape[1] ==1:
                return search_for_first_conv(conv_partition.input_nodes[0],module,pattern_partitions=pattern_partitions)
            else:
                return conv_partition
    args = _get_proper_input_args(curr_node,module)
    for node in args: 
        return search_for_first_conv(node, module,pattern_partitions)
    return None

def get_net_weight_node_channel_prune(curr_partition:fx.Node|SourcePartition|InternalMatch, module:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]], next_bn_partitions:dict[Any,SourcePartition], next_conv_node_list:dict[Any,List[SourcePartition]], ignore_node_name_list=[], global_pruning=False, net_weights = {}):
    params = dict(module.named_parameters())
    # outputs the net weight for a node, incorporates the sum of nodes of the below( inner dimensions )
    # if the node is depthwise, we should output the net_weight of the previous conv to this depthwise
    if isinstance(curr_partition,SourcePartition):
        if curr_partition.source == nn.Conv2d:
            is_depthwise_node = (params[curr_partition.nodes[0].target].shape[1] == 1)
            if is_depthwise_node:
                # search in args until we find a conv
                prev_conv_node  = search_for_first_conv(curr_partition.output_nodes[0], module,pattern_partitions)
                return net_weights[prev_conv_node.name][0].unsqueeze(1)
                
            else:
                out_channels = params[curr_partition.nodes[0].target].shape[0]
                net_weight = abs(get_bn_adjusted_weight(module,curr_partition,next_bn_partitions).reshape(out_channels, -1).detach())
                net_weight = net_weight.mean(dim=1).unsqueeze(1) if global_pruning else net_weight
                node_weight_transposed = torch.empty(out_channels,0).to(params[curr_partition.nodes[0].target].device)
                node_weight_transposed = get_sum_of_all_conv_below(curr_partition.output_nodes[0], module,pattern_partitions, next_bn_partitions, next_conv_node_list, node_weight_transposed, out_channels, ignore_node_name_list, global_pruning)
                # We cannot do global using these net weights because there could be multiple connections to one conv, whereas none for the next. Need some decay factor if we want to do global pruning # TODO
                net_weight = torch.concat([net_weight, node_weight_transposed], axis=1)
            # the net_weight over here is a two dim output, No * (Ni*w*h)
        elif curr_partition.source == nn.Linear:
            if len(curr_partition.nodes) == 4:
                i= 0
            elif len(curr_partition.nodes) == 6:
                i= 1
            elif len(curr_partition.nodes) == 8:
                i= 3
            elif len(curr_partition.nodes) == 7:
                i= 2
            net_weight = params[curr_partition.nodes[i].target]
        elif curr_partition.source == nn.LayerNorm:
            net_weight = params[curr_partition.nodes[0].target][:,None]
        elif curr_partition.source == nn.MultiheadAttention:
            if len(curr_partition.nodes) ==43:
                i= 37
            elif len(curr_partition.nodes) == 52:
                i = 46
            elif len(curr_partition.nodes) == 60:
                i = 54
            net_weight = params[curr_partition.nodes[i].target]
        elif curr_partition.source == nn.BatchNorm2d:
            net_weight = params[curr_partition.nodes[3].target][:,None]
        
    return net_weight.mean(dim=1).unsqueeze(1) if global_pruning else net_weight
    
def get_weight_from_parameter(node:fx.Node,fx_model:fx.GraphModule,global_pruning=False,dim:int=0):
    params = dict(fx_model.named_parameters())
    attr = params[node.target]
    if isinstance(attr,nn.Parameter):
        shape1 = list(range(len(attr.shape)))
        if dim != 0:
           shape1[0],shape1[dim] = shape1[dim],shape1[0]
           attr = attr.permute(shape1)
        attr = attr.reshape(attr.shape[0],-1)
        return attr.mean(1).unsqueeze(1) if global_pruning else attr
    
def get_net_weights_all(module:fx.GraphModule,pattern_partitions:dict[Any,List[SourcePartition]|tuple[fx.GraphModule,List[InternalMatch]]], next_conv_node_list:dict[Any,List[SourcePartition]], all_connected_nodes:List[List[fx.Node|SourcePartition|InternalMatch]], next_bn_partitions:dict[Any,SourcePartition], channel_pruning, global_pruning=False):
    fx_model = module
    all_modules = dict(fx_model.named_modules())
    params = dict(fx_model.named_parameters())
    model_graph = fx_model.graph
    
    net_weights = dict()
    
    
    all_connected_nodes_separated = []
    
    if all_connected_nodes is not None:
        for sublist in all_connected_nodes:
            all_connected_nodes_separated.extend([partition for partition,dim in sublist])
        for sublist in all_connected_nodes:
            ignore_node_name_list = []
            partition,dim = sublist[0]

            if  isinstance(partition,fx.Node) and partition.op == 'get_attr':
                attr = params[partition.target]
                weight_sublist =  torch.empty(attr.shape[dim],0).to(attr.device)
            elif isinstance(partition,SourcePartition):
                if partition.source == nn.Conv2d:
                    first_weight = params[partition.nodes[0].target]
                elif partition.source == nn.Linear:
                    if len(partition.nodes) == 4:
                        i= 0
                    elif len(partition.nodes) == 6:
                        i= 1
                    elif len(partition.nodes) == 8:
                        i= 3
                    elif len(partition.nodes) == 7:
                        i= 2
                    first_weight = params[partition.nodes[i].target]
                elif partition.source == nn.LayerNorm:
                    first_weight = params[partition.nodes[0].target]
                elif partition.source == nn.MultiheadAttention:
                    if len(partition.nodes) ==43:
                        i= 37
                    elif len(partition.nodes) == 52:
                        i = 46   
                    elif len(partition.nodes) == 60:
                        i = 54     
                    first_weight = params[partition.nodes[i].target]
                elif partition.source == nn.BatchNorm2d:
                    first_weight = params[partition.nodes[3].target]
                weight_sublist =  torch.empty(first_weight.shape[dim],0).to(first_weight.device)

            for partition,dim in sublist:
                if isinstance(partition,fx.Node) and partition.op == 'get_attr':
                    attr = params[partition.target]
                    weight_sublist = torch.concat([weight_sublist,get_weight_from_parameter(partition,fx_model,global_pruning,dim)],axis=1)
                if isinstance(partition,SourcePartition):
                        weight_sublist = torch.concat([weight_sublist, get_net_weight_node_channel_prune(partition, module,pattern_partitions, next_bn_partitions, next_conv_node_list, ignore_node_name_list, global_pruning, net_weights)], axis=1) 
            
            for partition,dim in sublist:
                if isinstance(partition,fx.Node) and partition.op == 'get_attr':
                    param_name = partition.target
                elif isinstance(partition,SourcePartition):
                    if partition.source == nn.Conv2d:
                        param_name = partition.nodes[0].target
                    elif partition.source == nn.Linear:
                        if len(partition.nodes) == 4:
                            i= 0
                        elif len(partition.nodes) == 6:
                            i= 1
                        elif len(partition.nodes) == 8:
                            i= 3
                        elif len(partition.nodes) == 7:
                            i= 2
                        param_name = partition.nodes[i].target
                    elif partition.source == nn.LayerNorm:
                        param_name = partition.nodes[0].target
                    elif partition.source == nn.MultiheadAttention:
                        if len(partition.nodes) ==43:
                            i= 37
                        elif len(partition.nodes) == 52:
                            i = 46        
                        elif len(partition.nodes) == 60:
                            i = 54     
                        param_name = partition.nodes[i].target
                    elif partition.source == nn.BatchNorm2d:
                        param_name = partition.nodes[3].target
                weight = params[param_name]
                while len(weight_sublist.shape)<len(weight.shape):
                    weight_sublist = weight_sublist.unsqueeze(-1)
                if dim != 0 :
                    shape= list(range(len(weight_sublist.shape)))
                    shape[0],shape[dim] = shape[dim],shape[0]    
                    weight_sublist = weight_sublist.permute(shape)
                net_weights[param_name] = ((weight_sublist.mean(dim=1) if global_pruning else weight_sublist),dim)
                if dim!=0:
                    weight_sublist = weight_sublist.permute(shape)
                weight_sublist = weight_sublist.flatten(1)
    
    all_partition_nodes =[]
    for cls,partitions in pattern_partitions.items():
        for partition in partitions:
            nodes = []
            if isinstance(partition,SourcePartition): nodes.extend(partition.nodes)
            # if isinstance(partition,InternalMatch): nodes.extend(list(partition.nodes_map.values()))
            all_partition_nodes.extend([node.name for node in nodes])                  
    
    for node in model_graph.nodes:
        if node.op=='get_attr': 
            if node.target in net_weights:
                continue
            attr = params[node.target]
            if node.name not in all_partition_nodes:
                net_weights[node.target] = ((weight_sublist.mean(dim=1) if global_pruning else weight_sublist),0)
    
    for cls,partitions in pattern_partitions.items():
        if cls == nn.MultiheadAttention:
            for partition in  partitions:               
                if len(partition.nodes) ==43:
                    i = 2
                    j = 37
                elif len(partition.nodes) == 52:
                    i = 1
                    j = 46        
                elif len(partition.nodes) == 60:
                    i = 1
                    j = 54    
                param_name = partition.nodes[i].target
                net_weights[param_name] = (params[param_name],0)
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[j].target
                    net_weights[param_name] = (params[param_name],0)
        elif cls == nn.LayerNorm:
            for partition in  partitions:               
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[0].target
                    net_weights[param_name] = (params[param_name],0)
        elif cls == nn.BatchNorm2d:
            for partition in  partitions:               
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[3].target
                    net_weights[param_name] = (params[param_name],0)
        elif cls == nn.Linear:
            for partition in  partitions:
                if len(partition.nodes) == 4:
                    i= 0               
                elif len(partition.nodes) == 6:
                    i= 1
                elif len(partition.nodes) == 8:
                    i= 3
                elif len(partition.nodes) == 7:
                    i= 2
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[i].target
                    net_weights[param_name] = (params[param_name],0)
        elif cls == nn.Conv2d:
            for partition in  partitions:               
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[0].target
                    net_weights[param_name] = (params[param_name],0)
                    
    return net_weights
