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

import operator
from typing import  Type, List, Dict, Any, Iterable
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.utils.source_matcher_utils import  SourcePartition

try:
    from timm import models as tmmodels
    has_timm = True
except:
    has_timm = False

try:
    from torchvision import models as tvmodels
    has_tv = True
except:
    has_tv = False

try:
    from transformers import models as hfmodels
    has_hf = True
except:
    has_hf = False
    

def remove_duplicates(items:list):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def _is_a_proper_input_node(node:fx.Node, model:fx.graph_module.GraphModule, old_results:dict[str:bool]={}):
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

def _get_proper_input_args(curr_partition:fx.Node|SourcePartition, fx_model:fx.GraphModule,):
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
    args = remove_duplicates(args)
    return args


# Note: The source code is copied from pytorch github (https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/utils/source_matcher_utils.py#L51)
# and modified as per requirement 
def get_source_partitions(graph:fx.Graph, wanted_sources:list, filter_fn = None):
    '''
    a custom made get_source_partitions that can handle any type of modules and functions that are wrapped for fx
    
    Note: This function is also defined on surgery.v3.utils. If this is modified later on, same changes have to be made on that function definition 
    
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



def get_pruning_partitions(module:fx.GraphModule):
    # some modules can contain other other module as submodule we have to go from larger modules to their submodule if they are required like Linear in MHA(Attention)

    
    wanted_sources = [
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.LayerNorm,
        nn.Linear,
        nn.MultiheadAttention,
    ]
    if has_tv:
        wanted_sources.extend([
            tvmodels.swin_transformer.SwinTransformer,
            tvmodels.swin_transformer.shifted_window_attention,
        ])
    
    if has_timm:
        wanted_sources.extend([
            tmmodels.vision_transformer.Attention,
            tmmodels.swin_transformer.WindowAttention,
        ])
    if has_hf:
        wanted_sources.extend([
            hfmodels.vit.modeling_vit.ViTAttention,
            hfmodels.swin.modeling_swin.SwinAttention,
        ])
    
    return get_source_partitions(module.graph,wanted_sources)

def get_parameter_indices(fx_model:fx.GraphModule, source:type, partition:SourcePartition):
    if source == nn.Conv2d:
        weight_index = 0
        bias_index = 1 
    elif source == nn.BatchNorm2d:
        weight_index = 3
        bias_index = 4
    elif source == nn.Linear:
        if len(partition.nodes) == 4:
            weight_index = 0
            bias_index = 2
        elif len(partition.nodes) == 6:
            weight_index = 1
            bias_index = 3
        elif len(partition.nodes) == 8:
            weight_index = 3
            bias_index = 5
        elif len(partition.nodes) == 7:
            weight_index = 2
            bias_index = 4
    elif source == nn.LayerNorm:
         weight_index = 0
         bias_index = 1
    elif source == nn.MultiheadAttention:
        # for both weights and both biases (inner projection and outer projection)
        if len(partition.nodes) ==43:
            weight_index = [2,37]
            bias_index = [4,39]
        elif len(partition.nodes) == 52:
            weight_index = [1,46]  
            bias_index = [8,48]
        elif len(partition.nodes) == 60:
            weight_index = [1,54]
            bias_index = [8,56]
    return weight_index, bias_index

def get_num_heads_head_dims(fx_model:fx.GraphModule, source:type, source_partition:SourcePartition):
    # TODO proper implementation for all attention layers with their variant
    attn_layers = [nn.MultiheadAttention]

    if has_tv:
        attn_layers.extend([
            tvmodels.swin_transformer.ShiftedWindowAttention,
            tvmodels.swin_transformer.shifted_window_attention,
        ])
    if has_timm:
        attn_layers.extend([
            tmmodels.vision_transformer.Attention,
            tmmodels.swin_transformer.WindowAttention
        ])
    if has_hf:
        attn_layers.extend([
            hfmodels.vit.modeling_vit.ViTAttention,
            hfmodels.swin.modeling_swin.SwinAttention,
        ])
    if source not in attn_layers:
        raise Exception(f'This is only for attention layer of transformer models so expected any of\n {",".join(attn_layers)}\n as source but got {source}')
    
    FINAL_EXCEPTION = Exception(f'Still not Supported for {source}')
    if source == nn.MultiheadAttention:
        if len(source_partition.nodes) ==  52:
            node_index = 30
        if len(source_partition.nodes) == 43:
            node_index = 23
        elif len(source_partition.nodes) == 60:
            node_index = 28     
                
        shape = source_partition.nodes[node_index].args[1]
        num_heads,head_dims =  shape[1],shape[3]
    else: 
        raise FINAL_EXCEPTION
            
    return num_heads,head_dims
# def get_pruning_class

def create_bn_conv_mapping(module:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]]):
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

def find_in_node(module:fx.GraphModule, orig_partition:SourcePartition, curr_partition:SourcePartition|fx.Node, pattern_partitions:dict[Any,List[SourcePartition]], next_conv_node_list:dict[Any,List[SourcePartition]]):
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

                        
def create_next_conv_node_list(module:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]]):
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

def get_bn_adjusted_weight(module:fx.GraphModule, conv_partition:SourcePartition, next_bn_partitions:dict[Any,SourcePartition]):
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
        if len(n_conv_partition.nodes) == 3: # not going here for some reason, need to see why, no bias seen
            params[n_conv_partition.nodes[1].target].data = params[n_conv_partition.nodes[1].target].data[nonzero_idx].contiguous()
        # modules[n_id.target].in_channels = params[n_conv_partition.nodes[0].target].shape[1]
        #BN parameters may not be removed, because we are just removing from input 
    return module 

def find_next_prunned_partitions(node:fx.Node, model:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]], old_result:dict[str,list[tuple[type,SourcePartition]]]=None):
    fx_model = model
    if old_result is None:
        old_result = {}
    if node.name in old_result:
        return old_result[node.name]
    result:list[tuple[type,SourcePartition]] = []
    for n_id in node.users:
        if n_id.op == 'output':
            continue
        found = False
        for cls,partitions in pattern_partitions.items():
            if isinstance(partitions,tuple):
                continue
            for partition in partitions:
                if n_id in partition.nodes:
                    result.append((partition.source,partition))
                    found = True
                    break
            if found:
                break
        if not found:
            result.extend(find_next_prunned_partitions(n_id,model,pattern_partitions,old_result) )
    old_result[node.name] = result 
    return result

def find_prev_pruned_partitions(node:fx.Node, model:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]], old_result:dict[str,list[tuple[type,SourcePartition]]] = None):
    
    if old_result is None:
        old_result = {}
    if node.name in old_result:
        return old_result[node.name]
    result:list[tuple[type,SourcePartition]] = []
    args =_get_proper_input_args(node,model)
    for n_id in args :
        if n_id.op == 'placeholder':
            continue
        if n_id.op == 'get_attr':
            continue
        elif n_id.op == 'output':
            continue
        found = False
        for cls,partitions in pattern_partitions.items():
            for partition in partitions:
                if n_id in partition.nodes:
                    found = True
                    result.append((cls,partition))
                    break
            if found :
                break
        if not found:
            result.extend(find_prev_pruned_partitions(n_id,model,pattern_partitions,old_result))
    old_result[node.name] = result 
    return result

def create_channel_pruned_model(model:fx.GraphModule):
    model.eval()
    # the QAT module will already be merged and thus we would not have to calculate all this.
    model = model
    params = dict(model.named_parameters())
    pattern_partitions = get_pruning_partitions(model)

    next_bn_partitions = create_bn_conv_mapping(model,pattern_partitions)
    next_conv_node_list = create_next_conv_node_list(model,pattern_partitions)

    all_connected_nodes= find_all_connected_nodes(model,pattern_partitions)
    net_weights = get_net_weights_all(model,pattern_partitions,next_conv_node_list,all_connected_nodes,next_bn_partitions,True)
    #pruning dimension set up
    pruning_dim = {}
    for node_target in net_weights:
        net_weight, dim = net_weights[node_target] 
        pruning_dim[node_target] = dim
    
    all_partition_nodes =[] 
    for cls,partitions in pattern_partitions.items():
        for partition in partitions:
            all_partition_nodes.extend([node.name for node in partition.nodes])
    
    
    def adjust_weight_of_next_partitions(node:fx.Node, nonzero_idx:torch.Tensor):
        next_pruned_nodes = find_next_prunned_partitions(node,model,pattern_partitions,)
        for cls,partition in next_pruned_nodes:
            if cls in (nn.LayerNorm,nn.BatchNorm2d):
                continue
            weight_index, bias_index = get_parameter_indices(model,cls,partition)
            if cls == nn.MultiheadAttention:
                weight_index = weight_index[0]
            param_name = partition.nodes[weight_index].target
            if params[param_name].shape[1] != nonzero_idx.shape[0]:
                continue
            params[param_name] = torch.nn.Parameter(params[param_name][:,nonzero_idx])            
            setattr(model,param_name,params[param_name])
            
    
    for node in model.graph.nodes:
        if node.name in all_partition_nodes:
            continue
        if node.target  not in net_weights:
            continue 
        elif node.op == 'get_attr':
            if node.target in params:
                param_name = node.target
                attr = params[param_name]
                dim = pruning_dim[node.target]
                shape = list(range(len(attr.shape)))
                shape[0],shape[dim] =shape[dim],shape[0]
                if dim != 0:
                    attr =attr.permute(shape) 
                nonzero_idx = ~(attr.view(attr.shape[0], -1).sum(dim=1) == 0)
                attr.data = attr.data[nonzero_idx].contiguous()
                if dim != 0:
                    attr =attr.permute(shape) 
                attr =torch.nn.Parameter( attr)
                setattr(model,param_name,attr)
                adjust_weight_of_next_partitions(node,nonzero_idx)
    
    reshape_changes = {}
    for node in model.graph.nodes:
        if node.name in all_partition_nodes: 
            continue
        if node.op  == 'call_function' and node.target == torch.ops.aten.view.default:
            prev_nodes = find_prev_pruned_partitions(node,model,pattern_partitions)
            if len(prev_nodes) == 0:
                continue
            cls,partition = prev_nodes[0]
            weight_index, bias_index = get_parameter_indices(model,cls,partition)
            if cls == nn.MultiheadAttention:
                weight_index, bias_index = weight_index[1], bias_index[1]
            param_name = partition.nodes[weight_index].target
            dim =pruning_dim [param_name]
            pruning_channels = params[param_name].shape[dim]
            for i,a in enumerate(node.args[1]):
                if a == pruning_channels:
                    reshape_changes[node.name] = (i,param_name,dim)        
    new_num_heads_head_dims :dict[str,tuple[int,int]] = {}         
    
    def find_prev_pruned_partitions_for_ln(node:fx.Node, model:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]], old_result:dict[str,list[tuple[type,SourcePartition]]] = None):
        if old_result is None:
            old_result = {}
        if node.name in old_result:
            return old_result[node.name]
        result:list[tuple[type,SourcePartition]] = []
        args =_get_proper_input_args(node,model)
        for n_id in args :
            if n_id.op == 'placeholder':
                continue
            if n_id.op == 'get_attr':
                continue
            elif n_id.op == 'output':
                continue
            found = False
            for cls,partitions in pattern_partitions.items():
                if  isinstance(partitions,tuple):
                    continue
                for partition in partitions:
                    if n_id in partition.nodes:
                        if cls in (nn.LayerNorm,nn.BatchNorm2d):
                            for inp in partition.input_nodes:
                                inp_result = find_prev_pruned_partitions_for_ln(inp,model,pattern_partitions,old_result)
                                found = len(inp_result) != 0
                                result.extend(inp_result)
                                if found:
                                    break
                            if found:
                                break
                        else:
                            found = True
                            result.append((cls,partition))
                            break
                if found :
                    break
            if not found:
                result.extend(find_prev_pruned_partitions(n_id,model,pattern_partitions,old_result))
        old_result[node.name] = result 
        return result
        
    nonzero_idxs = {} 
                
    for cls,partitions in pattern_partitions.items():
        if cls == nn.Conv2d:
            for partition in partitions:
                weight_index, bias_index =   get_parameter_indices(model,cls,partition)
                
                param_name = partition.nodes[weight_index].target
                nonzero_idx = ~(params[param_name].view(params[param_name].shape[0], -1).sum(dim=1) == 0)
                params[param_name].data = params[param_name].data[nonzero_idx].contiguous()
                setattr(model,param_name,params[param_name])
                # nn.Parameter can also be used over here
                if len(partition.nodes) == 3: # not going here for some reason, need to see why, no bias seen
                    param_name = partition.nodes[bias_index].target
                    params[param_name].data = params[param_name].data[nonzero_idx].contiguous()                
                    setattr(model,param_name,params[param_name])
                for node in partition.output_nodes:
                    adjust_weight_of_next_partitions(node,nonzero_idx)
                nonzero_idxs[partition.nodes[0].name] = nonzero_idx
                
        elif cls == nn.Linear:
            for partition in partitions:
                if any( 'output' in [n.op for n in out.users]for out in partition.output_nodes):
                    continue
                
                weight_index, bias_index =   get_parameter_indices(model,cls,partition)
                param_name = partition.nodes[weight_index].target
                nonzero_idx = ~(params[param_name].view(params[param_name].shape[0], -1).sum(dim=1) == 0)
                params[param_name].data =  params[param_name].data[nonzero_idx].contiguous()
                setattr(model,param_name,params[param_name])
                param_name = partition.nodes[bias_index].target
                params[param_name].data =  params[param_name].data[nonzero_idx].contiguous()
                setattr(model,param_name,params[param_name])
                for node in partition.output_nodes:
                    adjust_weight_of_next_partitions(node,nonzero_idx)
                nonzero_idxs[partition.nodes[0].name] = nonzero_idx
                
        elif cls == nn.MultiheadAttention:
            for partition in partitions:
                old_num_heads,old_head_dims = get_num_heads_head_dims(model,cls,partition)
                weight_indices, bias_indices =   get_parameter_indices(model,cls,partition)
                param_name = partition.nodes[weight_indices[0]].target
                
                in_proj_weight = params[param_name]
                in_proj_weight = in_proj_weight.reshape(3,old_num_heads,old_head_dims,in_proj_weight.shape[-1])
                
                in_proj_weight = in_proj_weight.permute(1,0,2,3)
                in_proj_head_nonzero_idx =  ~(in_proj_weight.reshape(in_proj_weight.shape[0], -1).sum(dim=1) == 0)
                in_proj_weight = in_proj_weight.permute(1,0,2,3)
                
                in_proj_weight = in_proj_weight.permute(2,1,0,3)
                in_proj_channel_nonzero_idx =  ~(in_proj_weight.reshape(in_proj_weight.shape[0], -1).sum(dim=1) == 0)
                in_proj_weight = in_proj_weight.permute(2,1,0,3)
                
                in_proj_weight = in_proj_weight.reshape(3,old_num_heads*old_head_dims,in_proj_weight.shape[-1])
                in_proj_weight = in_proj_weight.permute(1,0,2)
                in_proj_head_channel_nonzero_idx = ~(in_proj_weight.reshape(in_proj_weight.shape[0], -1).sum(dim=1) == 0)
                
                nonzero_idx1 = ~(params[param_name].view(params[param_name].shape[0], -1).sum(dim=1) == 0)
                params[param_name].data =  params[param_name].data[nonzero_idx1].contiguous()
                setattr(model,param_name,params[param_name])
                param_name = partition.nodes[bias_indices[0]].target
                params[param_name].data =  params[param_name].data[nonzero_idx1].contiguous()
                setattr(model,param_name,params[param_name])
                
                new_head_dim = len(torch.where(in_proj_channel_nonzero_idx)[0])
                new_num_heads = len(torch.where(in_proj_head_nonzero_idx)[0])
                
                param_name = partition.nodes[weight_indices[1]].target
                out_proj_weight =  params[param_name]
                out_proj_weight = out_proj_weight.permute(1,0)
                out_proj_weight.data = out_proj_weight.data[in_proj_head_channel_nonzero_idx].contiguous()
                out_proj_weight =out_proj_weight.permute(1,0)
                params[param_name].data = out_proj_weight.data
                setattr(model,param_name,params[param_name])
                
                nonzero_idx2 = ~(params[param_name].view(params[param_name].shape[0], -1).sum(dim=1) == 0)
                params[param_name].data =  params[param_name].data[nonzero_idx2].contiguous()
                setattr(model,param_name,params[param_name])
                param_name = partition.nodes[bias_indices[1]].target
                params[param_name].data =  params[param_name].data[nonzero_idx2].contiguous()
                setattr(model,param_name,params[param_name])
                new_num_heads_head_dims[partition.nodes[0].name] = (new_num_heads,new_head_dim)
                for node in partition.output_nodes:
                    adjust_weight_of_next_partitions(node,nonzero_idx)
                nonzero_idxs[partition.nodes[0].name] = nonzero_idx
    
    for cls, partitions in pattern_partitions.items():
        if cls == nn.BatchNorm2d:
            for partition in partitions:
                prev_pruned_partition = find_prev_pruned_partitions_for_ln(partition.input_nodes[0],model,pattern_partitions)
                nonzero_idx = None
                if len(prev_pruned_partition):
                    prev_cls,prev_partition = prev_pruned_partition[0]
                    nonzero_idx = nonzero_idxs[partition.nodes[0].name]

                weight_index, bias_index =   get_parameter_indices(model,cls,partition)
                param_name = partition.nodes[weight_index].target
                nonzero_idx = ~(params[param_name].view(params[param_name].shape[0], -1).sum(dim=1) == 0)if nonzero_idx is None else nonzero_idx
                params[param_name].data =  params[param_name].data[nonzero_idx].contiguous()
                setattr(model,param_name,params[param_name])
                param_name = partition.nodes[bias_index].target
                params[param_name].data =  params[param_name].data[nonzero_idx].contiguous()
                setattr(model,param_name,params[param_name])
                # modules[next_bn_partitions[node.target].target].track_running_stats = False
                running_mean = getattr(model,partition.nodes[5].target)
                running_var = getattr(model,partition.nodes[6].target)
                running_mean.data =  running_mean.data[nonzero_idx].contiguous()
                running_var.data =  running_var.data[nonzero_idx].contiguous()
                for node in partition.output_nodes:
                    adjust_weight_of_next_partitions(node,nonzero_idx)
                
        elif cls == nn.LayerNorm:
            for partition in partitions:
                prev_pruned_partition = find_prev_pruned_partitions_for_ln(partition.input_nodes[0],model,pattern_partitions)
                nonzero_idx = None
                if len(prev_pruned_partition):
                    prev_cls,prev_partition = prev_pruned_partition[0]
                    nonzero_idx = nonzero_idxs[prev_partition.nodes[0].name]

                weight_index, bias_index =   get_parameter_indices(model,cls,partition)
                param_name = partition.nodes[weight_index].target
                nonzero_idx =  ~(params[param_name].view(params[param_name].shape[0], -1).sum(dim=1) == 0)if nonzero_idx is None else nonzero_idx
                params[param_name].data =  params[param_name].data[nonzero_idx].contiguous()
                setattr(model,param_name,params[param_name])
                param_name = partition.nodes[bias_index].target
                params[param_name].data =  params[param_name].data[nonzero_idx].contiguous()
                setattr(model,param_name,params[param_name])
                for node in partition.output_nodes:
                    adjust_weight_of_next_partitions(node,nonzero_idx)
    
    def change_argument(node:fx.Node, index:int, value):
        expected_nodes = [torch.ops.aten.native_layer_norm.default,torch.ops.aten.expand.default,torch.ops.aten.view.default,torch.ops.aten._unsafe_view.default,operator.mul]
        if node.target in (torch.ops.aten.view.default,torch.ops.aten.expand.default,torch.ops.aten._unsafe_view.default):
            args = list(node.args)
            args[1] = list(args[1])
            args[1][index] = value
            node.args = args
        elif node.target in (operator.mul,torch.ops.aten.native_layer_norm.default):
            args = list(node.args)
            args[index] = value
            node.args = tuple(args)
        else: 
            raise Exception(f'got node with {node.target} expected {expected_nodes}')
            
    for node in model.graph.nodes:
        if node.name in all_partition_nodes: 
            continue
        if node.op  == 'call_function' and node.target == torch.ops.aten.view.default:
            index,param_name,dim = reshape_changes[node.name]
            change_argument(node,index, params[param_name].shape[dim])
    
    for cls,partitions in pattern_partitions.items():
        if cls in (nn.Conv2d,nn.BatchNorm2d):
            continue
        if cls == nn.LayerNorm:
            for partition in partitions:
                weight_index, bias_index = get_parameter_indices(model,cls,node)
                weight = params[partition.nodes[weight_index].target]
                new_normalized_shape = [weight.shape[0]]
                change_argument(partition.nodes[2],1,new_normalized_shape)
        if cls ==  nn.Linear:
            for partition in partitions:
                if len(partition.nodes) == 4:
                    continue
                elif len(partition.nodes )== 6:
                    v1,v2 = 0,5
                elif len(partition.nodes )== 7:
                    v1,v2 = 1,6
                elif len(partition.nodes )== 8:
                    v1,v2 = 2,7
                else:
                    continue
                weight_index, bias_index = get_parameter_indices(model,cls,partition)
                param_name = partition.nodes[weight_index].target
                weight = params[param_name]
                out_features, in_features = list(weight.shape)
                change_argument(partition.nodes[v1],1,in_features)
                change_argument(partition.nodes[v2],2,out_features)
        if cls == nn.MultiheadAttention:
            for partition in partitions:
                new_num_heads,new_head_dim = new_num_heads_head_dims[partition.nodes[0].name]
                weight_indices, bias_indices = get_parameter_indices(model,cls,partition)
                in_proj_weight = params[partition.nodes[weight_indices[0]].target]
                in_proj_out_features,in_proj_in_features =list(in_proj_weight.shape)
                out_proj_weight = params[partition.nodes[weight_indices[1]].target]
                out_proj_out_features,out_proj_in_features =list(out_proj_weight.shape)
                value_change_dict={
                    'new_head_dim':new_head_dim,   
                    'new_num_heads':new_num_heads,
                    'in_proj_out_features':in_proj_out_features, 
                    'in_proj_in_features':in_proj_in_features, 
                    'out_proj_out_features':out_proj_out_features, 
                    'out_proj_in_features':out_proj_in_features, 
                }
                # TODO For all other partition with different length
                # changing argument where change is needed 
                if len(partition.nodes) == 60:
                    indices_vals_dict:dict[int,list[tuple[int,str]]] = {
                        5:[(1,'in_proj_in_features')],
                        7:[(2,'in_proj_out_features')],
                        11:[(3,'out_proj_in_features')],
                        19:[(1,'new_num_heads')],
                        20:[(2,'new_head_dim')],
                        22:[(1,'new_num_heads')],
                        23:[(2,'new_head_dim')],
                        25:[(1,'new_num_heads')],
                        26:[(2,'new_head_dim')],
                        28:[(1,'new_num_heads'),(3,'new_head_dim')],
                        29:[(1,'new_num_heads'),(3,'new_head_dim')],
                        30:[(1,'new_num_heads'),(3,'new_head_dim')],
                        34:[(1,'new_num_heads'),(3,'new_head_dim')],
                        35:[(1,'new_num_heads')],
                        36:[(2,'new_head_dim')],
                        37:[(1,'new_num_heads'),(2,'new_head_dim')],
                        38:[(1,'new_head_dim')],
                        40:[(1,'new_num_heads')],
                        43:[(1,'new_num_heads')],
                        44:[(1,'new_num_heads')],
                        46:[(1,'new_num_heads'),(3,'new_head_dim')],
                        47:[(2,'new_head_dim')],
                        49:[(1,'new_num_heads'),(3,'new_head_dim')],
                        53:[(1,'out_proj_in_features')],
                        58:[(2,'out_proj_out_features')],
                    }
                else:
                    continue
                for i, detail_list in indices_vals_dict.items():
                    node = partition.nodes[i]
                    for index,value in detail_list:
                        value = value_change_dict[value]
                        change_argument(node,index,value)
        
    model.graph.lint()
    model.recompile()                                      
    return model

                

# remove print statements #TODO
def find_layers_in_prev(node:fx.Node, connected_list:List[fx.Node|SourcePartition], fx_model:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]], visited_nodes:list=None):
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
            all_get_attr_nodes_in_partitions.extend([node.name for node in partition.nodes if node.op == 'get_attr' ])
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
                # TODO about the bn other than after convs
                # connected_list.append(bn_partition)
                visited_nodes.append(bn_partition.nodes[7].name)
                # added = True
                find_layers_in_prev(partition.input_nodes[0],connected_list,fx_model,pattern_partitions,visited_nodes)
        if not added and nn.Linear in pattern_partitions :
            found = False
            for fc_partition in pattern_partitions[nn.Linear]:
                if n_id in fc_partition.nodes:
                    found = True
                    break 
            if found:
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
def find_all_connected_nodes(model:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]]):
    # returns the list of all conv that share the same output
    fx_model = model
    params = dict(fx_model.named_parameters())
    model_graph = fx_model.graph
    connected_list_prev = ['init']
    all_connected_list = []
    
    def extract_dims(connected_list_prev:list[fx.Node|str|SourcePartition],pattern_partitions:dict[Any,List[SourcePartition]]):
        src_ptns = [match for match in connected_list_prev if isinstance(match,SourcePartition)]
        pruning_channels = None
        if src_ptns:
            src_ptn = src_ptns[0]
            weight_index, bias_index = get_parameter_indices(model,src_ptn.source,src_ptn)
            if src_ptn.source == nn.MultiheadAttention:
                weight_index,bias_index = weight_index[1],bias_index[1]
            weight= params[src_ptn.nodes[weight_index].target]
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
            all_partition_nodes.extend([node.name for node in partition.nodes])
    
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

def get_sum_of_all_conv_below(curr_node:fx.Node, module:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]], next_bn_partitions:dict[Any,SourcePartition], next_conv_node_list:dict[Any,List[SourcePartition]], node_weight_transposed:torch.Tensor, out_channels:int, ignore_node_name_list=[], global_pruning=False):
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

def search_for_first_conv(curr_node:fx.Node, module:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]]):
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

def get_net_weight_node_channel_prune(curr_partition:fx.Node|SourcePartition, module:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]], next_bn_partitions:dict[Any,SourcePartition], next_conv_node_list:dict[Any,List[SourcePartition]], ignore_node_name_list=[], global_pruning=False, net_weights = {}):
    params = dict(module.named_parameters())
    # outputs the net weight for a node, incorporates the sum of nodes of the below( inner dimensions )
    # if the node is depthwise, we should output the net_weight of the previous conv to this depthwise
    if isinstance(curr_partition,SourcePartition):
        weight_index, bias_index = get_parameter_indices(module,curr_partition.source,curr_partition)
        if curr_partition.source == nn.Conv2d:
            
            is_depthwise_node = (params[curr_partition.nodes[weight_index].target].shape[1] == 1)
            if is_depthwise_node:
                # search in args until we find a conv
                prev_conv_node  = search_for_first_conv(curr_partition.output_nodes[0], module,pattern_partitions)
                return net_weights[prev_conv_node.name][0].unsqueeze(1)
                
            else:
                out_channels = params[curr_partition.nodes[weight_index].target].shape[0]
                net_weight = abs(get_bn_adjusted_weight(module,curr_partition,next_bn_partitions).reshape(out_channels, -1).detach())
                net_weight = net_weight.mean(dim=1).unsqueeze(1) if global_pruning else net_weight
                node_weight_transposed = torch.empty(out_channels,0).to(params[curr_partition.nodes[weight_index].target].device)
                node_weight_transposed = get_sum_of_all_conv_below(curr_partition.output_nodes[0], module,pattern_partitions, next_bn_partitions, next_conv_node_list, node_weight_transposed, out_channels, ignore_node_name_list, global_pruning)
                # We cannot do global using these net weights because there could be multiple connections to one conv, whereas none for the next. Need some decay factor if we want to do global pruning # TODO
                net_weight = torch.concat([net_weight, node_weight_transposed], axis=1)
            # the net_weight over here is a two dim output, No * (Ni*w*h)
        elif curr_partition.source == nn.Linear:
            net_weight = params[curr_partition.nodes[weight_index].target]
        elif curr_partition.source == nn.LayerNorm:
            net_weight = params[curr_partition.nodes[weight_index].target][:,None]
        elif curr_partition.source == nn.MultiheadAttention:
            weight_index =weight_index[1]
            net_weight = params[curr_partition.nodes[weight_index].target]
        elif curr_partition.source == nn.BatchNorm2d:
            net_weight = params[curr_partition.nodes[weight_index].target][:,None]
        
    return net_weight.mean(dim=1).unsqueeze(1) if global_pruning else net_weight
    
def get_weight_from_parameter(node:fx.Node, fx_model:fx.GraphModule, global_pruning=False, dim:int=0):
    params = dict(fx_model.named_parameters())
    attr = params[node.target]
    if isinstance(attr,nn.Parameter):
        shape1 = list(range(len(attr.shape)))
        if dim != 0:
           shape1[0],shape1[dim] = shape1[dim],shape1[0]
           attr = attr.permute(shape1)
        attr = attr.reshape(attr.shape[0],-1)
        return attr.mean(1).unsqueeze(1) if global_pruning else attr
    
def get_net_weights_all(module:fx.GraphModule, pattern_partitions:dict[Any,List[SourcePartition]], next_conv_node_list:dict[Any,List[SourcePartition]], all_connected_nodes:List[List[fx.Node|SourcePartition]], next_bn_partitions:dict[Any,SourcePartition], channel_pruning, global_pruning=False):
    fx_model = module
    params = dict(fx_model.named_parameters())
    model_graph = fx_model.graph
    
    net_weights = dict()
    
    
    all_connected_nodes_separated = []
    
    net_weights_added = []
    
    def adjust_and_store_net_weight(weight_sublist:torch.Tensor,param_name:str,dim:int):
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
                
                weight_index, bias_index = get_parameter_indices(module,partition. source, partition)
                if partition.source == nn.MultiheadAttention:
                    weight_index, bias_index = weight_index[1], bias_index[1]
                first_weight = params[partition.nodes[weight_index].target]
                weight_sublist =  torch.empty(first_weight.shape[dim],0).to(first_weight.device)

            for partition,dim in sublist:
                if isinstance(partition,fx.Node) and partition.op == 'get_attr':
                    attr = params[partition.target]
                    weight_sublist = torch.concat([weight_sublist,get_weight_from_parameter(partition,fx_model,global_pruning,dim)],axis=1)
                if isinstance(partition,SourcePartition):
                        weight_sublist = torch.concat([weight_sublist, get_net_weight_node_channel_prune(partition, module,pattern_partitions, next_bn_partitions, next_conv_node_list, ignore_node_name_list, global_pruning, net_weights)], axis=1) 
            
            for partition,dim in sublist:
                if partition in net_weights_added:
                    continue
                if isinstance(partition,fx.Node) and partition.op == 'get_attr':
                    param_name = partition.target
                elif isinstance(partition,SourcePartition):
                    weight_index, bias_index = get_parameter_indices(module,partition. source, partition)
                    if partition.source == nn.MultiheadAttention:
                        weight_index, bias_index = weight_index[1], bias_index[1]
                    param_name = partition.nodes[weight_index].target
                adjust_and_store_net_weight(weight_sublist,param_name,dim)
                net_weights_added.append(partition)
                
                if isinstance( partition,SourcePartition ) and partition.source == nn.Conv2d:

                    #  for next batchnorm
                    bn_partition = next_bn_partitions.get(partition.output_nodes[0].name,None)
                    if bn_partition:
                        weight_index, bias_index = get_parameter_indices(module,bn_partition.source,bn_partition)
                        param_name = partition.nodes[weight_index].target
                        adjust_and_store_net_weight(weight_sublist,param_name,0)
                        net_weights_added.append(bn_partition)

                    next_conv_nodes = next_conv_node_list[partition.output_nodes[0].name]
                    # for depthwise convs
                    for conv_partition in next_conv_nodes:
                        weight_index, bias_index = get_parameter_indices(module, conv_partition.source, conv_partition)
                        param_name = partition.nodes[weight_index].target
                        adjust_and_store_net_weight(weight_sublist, param_name, 0)
                        net_weights_added.append(conv_partition)
                        
                        #  for next batchnorm
                        bn_partition = next_bn_partitions.get(conv_partition.output_nodes[0].name,None)
                        if bn_partition:
                            weight_index, bias_index = get_parameter_indices(module, bn_partition.source, partition=bn_partition)
                            param_name = partition.nodes[weight_index].target
                            adjust_and_store_net_weight(weight_sublist, param_name, 0)
                            net_weights_added.append(bn_partition)
                            
    all_partition_nodes =[]
    for cls,partitions in pattern_partitions.items():
        for partition in partitions:
            all_partition_nodes.extend([node.name for node in partition.nodes])                  
    
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
                weight_indices, bias_indices = get_parameter_indices(module,partition. source, partition)
                weight_index1,weight_index2 = weight_indices[0],weight_indices[1]   
                param_name = partition.nodes[weight_index1].target
                net_weights[param_name] = (params[param_name], 0)
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[weight_index2].target
                    net_weights[param_name] = (params[param_name], 0)
        elif cls == nn.LayerNorm:
            for partition in  partitions:               
                weight_index, bias_index = get_parameter_indices(module,partition. source, partition)
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[weight_index].target
                    net_weights[param_name] = (params[param_name], 0)
        elif cls == nn.BatchNorm2d:
            for partition in  partitions:               
                weight_index, bias_index = get_parameter_indices(module,partition. source, partition)
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[weight_index].target
                    net_weights[param_name] = (params[param_name], 0)
        elif cls == nn.Linear:
            for partition in  partitions:
                weight_index, bias_index = get_parameter_indices(module,partition. source, partition)
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[weight_index].target
                    net_weights[param_name] = (params[param_name], 0)
        elif cls == nn.Conv2d:
            for partition in  partitions: 
                weight_index, bias_index = get_parameter_indices(module,partition. source, partition)              
                if partition not in all_connected_nodes_separated:
                    param_name = partition.nodes[weight_index].target
                    net_weights[param_name] = (params[param_name], 0)
                    
    return net_weights

# Note: This part is copied to surgery any changes made here must be copied there
from torch.onnx import symbolic_helper, register_custom_op_symbolic
def register_custom_ops_for_onnx(opset_version):
    def aten_unsafe_view(g, x, dim, *args):
        output = g.op("Reshape", x, dim)
        return output
    register_custom_op_symbolic(
        symbolic_name='aten::_unsafe_view', 
        symbolic_fn=aten_unsafe_view, 
        opset_version=opset_version)
    
    def aten_softmax(g, x,  *args):
        output = g.op("Softmax", x)
        return output
    register_custom_op_symbolic(
        symbolic_name='aten::_softmax', 
        symbolic_fn=aten_softmax, 
        opset_version=opset_version)