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


from typing import Iterable
import operator
import torch
import torch.fx as fx
import torch.nn as nn
from torchvision import models as tvmodels


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

def _get_proper_input_args(node:fx.Node,fx_model:fx.GraphModule):
    args = []
    for arg in node.args:
        if isinstance(arg,fx.Node) and _is_a_proper_input_node(arg,fx_model):
            args.append(arg)
        elif isinstance(arg,Iterable) and not isinstance(arg,str):
            for a in arg:
                if isinstance(a,fx.Node) and _is_a_proper_input_node(a,fx_model):
                    args.append(a)
    args = remove_duplicates(args)
    return args

def find_in_node(orig_node, curr_node, modules, next_conv_node_list):
    # recursive call to find the related conv layers to the orig conv layer in its users (below layers)
    condn = (curr_node.target!='output') and isinstance(curr_node.target, str) and (curr_node.target in modules) and isinstance(modules[curr_node.target], torch.nn.Conv2d)
    if condn and (curr_node.target in modules) and curr_node!=orig_node:
        if modules[curr_node.target].in_channels == modules[orig_node.target].out_channels:
            next_conv_node_list[orig_node.target].append(curr_node)
        return
    elif curr_node.target=='output':
        return 
    # For transformers - may needed to be enhanced #TODO
    elif (curr_node.op == 'call_method' and curr_node.target == 'permute' and list(curr_node.args)[-1]==1) \
      or (curr_node.op == 'call_method' and curr_node.target == 'transpose' and 1 in list(curr_node.args)[1:]) \
      or (curr_node.op == 'call_function' and curr_node.target == torch.permute and curr_node.args[1][-1] == 1)\
      or (curr_node.op == 'call_function' and curr_node.target == torch.transpose and 1 in list(curr_node.args)[1:]):
          return
    else:
        for sub_node in curr_node.users:
            find_in_node(orig_node, sub_node, modules, next_conv_node_list)
        return

def create_bn_conv_mapping(module):
    # need to check all the layers connected to conv2d and linear and we would have to include them all in the mapping list
    next_bn_nodes = dict()
    if isinstance(module, torch.fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = module
    else:
        fx_model = fx.symbolic_trace(module)
    
    # the QAT module will already be merged and thus we would not have to calculate all this.
        
    modules = dict(fx_model.named_modules())
    model_graph = fx_model.graph
    for node in model_graph.nodes:
        if node.target=='output':
            continue
        if node.args and node.op == 'call_module':
            if isinstance(modules[node.target], torch.nn.BatchNorm2d):
                if len(node.args[0].users)>1 or (node.args[0].target not in modules) : # dont merge if conv has multiple users or there is no connected conv
                    continue
                if isinstance(modules[node.args[0].target], torch.nn.Conv2d):
                    next_bn_nodes[node.args[0].target] = node
                    
    return next_bn_nodes
                        
def create_next_conv_node_list(module):
    # returns list of all nodes which are connected to the current node 
    next_conv_node_list = dict()
    if isinstance(module, torch.fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = module
    else:
        fx_model = fx.symbolic_trace(module)
    modules = dict(fx_model.named_modules())
    model_graph = fx_model.graph
    for node in model_graph.nodes:
        if node.target=='output':
            continue
        if node.args and isinstance(node.target, str) and (node.target in modules):
            if isinstance(modules[node.target], torch.nn.Conv2d):
                # there could be some layers like concat/etc which can also change the layer 
                # rn we do a sanity check if the channel numbers are not the same as expected,we skip adding that layer to the list, but needs to be made better #TODO (examples: concat layer)
                next_conv_node_list[node.target] = []
                find_in_node(node, node, modules, next_conv_node_list)
    return next_conv_node_list

def get_bn_adjusted_weight(node, modules, next_bn_nodes):
    module1 = modules[node.target]
    if node.target in next_bn_nodes:
        module2 = modules[next_bn_nodes[node.target].target]
        target = module1.weight.shape
        bn_weight = module2.weight.detach().clone()[:, None, None, None].expand(target)
        bn_var = torch.sqrt((module2.running_var + module2.eps)[:, None, None, None].expand(target)) # need to see this line, what to detach and clone #TODO
        net_weight = torch.div(torch.mul(module1.weight, bn_weight), bn_var)
    else:
        net_weight = module1.weight
    return net_weight


def remove_channels_conv_next(modules, node, next_bn_nodes, next_conv_node_list, nonzero_idx):
    # removes the output channels from current conv node, along with is BN, further removes all the input channels from the further connected conv nodes

    modules[node.target].weight.data = modules[node.target].weight.data[nonzero_idx].contiguous()
    # nn.Parameter can also be used over here
    if modules[node.target].bias is not None: # not going here for some reason, need to see why, no bias seen
        modules[node.target].bias.data = modules[node.target].bias.data[nonzero_idx].contiguous()
    modules[node.target].out_channels = modules[node.target].weight.shape[0]
    if node.target in next_bn_nodes:
        # modules[next_bn_nodes[node.target].target].track_running_stats = False
        modules[next_bn_nodes[node.target].target].running_mean.data =  modules[next_bn_nodes[node.target].target].running_mean.data[nonzero_idx].contiguous()
        modules[next_bn_nodes[node.target].target].running_var.data =  modules[next_bn_nodes[node.target].target].running_var.data[nonzero_idx].contiguous()
        modules[next_bn_nodes[node.target].target].weight.data =  modules[next_bn_nodes[node.target].target].weight.data[nonzero_idx].contiguous()
        modules[next_bn_nodes[node.target].target].bias.data =  modules[next_bn_nodes[node.target].target].bias.data[nonzero_idx].contiguous()
        modules[next_bn_nodes[node.target].target].num_features =  modules[node.target].weight.shape[0]
    
    for n_id in next_conv_node_list[node.target]:
        if modules[n_id.target].weight.shape[1]==1: #dwconv 
            modules[n_id.target].in_channels = modules[node.target].weight.shape[0]
            modules[n_id.target].groups = modules[node.target].weight.shape[0]
            continue
        if modules[n_id.target].weight.shape[1]!=nonzero_idx.shape[0]: 
            # the input channels have already been changed once, need not be changed again
            # however there could be some concat/etc and it needs to be accomodated #TODO
            continue
        modules[n_id.target].weight.data = modules[n_id.target].weight.data[:,nonzero_idx,:,:].contiguous()
        modules[n_id.target].in_channels = modules[n_id.target].weight.shape[1]
        #BN parameters may not be removed, because we are just removing from input 
    return modules 

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

    next_bn_nodes = create_bn_conv_mapping(model)
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
                modules = remove_channels_conv_next(modules, node, next_bn_nodes, next_conv_node_list, nonzero_idx) 
                
                # it is possible that next layer is dwconv and thus output and input would change accordingly, accomodate that
                for n_id in next_conv_node_list[node.target]:
                    if modules[n_id.target].weight.shape[1]==1:
                        modules = remove_channels_conv_next(modules, n_id, next_bn_nodes, next_conv_node_list, nonzero_idx)
                
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

    next_bn_nodes = create_bn_conv_mapping(model)
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
                modules = remove_channels_conv_next(modules, node, next_bn_nodes, next_conv_node_list, nonzero_idx) 
                
                # it is possible that next layer is dwconv and thus output and input would change accordingly, accomodate that
                for n_id in next_conv_node_list[node.target]:
                    if modules[n_id.target].weight.shape[1]==1:
                        modules = remove_channels_conv_next(modules, n_id, next_bn_nodes, next_conv_node_list, nonzero_idx)
                # we might even have to recurse through this, however would need a good exit statemnet #TODO   
                
                if len(next_conv_node_list[node.target])==0:
                    final_out_ch = nonzero_idx.shape[0]
                    final_nonzero_idx = nonzero_idx 
                
            if isinstance(modules[node.target], torch.nn.Linear):
                modules[node.target].weight = torch.nn.Parameter(modules[node.target].weight[:,final_nonzero_idx])
                modules[node.target].in_features = final_out_ch
                    
    return fx_model


# remove print statements #TODO
def find_layers_in_prev(node:fx.Node, connected_list:list,fx_model:fx.graph_module.GraphModule,visited_nodes:list=None):
    # find all the connected nodes in the args(parents) of the current node, whenever a conv is found, we can stop our searching
    modules = dict(fx_model.named_modules())
    if visited_nodes is None:
        visited_nodes = list()
    if node.name in visited_nodes:
        return 
    temp_args = _get_proper_input_args(node,fx_model)

    args = [n for n in temp_args if n.op != 'get_attr']
    args.extend([n for n in temp_args if n.op == 'get_attr'])
    for n_id in args:
        # if isinstance(n_id, torch.fx.Node): # removing if the arg is not a node, but constants
        n_id
        if n_id.args and n_id.op == 'call_module': 
            module = modules[n_id.target]
            
            #for conv2d and depthwise conv
            if isinstance(module, torch.nn.Conv2d):
                connected_list.append(n_id)
                visited_nodes.append(n_id.name)
                # if it is a depthwise layer, then we need to go even further into it
                if modules[n_id.target].weight.shape[1]==1:
                    find_layers_in_prev(n_id, connected_list,  fx_model,visited_nodes)

            # other layers from we have to take weights
            elif isinstance(module,(nn.LayerNorm,nn.MultiheadAttention,nn.Linear)):
                connected_list.append(n_id)
                visited_nodes.append(n_id.name)
            # otherwise proceed
            else:
                find_layers_in_prev(n_id,connected_list,fx_model,visited_nodes)
        elif n_id.op == 'call_function':
            if n_id.target == tvmodels.swin_transformer.shifted_window_attention:
                connected_list.append(n_id)
                visited_nodes.append(n_id.name)
            else:
                find_layers_in_prev(n_id,connected_list,fx_model,visited_nodes)
        
        elif n_id.op == 'placeholder':
            visited_nodes.append(n_id.name)
        elif n_id.op == 'get_attr':
            attr = fx_model
            for attr_name in n_id.target.split('.'):
                attr = getattr(attr,attr_name)
            if isinstance(attr,nn.Parameter):
                connected_list.append(n_id)
            visited_nodes.append(n_id.name)
        else:
            find_layers_in_prev(n_id, connected_list,  fx_model,visited_nodes) 
                    
    visited_nodes.append(node.name)
    return 

# TODO for other modules LayerNorm, MultiHeadAttention, Linear       
def find_all_connected_nodes(model):
    # returns the list of all conv that share the same output
    if isinstance(model, torch.fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = model
    else:
        fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    params = dict(fx_model.named_parameters())
    model_graph = fx_model.graph
    connected_list_prev = ['init']
    all_connected_list = []
    
    def extract_dims(connected_list_prev:list):
        module_nodes = [n for n in connected_list_prev if isinstance(n,fx.Node) and n.op == 'call_module']
        module_nodes.extend([ n for n in connected_list_prev if isinstance(n,fx.Node) and n.target == tvmodels.swin_transformer.shifted_window_attention])
        if module_nodes:
            if module_nodes[0].op == 'call_module':
                first_module = modules[module_nodes[0].target]
                if isinstance(first_module,nn.MultiheadAttention):
                    weight = first_module.out_proj.weight
                else:
                    weight = first_module.weight
            elif module_nodes[0].op == 'call_function':
                proj_weight_node = module_nodes[0].args[2]
                weight = params[proj_weight_node.target]
            pruning_channels =weight.size(0)
        else:
            pruning_channels = None
        for index,item in enumerate(connected_list_prev):
            if pruning_channels is None:
                connected_list_prev[index] = (item,0)
            else:
                if item in module_nodes:
                    connected_list_prev[index] = (item,0)
                else:
                    attr = fx_model
                    attr_names = item.target.split('.')
                    for name in attr_names:
                        attr = getattr(attr,name)
                    dim = list(attr.shape).index(pruning_channels)
                    connected_list_prev[index] = (item,dim)
        return connected_list_prev
    
    for node in model_graph.nodes:
        args = _get_proper_input_args(node,fx_model)
        # if (len(node.args)>1) and not(node.target in (torch.mul,operator.mul)): # to find layers like add, removing mul layer
        #     if all(isinstance(n_id, torch.fx.Node) for n_id in node.args): # to remove nodes like gemm, which has node, int as two inputs
                # this might be a problem for layers like concat, as they have axis as input and thus is a problem #TODO
                # also if there is concat layer, it wont affect us because any number of channels can be concat to anything #TODO
                # problem with mul layer as well
        is_mul = (node.op == 'call_function' and node.target in (torch.mul,operator.mul)) or (node.op == 'call_method' and node.target == 'mul')
        if len(args)>1 and not is_mul and node.target not in (tvmodels.swin_transformer.shifted_window_attention,):
            connected_list = []
            find_layers_in_prev(node, connected_list,  fx_model)
            if connected_list:
                if (connected_list_prev[-1] != connected_list[-1]) and (connected_list_prev[0] != connected_list[0]):
                    connected_list_prev = extract_dims(connected_list_prev)
                    # print(node,connected_list_prev)           
                    all_connected_list.append(connected_list_prev)
                connected_list_prev = connected_list
    connected_list_prev = extract_dims(connected_list_prev)
    # print(node,connected_list_prev)           
    all_connected_list.append(connected_list_prev)          
    return all_connected_list[1:]

def get_sum_of_all_conv_below(curr_node, all_modules, next_bn_nodes, next_conv_node_list, node_weight_transposed, out_channels, ignore_node_target_list=[], global_pruning=False):
    # gets the inner dimension concat of all the nodes conneected to the curr_node
    for node in next_conv_node_list[curr_node.target]:
        if node.target not in ignore_node_target_list:
            is_depthwise_node = (all_modules[node.target].weight.shape[1] == 1)
            wt_node = abs(get_bn_adjusted_weight(node=node, modules=all_modules, next_bn_nodes=next_bn_nodes).transpose(0,1).reshape(out_channels, -1).detach())
            wt_node = wt_node.mean(axis=1).unsqueeze(1) if global_pruning else wt_node
            if is_depthwise_node:
                node_weight_transposed = torch.concat([node_weight_transposed, wt_node], axis=1)
                node_weight_transposed = torch.concat([node_weight_transposed, get_sum_of_all_conv_below(node, all_modules, next_bn_nodes, next_conv_node_list, node_weight_transposed, out_channels, ignore_node_target_list, global_pruning)], axis=1)
            else:
                node_weight_transposed = torch.concat([node_weight_transposed, wt_node], axis=1)
            ignore_node_target_list.append(node.target)
    return node_weight_transposed.mean(axis=1).unsqueeze(1) if (global_pruning and node_weight_transposed.shape[1]) else node_weight_transposed

def search_for_first_conv(curr_node, all_modules):
    for node in curr_node.args:
        if node.args and isinstance(node.target, str) and (node.target in all_modules):
            if isinstance(all_modules[node.target], nn.Conv2d):
                is_depthwise_node = (all_modules[node.target].weight.shape[1] == 1)
                if not(is_depthwise_node):
                    return node
        return search_for_first_conv(node, all_modules)

def get_net_weight_node_channel_prune(curr_node, all_modules, next_bn_nodes, next_conv_node_list, ignore_node_target_list=[], global_pruning=False, net_weights = {}):
    module = all_modules[curr_node.target]
    # outputs the net weight for a node, incorporates the sum of nodes of the below( inner dimensions )
    # if the node is depthwise, we should output the net_weight of the previous conv to this depthwise
    if isinstance(module,nn.Conv2d):
        is_depthwise_node = (module.weight.shape[1] == 1)
        if is_depthwise_node:
            # search in args until we find a conv
            prev_conv_node  = search_for_first_conv(curr_node, all_modules)
            return net_weights[prev_conv_node.name][0].unsqueeze(1)
            
        else:
            out_channels = module.weight.shape[0]
            net_weight = abs(get_bn_adjusted_weight(node=curr_node, modules=all_modules, next_bn_nodes=next_bn_nodes).reshape(out_channels, -1).detach())
            net_weight = net_weight.mean(dim=1).unsqueeze(1) if global_pruning else net_weight
            node_weight_transposed = torch.empty(out_channels,0).to(module.weight.device)
            node_weight_transposed = get_sum_of_all_conv_below(curr_node, all_modules, next_bn_nodes, next_conv_node_list, node_weight_transposed, out_channels, ignore_node_target_list, global_pruning)
            # We cannot do global using these net weights because there could be multiple connections to one conv, whereas none for the next. Need some decay factor if we want to do global pruning # TODO
            net_weight = torch.concat([net_weight, node_weight_transposed], axis=1)
        # the net_weight over here is a two dim output, No * (Ni*w*h)
    elif isinstance(module,(nn.Linear)):
        net_weight = module.weight
    elif isinstance(module,nn.LayerNorm):
        net_weight = module.weight[:,None]
    elif isinstance(module,nn.MultiheadAttention):
        net_weight = module.out_proj.weight
    return net_weight.mean(dim=1).unsqueeze(1) if global_pruning else net_weight
    
def get_weight_from_parameter(node:fx.Node,fx_model:fx.GraphModule,global_pruning=False,dim:int=0):
    attr = fx_model
    for attr_name in node.target.split('.'):
        attr = getattr(attr,attr_name)
    if isinstance(attr,nn.Parameter):
        shape1 = list(range(len(attr.shape)))
        if dim != 0:
           shape1[0],shape1[dim] = shape1[dim],shape1[0]
           attr = attr.permute(shape1)
        attr = attr.reshape(attr.shape[0],-1)
        return attr.mean(1).unsqueeze(1) if global_pruning else attr
    
def get_net_weights_all(model, next_conv_node_list, all_connected_nodes, next_bn_nodes, channel_pruning, global_pruning=False):
    
    if isinstance(model, fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = model
    else:
        fx_model = fx.symbolic_trace(model)
        
    all_modules = dict(fx_model.named_modules())
    params = dict(fx_model.named_parameters())
    model_graph = fx_model.graph
    
    net_weights = dict()
    
    if all_connected_nodes is not None:
        for sublist in all_connected_nodes:
            ignore_node_target_list = []
            node,dim = sublist[0]
            if node.op == 'get_attr':
                attr = fx_model
                for attr_name in node.target.split('.'):
                    attr = getattr(attr,attr_name)
                weight_sublist =  torch.empty(attr.shape[dim],0).to(attr.device)
            elif node.op == 'call_module':
                first_module = all_modules[node.target]
                if isinstance(first_module,nn.MultiheadAttention):
                    first_weight = first_module.out_proj.weight  
                elif isinstance(first_module,(nn.Conv2d,nn.LayerNorm,nn.Linear,nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
                    first_weight = first_module.weight
                weight_sublist =  torch.empty(first_weight.shape[dim],0).to(first_weight.device)
            elif node.op == 'call_function':
                proj_weight_node = node.args[2]
                first_weight = params[proj_weight_node.target]
                weight_sublist =  torch.empty(first_weight.shape[dim],0).to(first_weight.device)
            for node,dim in sublist:
                if node.op == 'get_attr':
                    attr = fx_model
                    for attr_name in node.target.split('.'):
                        attr = getattr(attr,attr_name)
                    weight_sublist = torch.concat([weight_sublist,get_weight_from_parameter(node,fx_model,global_pruning,dim)],axis=1)
                elif node.op == 'call_function':
                    proj_weight_node = node.args[2]
                    weight_sublist = torch.concat([weight_sublist,get_weight_from_parameter(proj_weight_node,fx_model,global_pruning,dim)],axis=1)
                else:
                        weight_sublist = torch.concat([weight_sublist, get_net_weight_node_channel_prune(node, all_modules, next_bn_nodes, next_conv_node_list, ignore_node_target_list, global_pruning, net_weights)], axis=1) 
            for node,dim in sublist:
                if node.target in all_modules:
                    module = all_modules[node.target]
                    if isinstance(module,nn.MultiheadAttention):
                        weight = module.out_proj.weight
                        while len(weight_sublist.shape)<len(weight.shape):
                            weight_sublist = weight_sublist.unsqueeze(-1)
                        if dim != 0 :
                            shape= list(range(len(weight_sublist.shape)))
                            shape[0],shape[dim] = shape[dim],shape[0]
                            weight_sublist = weight_sublist.permute(shape)
                        net_weights[node.name+'_out_proj'] = ((weight_sublist.mean(dim=1) if global_pruning else weight_sublist),dim)
                        if dim!=0:
                            weight_sublist = weight_sublist.permute(shape)
                        weight_sublist = weight_sublist.flatten(1)
                    else:
                        weight = module.weight
                        while len(weight_sublist.shape)<len(weight.shape):
                            weight_sublist = weight_sublist.unsqueeze(-1)
                        if dim != 0 :
                            shape= list(range(len(weight_sublist.shape)))
                            shape[0],shape[dim] = shape[dim],shape[0]
                            weight_sublist = weight_sublist.permute(shape)
                        net_weights[node.name] = ((weight_sublist.mean(dim=1) if global_pruning else weight_sublist),dim)
                        if dim!=0:
                            weight_sublist = weight_sublist.permute(shape)
                        weight_sublist = weight_sublist.flatten(1)
                elif node.op == 'call_function':
                    proj_weight_node = node.args[2]
                    weight = params[proj_weight_node.target]
                    while len(weight_sublist.shape)<len(weight.shape):
                        weight_sublist = weight_sublist.unsqueeze(-1)
                    if dim != 0 :
                        shape= list(range(len(weight_sublist.shape)))
                        shape[0],shape[dim] = shape[dim],shape[0]    
                        weight_sublist = weight_sublist.permute(shape)
                    net_weights[node.name+'_proj'] = ((weight_sublist.mean(dim=1) if global_pruning else weight_sublist),dim)
                    if dim!=0:
                        weight_sublist = weight_sublist.permute(shape)
                    weight_sublist = weight_sublist.flatten(1)
                elif node.op == 'get_attr':
                    attr = fx_model
                    for attr_name in node.target.split('.'):
                        attr = getattr(attr,attr_name)
                    while len(weight_sublist.shape)<len(attr.shape):
                        weight_sublist = weight_sublist.unsqueeze(-1)
                    if dim != 0 :
                        shape= list(range(len(weight_sublist.shape)))
                        shape[0],shape[dim] = shape[dim],shape[0]    
                        weight_sublist = weight_sublist.permute(shape)
                    net_weights[node.name] = ((weight_sublist.mean(dim=1) if global_pruning else weight_sublist),dim)
                    if dim!=0:
                        weight_sublist = weight_sublist.permute(shape)
                    weight_sublist = weight_sublist.flatten(1)
    
    for node in model_graph.nodes:
        if node.target=='output':
            continue
        if node.name not in net_weights:
            if node.args and node.op == 'call_module':
                module = all_modules[node.target]
                if isinstance(module, nn.Conv2d):
                    if channel_pruning:
                        wt = get_net_weight_node_channel_prune(node, all_modules, next_bn_nodes, next_conv_node_list, global_pruning=global_pruning, net_weights=net_weights)
                        net_weights[node.name] = ((wt.mean(dim=1) if global_pruning else wt),0) 
                        # x = get_net_weight_node_channel_prune(node, all_modules, next_bn_nodes, next_conv_node_list)
                        # net_weights[node.name] = torch.rand(list(x.size())).to(x.device)
                        # the net_weight over here is a two dim output, No * (Ni*w*h)
                    else:
                        net_weights[node.name] = (abs(get_bn_adjusted_weight(node=node, modules=all_modules, next_bn_nodes=next_bn_nodes)),0) ### something to be done here, TODO
                
                elif isinstance(module,(nn.LayerNorm)) or ((isinstance(module,nn.Linear) and 'output' not in [n_id.op for n_id in node.users])):
                    wt = module.weight
                    net_weights[node.name] =(( wt.mean(dim=1) if global_pruning else wt),0)
                
                elif isinstance(module,nn.MultiheadAttention):
                    wt = module.in_proj_weight
                    net_weights[node.name] = ((wt.mean(dim=1) if global_pruning else wt),0)
                    wt = module.out_proj.weight
                    layer_name = node.name+'_out_proj'
                    if layer_name not in net_weights:
                        net_weights[layer_name] = ((wt.mean(dim=1) if global_pruning else wt),0)
            elif node.op == 'call_function':
                if node.target == tvmodels.swin_transformer.shifted_window_attention:
                    qkv_weight_node = node.args[1]
                    qkv_weight = params[qkv_weight_node.target]
                    net_weights[node.name] = ((qkv_weight.mean(dim=1) if global_pruning else qkv_weight),0)
                    proj_weight_node = node.args[2]
                    proj_weight = params[proj_weight_node.target]
                    layer_name = node.name+'_proj'
                    if layer_name not in net_weights:
                        net_weights[layer_name] = ((proj_weight.mean(dim=1) if global_pruning else proj_weight),0)
            elif node.op == 'get_attr':
                attr = fx_model
                for attr_name in node.target.split('.'):
                    attr = getattr(attr,attr_name)
                if isinstance(attr,nn.Parameter):
                    net_weights[node.name] = (attr,0)# 0 is taken as default
    return net_weights
