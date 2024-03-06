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


import torch
import torch.fx as fx
import torch.nn as nn

import operator

def find_in_node(orig_node, curr_node, modules, next_conv_node_list):
    # recursive call to find the related conv layers to the orig conv layer in its users (below layers)
    condn = (curr_node.target!='output') and isinstance(curr_node.target, str) and (curr_node.target in modules) and isinstance(modules[curr_node.target], torch.nn.Conv2d)
    if condn and (curr_node.target in modules) and curr_node!=orig_node:
        if modules[curr_node.target].in_channels == modules[orig_node.target].out_channels:
            next_conv_node_list[orig_node.target].append(curr_node)
        return
    elif curr_node.target=='output':
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
        if node.args and isinstance(node.target, str) and (node.target in modules):
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


def find_conv_layer_in_prev(node, connected_list, modules):
    # find all the connected nodes in the args(parents) of the current node, whenever a conv is found, we can stop our searching
    for n_id in node.args:
        if isinstance(n_id, torch.fx.node.Node): # removing if the arg is not a node, but constants
            if n_id.args and isinstance(n_id.target, str) and (n_id.target in modules) and isinstance(modules[n_id.target], torch.nn.Conv2d):
                connected_list.append(n_id)
                # if it is a depthwise layer, then we need to go even further into it
                if modules[n_id.target].weight.shape[1]==1:
                    find_conv_layer_in_prev(n_id, connected_list, modules)
                else:
                    return
            elif n_id.target=='x':
                return
            else:
                find_conv_layer_in_prev(n_id, connected_list, modules)
    return 

def find_all_connected_conv(model):
    # returns the list of all conv that share the same output
    if isinstance(model, torch.fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = model
    else:
        fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    model_graph = fx_model.graph
    connected_list_prev = ['init']
    all_connected_list = []
    for node in model_graph.nodes:
        if (len(node.args)>1) and not(node.target in (torch.mul,operator.mul)): # to find layers like add, removing mul layer
            if all(isinstance(n_id, torch.fx.node.Node) for n_id in node.args): # to remove nodes like gemm, which has node, int as two inputs
                # this might be a problem for layers like concat, as they have axis as input and thus is a problem #TODO
                # also if there is concat layer, it wont affect us because any number of channels can be concat to anything #TODO
                # problem with mul layer as well
                connected_list = []
                find_conv_layer_in_prev(node, connected_list, modules)
                if (connected_list_prev[-1]!=connected_list[-1]) and (connected_list_prev[0]!=connected_list[0]):
                    all_connected_list.append(connected_list_prev)
                connected_list_prev = connected_list
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
    # outputs the net weight for a node, incorporates the sum of nodes of the below( inner dimensions )
    # if the node is depthwise, we should output the net_weight of the previous conv to this depthwise
    is_depthwise_node = (all_modules[curr_node.target].weight.shape[1] == 1)
    if is_depthwise_node:
        # search in args until we find a conv
        prev_conv_node  = search_for_first_conv(curr_node, all_modules)
        return net_weights[prev_conv_node.target].unsqueeze(1)
        
    else:
        out_channels = all_modules[curr_node.target].weight.shape[0]
        net_weight = abs(get_bn_adjusted_weight(node=curr_node, modules=all_modules, next_bn_nodes=next_bn_nodes).reshape(out_channels, -1).detach())
        net_weight = net_weight.mean(dim=1).unsqueeze(1) if global_pruning else net_weight
        node_weight_transposed = torch.empty(out_channels,0).to(all_modules[curr_node.target].weight.device)
        node_weight_transposed = get_sum_of_all_conv_below(curr_node, all_modules, next_bn_nodes, next_conv_node_list, node_weight_transposed, out_channels, ignore_node_target_list, global_pruning)
        # We cannot do global using these net weights because there could be multiple connections to one conv, whereas none for the next. Need some decay factor if we want to do global pruning # TODO
        net_weight = torch.concat([net_weight, node_weight_transposed], axis=1)
        return net_weight.mean(dim=1).unsqueeze(1) if global_pruning else net_weight
    # the net_weight over here is a two dim output, No * (Ni*w*h)

def get_net_weights_all(model, next_conv_node_list, all_connected_nodes, next_bn_nodes, channel_pruning, global_pruning=False):
    
    if isinstance(model, fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = model
    else:
        fx_model = fx.symbolic_trace(model)
        
    all_modules = dict(fx_model.named_modules())
    model_graph = fx_model.graph
    
    net_weights = dict()
    
    if all_connected_nodes is not None:
        for sublist in all_connected_nodes:
            ignore_node_target_list = []
            weight_sublist = torch.empty(all_modules[sublist[0].target].weight.shape[0],0).to(all_modules[sublist[0].target].weight.device)
            for node in sublist:
                weight_sublist = torch.concat([weight_sublist, get_net_weight_node_channel_prune(node, all_modules, next_bn_nodes, next_conv_node_list, ignore_node_target_list, global_pruning, net_weights)], axis=1) 
            for node in sublist:
                net_weights[node.target] = weight_sublist.mean(dim=1) if global_pruning else weight_sublist
    
    for node in model_graph.nodes:
        if node.target=='output':
            continue
        if node.args and isinstance(node.target, str) and (node.target in all_modules):
            if isinstance(all_modules[node.target], nn.Conv2d):
                if node.target not in net_weights:
                    if channel_pruning:
                        wt = get_net_weight_node_channel_prune(node, all_modules, next_bn_nodes, next_conv_node_list, global_pruning=global_pruning, net_weights=net_weights)
                        net_weights[node.target] = wt.mean(dim=1) if global_pruning else wt 
                        # x = get_net_weight_node_channel_prune(node, all_modules, next_bn_nodes, next_conv_node_list)
                        # net_weights[node.target] = torch.rand(list(x.size())).to(x.device)
                        # the net_weight over here is a two dim output, No * (Ni*w*h)
                    else:
                        net_weights[node.target] = abs(get_bn_adjusted_weight(node=node, modules=all_modules, next_bn_nodes=next_bn_nodes)) ### something to be done here, TODO
                        
    return net_weights
