try:
    from torchvision import models as tvmodels
    has_tv = True
except Exception as e:
    has_tv = False

try:
    from timm import models as tmmodels
    has_timm = True
except:
    has_timm = False


import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.utils.parametrize as parametrize
from torch.ao.quantization import quantize_fx
import copy
import types

from .... import xnn

from .utils import get_bn_adjusted_weight, create_bn_conv_mapping, create_next_conv_node_list, find_all_connected_nodes, get_net_weight_node_channel_prune, get_net_weights_all,create_channel_pruned_model,_call_functions_to_look
from .parametrization import BlendPruningParametrization, SigmoidPruningParametrization, IncrementalPruningParametrization, HeadChannelBlendPruningParametrization, HeadOnlyBlendPruningParametrization, ChannelOnlyBlendPruningParametrization, PRUNING_CLASS_DICT
from ... import utils

def init(module, *args, example_inputs=None, example_kwargs=None, pruning_ratio=None, total_epochs=None, pruning_class='blend',p=2.0, copy_args=None,
                 pruning_global=False, pruning_type='channel', pruning_init_train_ep=5, pruning_m=None,  add_methods=True, **kwargs):
    copy_args = copy_args or []
    if hasattr(module, '_example_inputs') and hasattr(module, '_example_kwargs'):
        example_inputs= module._example_inputs
        example_kwargs= module._example_kwargs
    else:
        utils.add_example_args_kwargs(module,example_inputs=example_inputs, example_kwargs=example_kwargs)
    module.__prune_params__ =  xnn.utils.AttrDict()
    module.__prune_params__.epoch_count = 0
    module.__prune_params__.pruning_ratio = pruning_ratio
    module.__prune_params__.total_epochs = total_epochs
    module.__prune_params__.sparsity = 0
    module.__prune_params__.init_train_ep = pruning_init_train_ep
    module.__prune_params__.p = p
    
    if pruning_ratio==0:
        raise RuntimeError("pruning ratio of 0 is not supported , try turning off pruning and trying again")
    if not(pruning_ratio and total_epochs):
        raise RuntimeError("pruning ratio and total epochs are necessary to be provided")
    elif not(pruning_ratio):
        raise RuntimeError("pruning ratio should be provided")
    elif not(total_epochs):
        raise RuntimeError("total epochs should be provided")
    
    module.__prune_params__.pruning_class = PRUNING_CLASS_DICT[pruning_class]
    
    #responsible for creating a next mapping (basically helps combine the weight of BN and conv)
    module.__prune_params__.next_bn_nodes = create_bn_conv_mapping(module)
    
    module.__prune_params__.channel_pruning = False
    module.__prune_params__.n2m_pruning = False
    module.__prune_params__.prunechannelunstructured = False
    
    if pruning_type=='channel':
        module.__prune_params__.channel_pruning = True
    elif pruning_type=='n2m':
        module.__prune_params__.n2m_pruning = True
    elif pruning_type=='prunechannelunstructured':
        module.__prune_params__.prunechannelunstructured = True
    elif pruning_type=='unstructured':
        pass
    module.__prune_params__.global_pruning = pruning_global
    
    if module.__prune_params__.n2m_pruning:
        if pruning_m is None:
            raise RuntimeError("The value of m should be provided in case of n:m pruning")
        else:
            module.__prune_params__.m = pruning_m
    else:
        module.__prune_params__.m = None
    
    if module.__prune_params__.channel_pruning:
        # creating the next node list, which contains the connection to all convs to the current conv
        module.__prune_params__.next_conv_node_list = create_next_conv_node_list(module)
        # returns the list of all conv that share the same output
        module.__prune_params__.all_connected_nodes = find_all_connected_nodes(module)
    else:
        module.__prune_params__.next_conv_node_list = None
        module.__prune_params__.all_connected_nodes = None
    
    if module.__prune_params__.n2m_pruning and module.__prune_params__.global_pruning:
        print("Cannot do both global pruning along with n2m pruning, it doesn't make sense! \n")
        raise NotImplementedError
    
    for copy_arg in copy_args:
        setattr(module, copy_arg, getattr(module, copy_arg))
        
    # to get net weights for each of the layers, incorporating all the required dependancies
    module.__prune_params__.net_weights = get_net_weights_all(module, module.__prune_params__.next_conv_node_list, module.__prune_params__.all_connected_nodes, module.__prune_params__.next_bn_nodes, module.__prune_params__.channel_pruning, module.__prune_params__.global_pruning)
    
    if module.__prune_params__.global_pruning:
        if module.__prune_params__.channel_pruning:
            get_layer_pruning_ratio_channel(pruning_ratio)
        else:
            get_layer_pruning_ratio(pruning_ratio)
    
    if add_methods:
        # add a wrapper for model.train()
        module._insert_and_remove_parametrization_during_training = types.MethodType(insert_and_remove_parametrization_during_training, module) 
        module.__pruning_train_backup__ = types.MethodType(module.train.__func__, module)
        module.train = types.MethodType(train, module)
        module.eval = types.MethodType(train, module)
        # other methods
        module.get_layer_pruning_ratio = types.MethodType(get_layer_pruning_ratio, module)
        module.get_layer_pruning_ratio_channel = types.MethodType(get_layer_pruning_ratio_channel, module)
        module.insert_parametrization = types.MethodType(insert_parametrization, module)
        module.remove_parametrization = types.MethodType(remove_parametrization, module)
        module.calculate_sparsity = types.MethodType(calculate_sparsity, module)
    
    return module


def get_layer_pruning_ratio(module, pruning_ratio=0.6):
    fx_model = fx.symbolic_trace(module)
    modules = dict(fx_model.named_modules())
    model_graph = fx_model.graph
    # can also include the batchnorm merged weights over here
    set_of_all_weights = torch.empty(0)
    for node in model_graph.nodes:
        if node.target=='output':
            continue
        if node.args and isinstance(node.target, str):
            if isinstance(modules[node.target], torch.nn.Conv2d) and (modules[node.target].weight.size(1) != 1):
                set_of_all_weights = torch.cat((set_of_all_weights, modules[node.target].weight.mean(dim=[1,2,3]))) if module.__prune_params__.channel_pruning else torch.cat((set_of_all_weights, modules[node.target].weight.view(-1)))
                
    topk = torch.topk(torch.abs(set_of_all_weights), k=int(pruning_ratio*len(set_of_all_weights)), largest=False)
    indexes = topk.indices
    sorted_idx, _ = torch.sort(indexes)
    
    pruning_ratio = dict()
    idx_iter = 0
    total_params = 0
    limit_factor=0.7 # to not prune a layer with more than limit_factor*100% of its weights
    for node in model_graph.nodes:
        if node.target=='output':
            continue
        if node.args and isinstance(node.target, str):
            if isinstance(modules[node.target], torch.nn.Conv2d):
                if (modules[node.target].weight.size(1) != 1):
                    net_params = modules[node.target].weight.shape[0] if module.__prune_params__.channel_pruning else torch.numel(modules[node.target].weight)
                    total_params+=net_params
                    curr_idx = torch.min((sorted_idx < total_params),0).indices.item()-1
                    if curr_idx<0:
                        curr_idx = len(sorted_idx)
                    pruning_ratio[node.target] = (curr_idx-idx_iter)/net_params
                    if pruning_ratio[node.target] >= limit_factor:
                        print("{} is getting the pruning ratio of {} which is a problem, so limiting to {}".format(node.target, pruning_ratio[node.target], limit_factor))
                        pruning_ratio[node.target] = limit_factor
                    idx_iter = curr_idx
                else:
                    pruning_ratio[node.target] = 0 
                
    module.__prune_params__.pruning_ratio = pruning_ratio
    print(pruning_ratio)
    
    return module


def get_layer_pruning_ratio_channel(module, pruning_ratio=0.6): ################## not complete TODO : Global channel pruning
    fx_model = fx.symbolic_trace(module)
    modules = dict(fx_model.named_modules())
    model_graph = fx_model.graph
    # can also include the batchnorm merged weights over here
    # set_of_all_weights = torch.empty(0)
    set_of_all_channel_avg_norms = torch.empty(0)
    for node in model_graph.nodes:
        if node.target=='output':
            continue
        if node.args and isinstance(node.target, str):
            if isinstance(modules[node.target], torch.nn.Conv2d):
                if (modules[node.target].weight.shape[1]!=1) and (modules[node.target].weight.shape[0]>32):
                    net_weight ,dim = module.__prune_params__.net_weights[node.name]
                    mean, std = torch.mean(net_weight,dim), torch.std(net_weight,dim)
                    k = (net_weight-mean)/std
                    set_of_all_channel_avg_norms = torch.cat((set_of_all_channel_avg_norms, k))
                # set_of_all_weights = torch.cat((set_of_all_weights, modules[node.target].weight.view(-1)))
                
    topk = torch.topk(torch.abs(set_of_all_channel_avg_norms), k=int(len(set_of_all_channel_avg_norms)*pruning_ratio), largest=False)
    indexes = topk.indices
    sorted_idx, _ = torch.sort(indexes)
    
    pruning_ratio = dict()
    idx_iter = 0
    total_params = 0
    max_prune = 0.8 # to not prune a layer with more than max_prune*100% of its channels
    for node in model_graph.nodes:
        if node.target=='output':
            continue
        if node.args and isinstance(node.target, str):
            if isinstance(modules[node.target], torch.nn.Conv2d):
                net_params = torch.numel(modules[node.target].weight)
                total_params+=net_params
                curr_idx = torch.min((sorted_idx < total_params),0).indices.item()-1
                if curr_idx<0:
                    curr_idx = len(sorted_idx)
            pruning_ratio[node.target] = min((curr_idx-idx_iter)/net_params, max_prune)
            idx_iter = curr_idx 
    
    # depthwise is not pruning same as the previous conv, maybe some where, 1-2 extra channels pruned, will reduce accuracy           
    module.__prune_params__.pruning_ratio = pruning_ratio
    print(pruning_ratio)
    
    return module
    
    
def train(module, mode: bool = True): 
    if hasattr(module, "__pruning_train_backup__"):
        module.__pruning_train_backup__(mode=mode)
        module = module._insert_and_remove_parametrization_during_training(mode)
    return module


def insert_and_remove_parametrization_during_training(module, mode: bool = True):
    if mode: #train mode
        remove_parametrization(module, leave_parameterized=False) # we do not want to keep the old mask, rest of the weights are adjusted according to this one
        module.__prune_params__.epoch_count += 1
        insert_parametrization(module)
        
    elif module.__prune_params__.epoch_count==module.__prune_params__.total_epochs: # evaluation in the final epoch, we would want to completely prune out the weights
        insert_parametrization(module, binary_mask=True) # binary_mask=True gives hard mask
        remove_parametrization(module)
        calculate_sparsity(module)
        print("The final sparsity of the network is {}".format(module.__prune_params__.sparsity))
    return module


def insert_parametrization(module, binary_mask=False):
    # for each of the nodes/layers, we calculate the parametrization/ mask and then register it over the weights and biases
    
    if isinstance(module, fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = module
    else:
        fx_model = fx.symbolic_trace(module)
        
    modules = dict(module.named_modules())
    model_graph = fx_model.graph
    
    # TODO Experiment among (HeadBlendPruningParametrization, HeadOnlyBlendPruningParametrization, ChannelOnlyBlendPruningParametrization)
    attn_proj_class = HeadChannelBlendPruningParametrization
    attn_proj_class = HeadOnlyBlendPruningParametrization
    attn_proj_class = ChannelOnlyBlendPruningParametrization
    
    pruning_class = module.__prune_params__.pruning_class
    net_weights = module.__prune_params__.net_weights
    module_pruning_ratio= module.__prune_params__.pruning_ratio
    next_bn_nodes = module.__prune_params__.next_bn_nodes
    pruning_kwargs =dict(modules=modules, 
                            channel_pruning=module.__prune_params__.channel_pruning, 
                            n2m_pruning=module.__prune_params__.n2m_pruning, 
                            init_train_ep=module.__prune_params__.init_train_ep, 
                            prunechannelunstructured=module.__prune_params__.prunechannelunstructured,
                            epoch_count=module.__prune_params__.epoch_count, 
                            total_epochs=module.__prune_params__.total_epochs, 
                            binary_mask=binary_mask, 
                            m=module.__prune_params__.m,
                            )
    if pruning_class == BlendPruningParametrization:
        p_kwargs = {'p':module.__prune_params__.p}
    else:
        p_kwargs = {}
    
    for node in model_graph.nodes:
        if node.op in ('output','placeholder'):
            continue
        elif node.op == 'call_function':
            #TODO about the weights and bias nodes
            if has_tv and node.target == tvmodels.swin_transformer.shifted_window_attention:
                qkv_weight_node  = node.args[1]
                proj_weight_node  = node.args[2]
                qkv_module,_ =qkv_weight_node.target.rsplit('.',1)
                qkv_module = modules[qkv_module]
                proj_module,_ =proj_weight_node.target.rsplit('.',1)
                proj_module = modules[proj_module]
                in_proj_net_weight,in_proj_dim = net_weights[node.name]
                out_proj_net_weight,out_proj_dim =net_weights[node.name+'_proj']
                pruning_ratio = module_pruning_ratio if isinstance(module_pruning_ratio, float) else module_pruning_ratio[node.target]
                # For Projection of Attention Layer
                parameterization = pruning_class(curr_node=node,pruning_ratio=pruning_ratio, net_weight = out_proj_net_weight, pruning_dim = out_proj_dim, **pruning_kwargs, **p_kwargs)
                parametrize.register_parametrization(proj_module, "weight", parameterization)
                if module.__prune_params__.channel_pruning:
                    if proj_module.bias is not None:
                        parametrize.register_parametrization(proj_module, "bias", parameterization)
                
                # For QKV of MultiHeadAttention Layer
                if module.__prune_params__.channel_pruning:
                    parameterization = attn_proj_class(curr_node=node, pruning_ratio=pruning_ratio, net_weight = in_proj_net_weight, pruning_dim = in_proj_dim, **pruning_kwargs,**p_kwargs)
                    parametrize.register_parametrization(qkv_module, "weight", parameterization)
                    if qkv_module.bias is not None:
                        parametrize.register_parametrization(qkv_module, "bias", parameterization)
                else:
                    parameterization = pruning_class(curr_node=node, pruning_ratio=pruning_ratio, net_weight = net_weights[node.name], pruning_dim =in_proj_dim, **pruning_kwargs, **p_kwargs)
                    parametrize.register_parametrization(qkv_module.out, "weight", parameterization)
                
        elif node.op == 'get_attr':
            if any(f in [n.target for n in node.users] for f in _call_functions_to_look):
                continue
            attr = fx_model
            attr_names = node.target.split('.')
            module =modules['.'.join(attr_names[:-1])]
            for attr_name in attr_names:
                attr = getattr(attr,attr_name)
            if isinstance(attr,nn.Parameter):
                net_weight, dim = net_weights[node.name]
                pruning_ratio = module_pruning_ratio if isinstance(module_pruning_ratio, float) else module_pruning_ratio[node.target]
                
                parameterization = pruning_class(curr_node=node, pruning_ratio=pruning_ratio, net_weight = net_weight, pruning_dim = dim, **pruning_kwargs, **p_kwargs)
                parametrize.register_parametrization(module, attr_names[-1], parameterization)
                
        elif node.args and node.op == 'call_module':
            module = modules[node.target]
            
            # For Conv2d
            if isinstance(module, nn.Conv2d):
                net_weight,dim = net_weights[node.name]
                pruning_ratio = module_pruning_ratio if isinstance(module_pruning_ratio, float) else module_pruning_ratio[node.target]
                if pruning_class == BlendPruningParametrization:
                    p_kwargs = {'p':module.__prune_params__.p}
                else:
                    p_kwargs = {}
                
                parameterization = pruning_class(curr_node=node, pruning_ratio=pruning_ratio, net_weight = net_weight, pruning_dim = dim, **pruning_kwargs, **p_kwargs)
                
                parametrize.register_parametrization(module, "weight", parameterization)
                if module.__prune_params__.channel_pruning:
                    if module.bias is not None:
                        parametrize.register_parametrization(module, "bias", parameterization)
                    if node.target in next_bn_nodes:
                        parametrize.register_parametrization(modules[next_bn_nodes[node.target].target], "weight", parameterization) 
                        parametrize.register_parametrization(modules[next_bn_nodes[node.target].target], "bias", parameterization)
                    # also put the same parametrization in the next dwconv(along with its BN), if there is one connected to this 
                    for n_id in module.__prune_params__.next_conv_node_list[node.target]:
                        if modules[n_id.target].weight.shape[1]==1: #n_id node is dwconv 
                            parametrize.register_parametrization(modules[n_id.target], "weight", parameterization)
                            if modules[n_id.target].bias is not None:
                                parametrize.register_parametrization(modules[n_id.target], "bias", parameterization)
                            if n_id.target in next_bn_nodes:
                                parametrize.register_parametrization(modules[next_bn_nodes[n_id.target].target], "weight", parameterization) 
                                parametrize.register_parametrization(modules[next_bn_nodes[n_id.target].target], "bias", parameterization)
            # For LayerNorm
            elif isinstance(module, nn.LayerNorm):
                net_weight,dim = net_weights[node.name]
                pruning_ratio = module_pruning_ratio if isinstance(module_pruning_ratio, float) else module_pruning_ratio[node.target]
                if pruning_class == BlendPruningParametrization:
                    p_kwargs = {'p':module.__prune_params__.p}
                else:
                    p_kwargs = {}
                
                parameterization = pruning_class(curr_node=node, pruning_ratio=pruning_ratio, net_weight = net_weight, pruning_dim = dim, **pruning_kwargs, **p_kwargs)
                
                parametrize.register_parametrization(module, "weight", parameterization)
                if module.__prune_params__.channel_pruning:
                    if module.bias is not None:
                        parametrize.register_parametrization(module, "bias", parameterization)
            
            # For MultiHeadAttention 
            elif isinstance(module, nn.MultiheadAttention):
                in_proj_net_weight,in_proj_dim = net_weights[node.name]
                out_proj_net_weight,out_proj_dim =net_weights[node.name+'_out_proj']
                pruning_ratio = module_pruning_ratio if isinstance(module_pruning_ratio, float) else module_pruning_ratio[node.target]
                if pruning_class == BlendPruningParametrization:
                    p_kwargs = {'p':module.__prune_params__.p}
                else:
                    p_kwargs = {}
                # For Outer Projection of MultiHeadAttention Layer
                parameterization = pruning_class(ccurr_node=node,pruning_ratio=pruning_ratio, net_weight = out_proj_net_weight, pruning_dim = out_proj_dim, **pruning_kwargs, **p_kwargs)
                parametrize.register_parametrization(module.out_proj, "weight", parameterization)
                if module.__prune_params__.channel_pruning:
                    if module.out_proj.bias is not None:
                        parametrize.register_parametrization(module.out_proj, "bias", parameterization)
                
                # For Inner Projection of MultiHeadAttention Layer
                if module.__prune_params__.channel_pruning:
                    parameterization = attn_proj_class(curr_node=node,pruning_ratio=pruning_ratio, net_weight = in_proj_net_weight, pruning_dim = in_proj_dim, **pruning_kwargs, **p_kwargs)
                    parametrize.register_parametrization(module, "in_proj_weight", parameterization)
                    if module.in_proj_bias is not None:
                        parametrize.register_parametrization(module, "in_proj_bias", parameterization)
                else:
                    parameterization = pruning_class(curr_node=node,pruning_ratio=pruning_ratio, net_weight = in_proj_net_weight, pruning_dim = in_proj_dim, **pruning_kwargs, **p_kwargs)
                    parametrize.register_parametrization(module.out, "in_proj_weight", parameterization)
                
            #  For Linear layers in model including linears of attention in timm except last that predicts the output       
            elif isinstance(module, nn.Linear) and ('output' not in [n_id.op for n_id in node.users]) :
                net_weight,dim = net_weights[node.name]
                pruning_ratio = module_pruning_ratio if isinstance(module_pruning_ratio, float) else module_pruning_ratio[node.target]
                
                parent_module,name = node.target.rsplit('.',1)
                parent_module = modules[parent_module]
                
                #Layer inside attention of timm 
                if has_timm and isinstance(parent_module,(tmmodels.swin_transformer.WindowAttention,tmmodels.vision_transformer.Attention)):
                    # For  inner of Attention Layer
                    if name == 'qkv':
                        if module.__prune_params__.channel_pruning:
                            parameterization = attn_proj_class(curr_node=node, pruning_ratio=pruning_ratio, net_weight = net_weight, pruning_dim = dim, **pruning_kwargs, **p_kwargs)
                            parametrize.register_parametrization(module, "weight", parameterization)
                            if module.in_proj_bias is not None:
                                parametrize.register_parametrization(module, "bias", parameterization)
                        else:
                            parameterization = pruning_class(curr_node=node, pruning_ratio=pruning_ratio, net_weight = net_weight, pruning_dim = dim, **pruning_kwargs, **p_kwargs)
                            parametrize.register_parametrization(module.out, "in_proj_weight", parameterization)
                    
                    # For Projection of Attention Layer
                    elif name == 'proj':
                        parameterization = pruning_class(curr_node=node, pruning_ratio=pruning_ratio, net_weight = net_weight, pruning_dim = dim, **pruning_kwargs, **p_kwargs)
                        parametrize.register_parametrization(module.out_proj, "weight", parameterization)
                        if module.__prune_params__.channel_pruning:
                            if module.out_proj.bias is not None:
                                parametrize.register_parametrization(module.out_proj, "bias", parameterization)  
                # Normal Linear Layer
                else:                    
                    parameterization = module.__prune_params__.pruning_class(curr_node=node, pruning_ratio=pruning_ratio, net_weight = net_weight, pruning_dim = dim, **pruning_kwargs, **p_kwargs)
                    parametrize.register_parametrization(module, "weight", parameterization)
                    if module.__prune_params__.channel_pruning:
                        if module.bias is not None:
                            parametrize.register_parametrization(module, "bias", parameterization)       
                                    
    return module


def remove_parametrization(module, leave_parameterized=True):
    # leave_parametrized=False would leave the original parameter instead of the parametrized parameter
    params  = dict (module.named_parameters()) 
    modules = dict(module.named_modules())
    
    for name,param in params.items():
        names = name.split('.')
        if len(names)>=3 and names[-1] == 'original' and names[-3] == 'parametrizations':
            module,param_name = modules['.'.join(names[:-3])],names[-2]
            if parametrize.is_parametrized(module, param_name):
                parametrize.remove_parametrizations(module, param_name, leave_parametrized=leave_parameterized) 
    return module


def calculate_sparsity(module):
    num_zeros = 0
    num_elements = 0
    
    # Make layer wise computionally pruning ratio, overall as well #TODO 
    if isinstance(module, fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
        fx_model = module
    else:
        fx_model = fx.symbolic_trace(module)
        
    modules = dict(fx_model.named_modules())
    model_graph = fx_model.graph
    modules_params = []
    for node in model_graph.nodes:
        if node.op == 'call_function':
            if has_tv and node.target == tvmodels.swin_transformer.shifted_window_attention:
                qkv_weight_node  = node.args[1]
                proj_weight_node  = node.args[2]
                qkv_module,_ =qkv_weight_node.target.rsplit('.',1)
                qkv_module = modules[qkv_module]
                modules_params.append((qkv_module,'weight'))
                modules_params.append((qkv_module,'bias'))
                proj_module,_ =proj_weight_node.target.rsplit('.',1)
                proj_module = modules[proj_module]
                modules_params.append((proj_module,'weight'))
                modules_params.append((proj_module,'bias'))
        elif node.op == 'get_attr':
            if any(f in [n.target for n in node.users] for f in _call_functions_to_look):
                continue
            attr = fx_model
            attr_names = node.target.split('.')
            for attr_name in attr_names[:-1]:
                attr = getattr(attr,attr_name)
            modules_params.append((attr,attr_names[-1]))
        elif node.op == 'call_module':
            module = modules[node.target]
            if isinstance(module,(nn.Conv2d,nn.LayerNorm,nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.Linear)):
                modules_params.append((module,'weight'))   
                modules_params.append((module,'bias'))
            elif isinstance(module,nn.MultiheadAttention):
                modules_params.append((module,'in_proj_weight'))   
                modules_params.append((module,'in_proj_bias'))   
                modules_params.append((module.out_proj,'weight'))   
                modules_params.append((module.out_proj,'bias'))  
    for module,param_name in modules_params:
        tensor = getattr(module,param_name)
        num_zeros += torch.sum(tensor==0).item()
        num_elements += torch.numel(tensor)

    module.__prune_params__.sparsity = num_zeros / num_elements
    return module
