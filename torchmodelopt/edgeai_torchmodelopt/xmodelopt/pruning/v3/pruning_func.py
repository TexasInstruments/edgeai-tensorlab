import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.utils.source_matcher_utils import  SourcePartition
import torch.nn.utils.parametrize as parametrize
from torch.ao.quantization import quantize_fx
import types

from .... import xnn
from .utils import get_bn_adjusted_weight, create_bn_conv_mapping, create_next_conv_node_list, find_all_connected_nodes, get_net_weight_node_channel_prune, get_net_weights_all, get_pruning_partitions,get_num_heads_head_dims, get_parameter_indices
from .parametrization import BlendPruningParametrization, SigmoidPruningParametrization, IncrementalPruningParametrization, ChannelOnlyBlendPruningParametrization, HeadChannelBlendPruningParametrization, HeadOnlyBlendPruningParametrization, PRUNING_CLASS_DICT
from ... import utils

def init(module, *args, example_inputs:list=None, example_kwargs:dict=None, pruning_ratio=None, total_epochs=None, pruning_class='blend',p=2.0, pruning_global=False, copy_args=None,
            pruning_type='channel', pruning_init_train_ep=5, pruning_m=None, add_methods=True, copy_attrs=None, **kwargs):
    copy_attrs = copy_attrs or []
    copy_args = copy_args or []
    example_inputs =[] if example_inputs is None else example_inputs
    example_kwargs = example_kwargs or {}
    
    if not (hasattr(module, '_example_inputs') and hasattr(module, '_example_kwargs')):
    # else:
        # This should not get called unless this function is called separately, when called from wrapper model should have example inputs and kwargs
        utils.add_example_args_kwargs(module, example_inputs=example_inputs, example_kwargs=example_kwargs)
    example_inputs = module._example_inputs.pop(0)
    example_kwargs = module._example_kwargs.pop(0)
    
    if isinstance(module,fx.GraphModule):
        #TODO Differnetiate between fx and pt2e graph modules
        # Assuming Default pt2e here
        gm_module = module
    else:
        example_inputs = tuple(example_inputs)
        check_guards = kwargs.get('check_guards', True)
        example_inputs = tuple(example_inputs)
        gm_module = torch.export.export(module, example_inputs, kwargs=example_kwargs).module(check_guards=check_guards)
        from ...utils.helper_functions import allow_exported_model_train_eval
        allow_exported_model_train_eval(gm_module)
    
    gm_module.__prune_params__ =  xnn.utils.AttrDict()
    gm_module.__prune_params__.epoch_count = 0
    gm_module.__prune_params__.pruning_ratio = pruning_ratio
    gm_module.__prune_params__.total_epochs = total_epochs
    gm_module.__prune_params__.sparsity = 0
    gm_module.__prune_params__.init_train_ep = pruning_init_train_ep
    gm_module.__prune_params__.p = p
    
    if pruning_ratio==0:
        raise RuntimeError("pruning ratio of 0 is not supported , try turning off pruning and trying again")
    if not(pruning_ratio and total_epochs):
        raise RuntimeError("pruning ratio and total epochs are necessary to be provided")
    elif not(pruning_ratio):
        raise RuntimeError("pruning ratio should be provided")
    elif not(total_epochs):
        raise RuntimeError("total epochs should be provided")
        
    gm_module.__prune_params__.pruning_class = PRUNING_CLASS_DICT[pruning_class]
    
    #responsible for creating a next mapping (basically helps combine the weight of BN and conv)
    
    gm_module.__prune_params__.pruning_partitions = get_pruning_partitions(gm_module)
    gm_module.__prune_params__.next_bn_nodes = create_bn_conv_mapping(gm_module, gm_module.__prune_params__.pruning_partitions)
    gm_module.__prune_params__.channel_pruning = False
    gm_module.__prune_params__.n2m_pruning = False
    gm_module.__prune_params__.prunechannelunstructured = False
    gm_module.__prune_params__.parametrized_params = set()
    
    if pruning_type=='channel':
        gm_module.__prune_params__.channel_pruning = True
    elif pruning_type=='n2m':
        gm_module.__prune_params__.n2m_pruning = True
    elif pruning_type=='prunechannelunstructured':
        gm_module.__prune_params__.prunechannelunstructured = True
    elif pruning_type=='unstructured':
        pass
    gm_module.__prune_params__.global_pruning = pruning_global
    
    if gm_module.__prune_params__.n2m_pruning:
        if pruning_m is None:
            raise RuntimeError("The value of m should be provided in case of n:m pruning")
        else:
            gm_module.__prune_params__.m = pruning_m
    else:
        gm_module.__prune_params__.m = None
    
    if gm_module.__prune_params__.channel_pruning:
        # creating the next node list, which contains the connection to all convs to the current conv
        gm_module.__prune_params__.next_conv_node_list = create_next_conv_node_list(gm_module, gm_module.__prune_params__.pruning_partitions)
        # returns the list of all conv that share the same output
        gm_module.__prune_params__.all_connected_nodes = find_all_connected_nodes(gm_module, gm_module.__prune_params__.pruning_partitions)
    else:
        gm_module.__prune_params__.next_conv_node_list = None
        gm_module.__prune_params__.all_connected_nodes = None
    
    if gm_module.__prune_params__.n2m_pruning and gm_module.__prune_params__.global_pruning:
        print("Cannot do both global pruning along with n2m pruning, it doesn't make sense! \n")
        raise NotImplementedError
    
    for copy_arg in copy_args:
        if hasattr(module, copy_arg):
            setattr(gm_module, copy_arg, getattr(module, copy_arg))
        
    # to get net weights for each of the layers, incorporating all the required dependancies
    gm_module.__prune_params__.net_weights = get_net_weights_all(gm_module, gm_module.__prune_params__.pruning_partitions, gm_module.__prune_params__.next_conv_node_list, gm_module.__prune_params__.all_connected_nodes, gm_module.__prune_params__.next_bn_nodes, gm_module.__prune_params__.channel_pruning, gm_module.__prune_params__.global_pruning)
    
    if gm_module.__prune_params__.global_pruning:
        if gm_module.__prune_params__.channel_pruning:
            gm_module.__prune_params__.get_layer_pruning_ratio_channel(pruning_ratio)
        else:
            gm_module.__prune_params__.get_layer_pruning_ratio(pruning_ratio)
    #
    if add_methods:
        # add a wrapper for model.train()
        gm_module._insert_and_remove_parametrization_during_training = types.MethodType(insert_and_remove_parametrization_during_training, gm_module) 
        gm_module.__pruning_train_backup__ = types.MethodType(module.train.__func__, gm_module)
        gm_module.train = types.MethodType(train, gm_module)
        gm_module.eval = types.MethodType(train, gm_module)
        # other methods
        gm_module.get_layer_pruning_ratio = types.MethodType(get_layer_pruning_ratio, gm_module)
        gm_module.get_layer_pruning_ratio_channel = types.MethodType(get_layer_pruning_ratio_channel, gm_module)
        gm_module.insert_parametrization = types.MethodType(insert_parametrization, gm_module)
        gm_module.remove_parametrization = types.MethodType(remove_parametrization, gm_module)
        gm_module.calculate_sparsity = types.MethodType(calculate_sparsity, gm_module)
    return gm_module

#TODO pt2e implementation
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

#TODO pt2e implementaion
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


def eval(self, mode: bool = False):
    return train(self, mode)


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


from ...utils.helper_functions import get_class_string
def insert_parametrization(module, binary_mask=False):
    # for each of the nodes/layers, we calculate the parametrization/ mask and then register it over the weights and biases
    
    params = dict(module.named_parameters())
    modules = dict(module.named_modules())
    # TODO Experiment among (HeadBlendPruningParametrization, HeadOnlyBlendPruningParametrization, ChannelOnlyBlendPruningParametrization)
    attn_proj_class = HeadChannelBlendPruningParametrization
    attn_proj_class = HeadOnlyBlendPruningParametrization
    attn_proj_class = ChannelOnlyBlendPruningParametrization
    
    all_partition_nodes =[] 
    for cls,partitions in module.__prune_params__.pruning_partitions.items():
        for partition in partitions:
            nodes = []
            if isinstance(partition,SourcePartition): nodes.extend(partition.nodes)
            # if isinstance(partition,InternalMatch): nodes.extend(list(partition.nodes_map.values()))
            all_partition_nodes.extend([node.name for node in nodes])
    pruning_ratio = module.__prune_params__.pruning_ratio if isinstance(module.__prune_params__.pruning_ratio, float) else module.__prune_params__.pruning_ratio[node.target]
    net_weights = module.__prune_params__.net_weights
    pruning_class = module.__prune_params__.pruning_class
    kwargs ={
        'channel_pruning' :module.__prune_params__.channel_pruning,
        'pruning_ratio':pruning_ratio,
        'n2m_pruning':module.__prune_params__.n2m_pruning, 
        'init_train_ep':module.__prune_params__.init_train_ep, 
        'prunechannelunstructured':module.__prune_params__.prunechannelunstructured, 
        'epoch_count':module.__prune_params__.epoch_count, 
        'total_epochs':module.__prune_params__.total_epochs, 
        'binary_mask':binary_mask, 
        'm':module.__prune_params__.m,
        }
    if  pruning_class == BlendPruningParametrization:
        kwargs['p'] = module.__prune_params__.p
    for node in module.graph.nodes:
        if node.name in all_partition_nodes:
            continue
        if node.target  not in net_weights:
            continue 
        elif node.op == 'get_attr':
            split = node.target.rsplit('.',1)
            parent, param_name = split if len(split) == 2 else ('', split[0])
            # attr =  params['.'.join([parent,'parametrizations',param_name,'original']) ]if  parametrize.is_parametrized(modules[parent],param_name) else getattr(modules[parent], param_name) 
            if parametrize.is_parametrized(modules[parent],param_name) or isinstance(getattr(modules[parent], param_name),nn.Parameter):
                net_weight, dim = net_weights[node.target]
                param_name = node.target
                split = param_name.rsplit('.',1)
                parent, param_name = split if len(split) == 2 else ('', split[0])
                parametrization =  pruning_class(fx_model=module,source=node,source_partition=node, net_weight = net_weight,pruning_dim = dim,**kwargs)
                parametrize.register_parametrization(modules[parent], param_name, parametrization)
    
    for cls,partitions in module.__prune_params__.pruning_partitions.items():
        if isinstance(cls, type):
            cls = get_class_string(cls)
        if cls == get_class_string(nn.Conv2d):
            for partition in partitions:
                weight_index, bias_index = get_parameter_indices(module,cls,partition)
                param_name = partition.nodes[weight_index].target
                net_weight,dim =  net_weights[param_name]
                split = param_name.rsplit('.',1)
                parent, param_name = split if len(split) == 2 else ('', split[0])
                parametrization =  pruning_class(fx_model=module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                parametrize.register_parametrization(modules[parent], param_name, parametrization)
                if len(partition.nodes) ==3 and module.__prune_params__.channel_pruning:
                    param_name = partition.nodes[bias_index].target
                    split = param_name.rsplit('.',1)
                    parent, param_name = split if len(split) == 2 else ('', split[0])
                    parametrize.register_parametrization(modules[parent], param_name, parametrization)
                
        elif cls == get_class_string(nn.BatchNorm2d):
            for partition in partitions:
                weight_index, bias_index = get_parameter_indices(module,cls,partition)
                param_name = partition.nodes[weight_index].target
                net_weight,dim =  net_weights[param_name]
                split = param_name.rsplit('.',1)
                parent, param_name = split if len(split) == 2 else ('', split[0])
                parametrization =  pruning_class(fx_model=module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                parametrize.register_parametrization(modules[parent], param_name, parametrization)
                if len(partition.nodes) ==11 and module.__prune_params__.channel_pruning:
                    param_name = partition.nodes[bias_index].target
                    split = param_name.rsplit('.',1)
                    parent, param_name = split if len(split) == 2 else ('', split[0])
                    parametrize.register_parametrization(modules[parent], param_name, parametrization)
        elif cls == get_class_string(nn.Linear):
            for partition in partitions:
                if any( 'output' in [n.op for n in out.users]for out in partition.output_nodes):
                    continue
                weight_index, bias_index = get_parameter_indices(module,cls,partition)
                param_name = partition.nodes[weight_index].target
                net_weight,dim =  net_weights[param_name]
                split = param_name.rsplit('.',1)
                parent, param_name = split if len(split) == 2 else ('', split[0])
                parametrization =  pruning_class(fx_model=module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                parametrize.register_parametrization(modules[parent], param_name, parametrization)
                if module.__prune_params__.channel_pruning:
                    param_name = partition.nodes[bias_index].target
                    split = param_name.rsplit('.',1)
                    parent, param_name = split if len(split) == 2 else ('', split[0])
                    parametrize.register_parametrization(modules[parent], param_name, parametrization)
        elif cls == get_class_string(nn.LayerNorm):
            for partition in partitions:
                weight_index, bias_index = get_parameter_indices(module,cls,partition)
                param_name = partition.nodes[weight_index].target
                net_weight,dim =  net_weights[param_name]
                split = param_name.rsplit('.',1)
                parent, param_name = split if len(split) == 2 else ('', split[0])
                parametrization =  pruning_class(fx_model=module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                parametrize.register_parametrization(modules[parent], param_name, parametrization)
                if len(partition.nodes) ==6 and module.__prune_params__.channel_pruning:
                    param_name = partition.nodes[bias_index].target
                    split = param_name.rsplit('.',1)
                    parent, param_name = split if len(split) == 2 else ('', split[0])
                    parametrize.register_parametrization(modules[parent], param_name, parametrization)
        elif cls == get_class_string(nn.MultiheadAttention):
            for partition in partitions:
                weight_indices, bias_indices = get_parameter_indices(module,cls,partition)
                param_name = partition.nodes[weight_indices[0]].target
                net_weight,dim =  net_weights[param_name]
                split = param_name.rsplit('.',1)
                parent, param_name = split if len(split) == 2 else ('', split[0])
                if module.__prune_params__.channel_pruning:
                    parametrization1 = attn_proj_class(fx_model=module,source=cls,source_partition=partition, net_weight = net_weight,**kwargs)
                else: 
                    parametrization1 =  pruning_class(fx_model=module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                parametrize.register_parametrization(modules[parent], param_name, parametrization1)
                param_name = partition.nodes[weight_indices[1]].target
                net_weight,dim =  net_weights[param_name]
                split = param_name.rsplit('.',1)
                parent, param_name = split if len(split) == 2 else ('', split[0])
                parametrization2 =  pruning_class(fx_model=module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                parametrize.register_parametrization(modules[parent], param_name, parametrization2)
                
                if module.__prune_params__.channel_pruning:                     
                    param_name = partition.nodes[bias_indices[0]].target
                    split = param_name.rsplit('.',1)
                    parent, param_name = split if len(split) == 2 else ('', split[0])
                    parametrize.register_parametrization(modules[parent], param_name, parametrization1)
                    param_name = partition.nodes[bias_indices[1]].target
                    split = param_name.rsplit('.',1)
                    parent, param_name = split if len(split) == 2 else ('', split[0])
                    parametrize.register_parametrization(modules[parent], param_name, parametrization2)
                                    
    return module

def remove_parametrization(module, leave_parameterized=True):
    # leave_parametrized=False would leave the original parameter instead of the parametrized parameter
    params  = dict (module.named_parameters()) 
    modules = dict(module.named_modules())
    for name,param in params.items():
        names = name.split('.')
        if len(names)>=3 and names[-1] == 'original' and names[-3] == 'parametrizations':
            param_name = names[-2]
            parent = '.'.join(names[:-3])
            if parametrize.is_parametrized(modules[parent], param_name):
                module.__prune_params__.parametrized_params.add('.'.join([parent,names[-2]]) if len(parent) else names[-2])
                parametrize.remove_parametrizations(modules[parent], param_name, leave_parametrized=leave_parameterized) 
    return module
        
def calculate_sparsity(module):
    num_zeros = 0
    num_elements = 0
    
    # Make layer wise computionally pruning ratio, overall as well #TODO 
    params = dict(module.named_parameters())
    for param_name in module.__prune_params__.parametrized_params:
        tensor = params[param_name]
        num_zeros += torch.sum(tensor==0).item()
        num_elements += torch.numel(tensor)

    module.__prune_params__.sparsity = num_zeros / num_elements if num_elements != 0 else None
    return module

