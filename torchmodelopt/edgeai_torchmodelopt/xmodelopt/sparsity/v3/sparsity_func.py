import torch
from torch import _dynamo as torch_dynamo
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.utils.source_matcher_utils import  SourcePartition
import torch.nn.utils.parametrize as parametrize
from torch.ao.quantization import quantize_fx
import types

from .... import xnn
from .utils import get_sparsity_nodes, register_n2m_filters, get_all_weights, register_n2m_weight_funcs
from .parametrization import SPARSITY_CLASS_DICT, N2MSparsityParametrization
from ... import utils

def init(module, *args, example_inputs:list=None, example_kwargs:dict=None, sparsity_ratio=None, total_epochs=None, p=2.0, sparsity_global=False, copy_args=None,
            sparsity_type='n2m', sparsity_init_train_ep=5, sparsity_m=None, add_methods=True, aten_graph=True, copy_attrs=None, **kwargs):
    copy_attrs = copy_attrs or []
    copy_args = copy_args or []
    example_inputs =[] if example_inputs is None else example_inputs
    example_kwargs = example_kwargs or {}
    mode = kwargs.get('mode', 'topk')
    
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
    
    gm_module.__sparse_params__ =  xnn.utils.AttrDict()
    gm_module.__sparse_params__.aten_graph = gm_module.__sparse_params__.pre_dispatch = aten_graph
    gm_module.__sparse_params__.epoch_count = 0
    gm_module.__sparse_params__.sparsity_ratio = sparsity_ratio
    gm_module.__sparse_params__.total_epochs = total_epochs
    gm_module.__sparse_params__.sparsity = 0
    gm_module.__sparse_params__.init_train_ep = sparsity_init_train_ep
    gm_module.__sparse_params__.p = p
    gm_module.__sparse_params__.mode = mode
    
    if sparsity_ratio==0:
        raise RuntimeError("sparsity ratio of 0 is not supported , try turning off sparsity and trying again")
    if not(sparsity_ratio and total_epochs):
        raise RuntimeError("sparsity ratio and total epochs are necessary to be provided")
    elif not(sparsity_ratio):
        raise RuntimeError("sparsity ratio should be provided")
    elif not(total_epochs):
        raise RuntimeError("total epochs should be provided")
        
    gm_module.__sparse_params__.sparsity_class = SPARSITY_CLASS_DICT[sparsity_type]

    gm_module.__sparse_params__.n2m_sparsity = False
    gm_module.__sparse_params__.unstructured = False
    gm_module.__sparse_params__.parametrized_params = set()
    

    if sparsity_type=='n2m':
        gm_module.__sparse_params__.n2m_sparsity = True
    elif sparsity_type=='unstructured':
        gm_module.__sparse_params__.unstructured = True
        
    gm_module.__sparse_params__.global_sparsity = sparsity_global
    
    args = []        
    if gm_module.__sparse_params__.n2m_sparsity:
        if sparsity_m is None:
            raise RuntimeError("The value of m should be provided in case of n:m sparsity")
        else:
            gm_module.__sparse_params__.m = sparsity_m
            gm_module.__sparse_params__.n = sparsity_n = round(sparsity_ratio*sparsity_m)
        args += [sparsity_n, sparsity_m]
        register_n2m_filters(sparsity_n, sparsity_m)
        register_n2m_weight_funcs(sparsity_n, sparsity_m)
    else:
        gm_module.__sparse_params__.m = None
    
    args += [sparsity_type]
    gm_module.__sparse_params__.sparsity_nodes = get_sparsity_nodes(gm_module, *args)
    gm_module.__sparse_params__.weights = get_all_weights(gm_module, gm_module.__sparse_params__.sparsity_nodes, )
    
    if gm_module.__sparse_params__.n2m_sparsity and gm_module.__sparse_params__.global_sparsity:
        print("Cannot do both global sparsity along with n2m sparsity, it doesn't make sense! \n")
        raise NotImplementedError
    
    for copy_arg in copy_args:
        if hasattr(module, copy_arg):
            setattr(gm_module, copy_arg, getattr(module, copy_arg))
    
    # if gm_module.__sparse_params__.global_sparsity:
    #     gm_module.__sparse_params__.get_layer_sparsity_ratio(sparsity_ratio)
    #
    if add_methods:
        # add a wrapper for model.train()
        gm_module._insert_and_remove_parametrization_during_training = types.MethodType(insert_and_remove_parametrization_during_training, gm_module) 
        gm_module.__sparsity_train_backup__ = types.MethodType(module.train.__func__, gm_module)
        gm_module.train = types.MethodType(train, gm_module)
        gm_module.eval = types.MethodType(train, gm_module)
        # # other methods
        # gm_module.get_layer_sparsity_ratio = types.MethodType(get_layer_sparsity_ratio, gm_module)
        # gm_module.get_layer_sparsity_ratio_channel = types.MethodType(get_layer_sparsity_ratio_channel, gm_module)
        gm_module.insert_parametrization = types.MethodType(insert_parametrization, gm_module)
        gm_module.remove_parametrization = types.MethodType(remove_parametrization, gm_module)
        gm_module.calculate_sparsity = types.MethodType(calculate_sparsity, gm_module)
    return gm_module

# TODO implement for sparsity from sparsity with pt2e
def get_layer_sparsity_ratio(module, sparsity_ratio=0.6):
    pass

def train(module, mode: bool = True): 
    if hasattr(module, "__sparsity_train_backup__"):
        module.__sparsity_train_backup__(mode=mode)
        module = module._insert_and_remove_parametrization_during_training(mode)
    return module

def eval(self, mode: bool = False):
    return train(self, mode)

def insert_and_remove_parametrization_during_training(module, mode: bool = True):
    if mode: #train mode
        remove_parametrization(module, leave_parameterized=False) # we do not want to keep the old mask, rest of the weights are adjusted according to this one
        module.__sparse_params__.epoch_count += 1
        insert_parametrization(module)
        
    elif module.__sparse_params__.epoch_count==module.__sparse_params__.total_epochs: # evaluation in the final epoch, we would want to completely sparse out the weights
        remove_parametrization(module, leave_parameterized=False) # we do not want to keep the old mask, rest of the weights are adjusted according to this one
        insert_parametrization(module, binary_mask=True) # binary_mask=True gives hard mask
        remove_parametrization(module)
        calculate_sparsity(module)
        print("The final sparsity of the network is {}".format(module.__sparse_params__.sparsity))
    return module

def calculate_sparsity(module):
    num_zeros = 0
    num_elements = 0
    
    # Make layer wise computionally sparsity ratio, overall as well #TODO 
    params = dict(module.named_parameters())
    modules = dict(module.named_modules())
    for param_name in module.__sparse_params__.parametrized_params:
        parent_module = param_name.rsplit('.',1)
        parent_module , param_name = parent_module if len(parent_module)>1 else ('', parent_module)
        parent_module = modules[parent_module]
        tensor = getattr(parent_module,param_name)
        num_zeros += torch.sum(tensor==0).item()
        num_elements += torch.numel(tensor)

    module.__sparse_params__.sparsity = num_zeros / num_elements
    return module

def remove_parametrization(module: fx.GraphModule, leave_parameterized=True):
    # leave_parametrized=False would leave the original parameter instead of the parametrized parameter
    params  = dict (module.named_parameters()) 
    modules = dict(module.named_modules())
    for name,param in params.items():
        names = name.split('.')
        if len(names)>=3 and names[-1] == 'original' and names[-3] == 'parametrizations':
            parent_module = '.'.join(names[:-3])
            parent_module,param_name = modules[parent_module],names[-2]
            if parametrize.is_parametrized(parent_module, param_name):
                module.__sparse_params__.parametrized_params.add('.'.join(names[:-3]+names[-2:-1]))
                parametrize.remove_parametrizations(parent_module, param_name, leave_parametrized=leave_parameterized) 
    return module

def insert_parametrization(module:fx.GraphModule, binary_mask=False):
    params = dict(module.named_parameters())
    modules = dict(module.named_modules())
    sparsity_ratio = module.__sparse_params__.sparsity_ratio 
    weights_dict = module.__sparse_params__.weights
    sparsity_class = module.__sparse_params__.sparsity_class
    
    kwargs = {}
    if sparsity_class == N2MSparsityParametrization:
        kwargs.update(dict(
            n = module.__sparse_params__.n,
            m = module.__sparse_params__.m,
            binary_mask = binary_mask,
            epoch_count = module.__sparse_params__.epoch_count,
            init_train_ep = module.__sparse_params__.init_train_ep,
            total_epochs = module.__sparse_params__.total_epochs,
            p = module.__sparse_params__.p,
            mode = module.__sparse_params__.mode,
        ))
    for key, nodes_list in module.__sparse_params__.sparsity_nodes.items():
        weights_list = weights_dict[key]
        for nodes, weights in zip(nodes_list, weights_list):
            if isinstance(weights, set):
                weights = list(weights)
            if isinstance(weights, (list, tuple)):
                weights = {x:[x] for x in weights}
            assert isinstance(weights, dict), 'Weight should be a dict by now!'
            for main_weight, weights_to_be_sparsified in weights.items():
                tensor = params[main_weight]
                # print(main_weight, tensor.shape)
                parametrization = sparsity_class(key, nodes, tensor=tensor, **kwargs)
                for weight in weights_to_be_sparsified:
                    parent_module = weight.rsplit('.',1)
                    parent_module , name = parent_module if len(parent_module)>1 else ('', parent_module)
                    parent_module = modules[parent_module]
                    parametrize.register_parametrization(parent_module, name, parametrization)
            
            