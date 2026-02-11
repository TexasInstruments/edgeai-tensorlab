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

filter_funcs_dict = dict()

def register_filter(*args, func=None):
    """Registers a filter function for use in sparsity operations.
    
    This decorator function registers a filter function with the specified arguments
    as a key in the global filter_funcs_dict. Filter functions are used to identify
    nodes in the computation graph that are compatible with specified sparsity patterns.
    
    Args:
        *args: Variable length arguments used as a key for the filter function.
               Typically includes layer type and sparsity pattern information.
        func: The filter function to register. If None, returns a decorator.
        
    Returns:
        If func is provided, returns the registered function.
        If func is None, returns a decorator function that will register the decorated function.
        
    Example:
        @register_filter('Conv2d', 2, 4)
        def conv_filter_func(module):
            # Filter implementation
            return filtered_nodes
    """
    def _registered(func):
        filter_funcs_dict[args] = func
        return func
    if func  is not None:
        return _registered(func)
    return _registered

def register_n2m_filter(key, n, m, func=None):
    """Registers a filter function specifically for n:m sparsity pattern.
    
    This is a convenience function that calls register_filter with 'n2m' as part
    of the key to indicate the n:m sparsity pattern. The n:m pattern keeps n non-zero
    elements in each block of m elements, resulting in (m-n)/m sparsity.
    
    Args:
        key: The type of layer or operation to filter (e.g., 'Conv2d', 'Linear').
        n: The n value in n:m sparsity pattern (number of non-zero elements per block).
        m: The m value in n:m sparsity pattern (block size).
        func: The filter function to register. If None, returns a decorator.
        
    Returns:
        The registered function or a decorator.
        
    Example:
        @register_n2m_filter('Conv2d', 2, 4)
        def conv_n2m_filter(module):
            # Implementation to filter Conv2d layers compatible with 2:4 sparsity
            return filtered_nodes
    """
    return register_filter(key,  n, m, 'n2m', func=func)

def register_n2m_filters(n, m):
    """Registers a set of filter functions for various layer types for n:m sparsity pattern.
    
    This function defines and registers filter functions for Conv2d, Linear, and matmul
    operations with n:m sparsity patterns. These filters identify nodes in the computation
    graph that can be sparsified according to the n:m pattern.
    
    The function defines three filter functions:
    1. convs_filter_func: For Conv2d operations
    2. linears_filter_func: For Linear operations
    3. matmul_filter_func: For matrix multiplication operations
    
    Each filter verifies that the operation's dimensions are compatible with the n:m pattern
    (i.e., input and output channels are divisible by m).
    
    Args:
        n: The n value in n:m sparsity pattern (number of non-zero elements per block).
        m: The m value in n:m sparsity pattern (block size).
        
    Note:
        These filter functions examine the node's weight tensor shapes to determine
        compatibility with n:m sparsity patterns. A layer is compatible if its dimensions
        (e.g., input/output channels) are divisible by m.
        
        These functions serve as examples for implementing additional filter functions
        for other layer types or sparsity patterns.
    """
    @register_n2m_filter('Conv2d', n, m)
    def convs_filter_func(module):
        """Filters Conv2d operations that are compatible with n:m sparsity.
        
        This function examines all Conv2d operations in the module and determines
        which ones are compatible with n:m sparsity based on their dimensions.
        For Conv2d, both input and output channels must be divisible by m.
        
        Args:
            module: A GraphModule to filter nodes from.
            
        Returns:
            list: List of node lists that can be sparsified.
        """
        assert isinstance(module, (fx.GraphModule)), f'GraphModule object should be given! but got object of type {module.__class__.__name__}'
        ret = []
        params = dict(module.named_parameters())
        graph = module.graph 
        for node in graph.nodes:
            if node.target != torch.ops.aten.conv2d.default:
                continue
            weight = node.args[1]
            if weight.op != 'get_attr':
                # Skipping as weight tensor is variable to conv
                continue
            weight = params.get(weight.target)
            out_channel, in_channel, *kernel_size = weight.shape
            if out_channel % m != 0 or in_channel % m != 0 or not all(k==1 for k in kernel_size ):
                # skipping as conv is not supported for n:m sparsity
                continue
            ret.append([node])
        
        return ret

    @register_n2m_filter('Linear', n, m)
    def linears_filter_func(module):
        """Filters Linear operations that are compatible with n:m sparsity.
        
        This function examines all Linear operations in the module and determines
        which ones are compatible with n:m sparsity based on their dimensions.
        For Linear layers, both input and output features must be divisible by m.
        
        Args:
            module: A GraphModule to filter nodes from.
            
        Returns:
            list: List of node lists that can be sparsified.
        """
        assert isinstance(module, (fx.GraphModule)), f'GraphModule object should be given! but got object of type {module.__class__.__name__}'
        ret = []
        params = dict(module.named_parameters())
        graph = module.graph 
        for node in graph.nodes:
            if node.target != torch.ops.aten.linear.default:
                continue
            weight = node.args[1]
            if weight.op != 'get_attr':
                # Skipping as weight tensor is variable to conv
                continue
            weight = params.get(weight.target)
            out_channel, in_channel= weight.shape
            if out_channel % m != 0 or in_channel % m != 0 :
                # skipping as linear is not supported for n:m sparsity
                continue
            ret.append([node])
        
        return ret

    @register_n2m_filter('matmul', n, m)
    def matmul_filter_func(module):
        """Filters matmul operations that are compatible with n:m sparsity.
        
        This function examines all matrix multiplication operations in the module
        and determines which ones are compatible with n:m sparsity based on their
        dimensions. For matmul operations, both dimensions of the weight matrices
        must be divisible by m.
        
        Unlike Conv2d and Linear filters, this function checks both arguments to the
        matmul operation since either could be the weight tensor that needs sparsification.
        This only hnadles matmuls with immediate inputs are parameters as the parameter can
        go through any pre-processing before coming to matmul resulting a large number of 
        possibilities.
        
        Args:
            module: A GraphModule to filter nodes from.
            
        Returns:
            list: List of node lists that can be sparsified.
        """
        assert isinstance(module, (fx.GraphModule)), f'GraphModule object should be given! but got object of type {module.__class__.__name__}'
        ret = []
        params = dict(module.named_parameters())
        graph = module.graph 
        for node in graph.nodes:
            if node.target != torch.ops.aten.matmul.default:
                continue
            
            weight = node.args[1]
            if weight.op == 'get_attr':
                weight = params.get(weight.target)
                out_channel, in_channel= weight.shape
                if out_channel % m == 0 and in_channel % m == 0 :
                    ret.append([node])
            weight = node.args[0]
            if weight.op == 'get_attr':
                weight = params.get(weight.target)
                out_channel, in_channel= weight.shape
                if out_channel % m == 0 and in_channel % m == 0 :
                    ret.append([node]) if [node] not in ret else None
            
        
        return ret


def get_sparsity_nodes(module: fx.GraphModule, *args):
    """Retrieves nodes from a module that are compatible with the specified sparsity pattern.
    
    This function iterates through registered filter functions and applies them to the module
    to identify nodes that match the specified sparsity pattern arguments. It selects only
    those filter functions whose keys contain all of the specified arguments.
    
    Args:
        module (fx.GraphModule): The graph module to analyze for sparsifiable nodes.
        *args: Variable length arguments used to filter which sparsity patterns to consider.
            Only filter functions whose keys contain all of the provided args will be used.
            For example, providing ('Conv2d', 2, 4) would match filter functions registered
            with keys that include all three of these elements.
            
    Returns:
        dict: A dictionary mapping filter keys to lists of nodes that can be sparsified.
            Each key corresponds to a registered filter function, and the value is the
            result of applying that filter function to the module.
            
    Example:
        If args = ('n2m',), only filter functions registered with keys containing 'n2m'
        will be used. This allows for filtering based on sparsity type.
    """
    ret = {}
    for key,func in filter_funcs_dict.items():
        if args and not all(a in key for a in args):
            continue
        ret[key] = func(module)
    return ret

weight_func_dict = {}

def register_weigth_func(*args, func=None):
    """Registers a weight function for use in sparsity operations.
    
    This decorator function registers a weight function with the specified arguments
    as a key in the global weight_func_dict. Weight functions are responsible for
    extracting weight parameters from nodes identified by filter functions.
    
    Args:
        *args: Variable length arguments used as a key for the weight function.
               Typically matches the key used for the corresponding filter function.
        func: The weight function to register. If None, returns a decorator.
        
    Returns:
        If func is provided, returns the registered function.
        If func is None, returns a decorator function that will register the decorated function.
        
    Note:
        This function name contains a typo ('weigth' instead of 'weight'), but it is
        maintained for backward compatibility.
        
    Example:
        @register_weigth_func('Conv2d', 2, 4, 'n2m')
        def get_conv_weights(module, nodes):
            # Implementation to extract weight tensors
            return weights
    """
    def _registered(func):
        weight_func_dict[args] = func
        return func
    if func  is not None:
        return _registered(func)
    return _registered

def register_n2m_weight_func(key, n, m, func=None):
    """Registers a weight function specifically for n:m sparsity pattern.
    
    This is a convenience function that calls register_weigth_func with 'n2m' as part
    of the key to indicate the n:m sparsity pattern. Weight functions registered with
    this decorator are responsible for extracting weight parameters from nodes that
    will be sparsified using the n:m pattern.
    
    Args:
        key: The type of layer or operation (e.g., 'Conv2d', 'Linear').
        n: The n value in n:m sparsity pattern (number of non-zero elements per block).
        m: The m value in n:m sparsity pattern (block size).
        func: The weight function to register. If None, returns a decorator.
        
    Returns:
        The registered function or a decorator.
        
    Example:
        @register_n2m_weight_func('Conv2d', 2, 4)
        def get_conv_weights(module, nodes):
            # Implementation to extract weight tensors from Conv2d nodes
            return weights
    """
    return register_weigth_func(key, n, m, 'n2m', func=func)

def register_n2m_weight_funcs(n,m):
    """Registers a set of weight functions for various layer types for n:m sparsity pattern.
    
    This function defines and registers weight functions for Conv2d, Linear, and matmul
    operations with n:m sparsity patterns. These functions extract the weight parameters
    from the identified nodes to be used in sparsification.
    
    The function registers three weight functions:
    1. get_conv_weights: For Conv2d operations
    2. get_linear_weights: For Linear operations 
    3. get_matmul_weights: For matrix multiplication operations
    
    Args:
        n: The n value in n:m sparsity pattern (number of non-zero elements per block).
        m: The m value in n:m sparsity pattern (block size).
        
    Note:
        These weight functions return weight parameter identifiers in one of these formats:
        - List/tuple/set: If weights are used to mask only themselves.
        - Dictionary: Mapping weights that create masks to lists of weights the masks
          will be applied to (the source weights must be included in their own list).
        
        Current implementation limitations:
        - Only a single weight can be used to create a mask.
          TODO: If required, implement support for multiple weights affecting mask generation.
        
        - Mask can only be directly multiplied with the weight in parametrization.
          TODO: If required, implement support for applying masks to different types of weights.
    """
    @register_n2m_weight_func('Conv2d', n, m)
    def get_conv_weights(module: fx.GraphModule, nodes:list[fx.Node]):
        """Extracts weight parameters from Conv2d nodes.
        
        This function extracts the weight tensor parameters from Conv2d operation nodes
        in the computation graph. For Conv2d operations in PyTorch, the weight tensor
        is typically the second argument (index 1) to the operation.
        
        Args:
            module: The GraphModule containing the nodes.
            nodes: List of Conv2d nodes to extract weights from.
            
        Returns:
            list: List of weight parameter targets (attribute names).
        """
        conv = nodes[0]
        weight = conv.args[1]
        return [weight.target]
    
    @register_n2m_weight_func('Linear', n, m)
    def get_linear_weights(module: fx.GraphModule, nodes:list[fx.Node]):
        """Extracts weight parameters from Linear nodes.
        
        This function extracts the weight tensor parameters from Linear operation nodes
        in the computation graph. For Linear operations in PyTorch, the weight tensor
        is typically the second argument (index 1) to the operation.
        
        Args:
            module: The GraphModule containing the nodes.
            nodes: List of Linear nodes to extract weights from.
            
        Returns:
            list: List of weight parameter targets (attribute names).
        """
        fc = nodes[0]
        weight = fc.args[1]
        return [weight.target]
    
    @register_n2m_weight_func('matmul', n, m)
    def get_matmul_weights(module: fx.GraphModule, nodes:list[fx.Node]):
        """Extracts weight parameters from matmul nodes.
        
        This function extracts the weight tensor parameters from matrix multiplication
        operation nodes in the computation graph. Unlike Conv2d and Linear operations,
        either argument to the matmul operation could be a weight tensor that needs
        to be sparsified. This function identifies and returns all weight parameters
        found in the matmul arguments.
        
        Args:
            module: The GraphModule containing the nodes.
            nodes: List of matmul nodes to extract weights from.
            
        Returns:
            list: List of weight parameter targets (attribute names).
        """
        matmul = nodes[0]
        return [a.target for a in matmul.args if a.op == 'get_attr']
    

def get_all_weights(module: fx.GraphModule, nodes_dict:dict[tuple,list[fx.Node]]):
    """Extracts all weight parameters from the nodes identified for sparsification.
    
    This function iterates through the dictionary of nodes identified for sparsification
    and applies the corresponding weight functions to extract the weight parameters.
    The weight functions are selected based on the same keys used for the filter functions,
    ensuring that the appropriate weight extraction logic is applied to each node type.
    
    Args:
        module (fx.GraphModule): The graph module containing the nodes.
        nodes_dict (dict): A dictionary mapping filter keys to lists of nodes.
            This is typically the output of get_sparsity_nodes().
            
    Returns:
        dict: A dictionary mapping filter keys to lists of weight parameters.
            Each key corresponds to a filter key from nodes_dict, and the value
            is a list containing the results of applying the corresponding weight
            function to each group of nodes.
            
    Example:
        For a key ('Conv2d', 2, 4, 'n2m'), the function would use the weight function
        registered with that key to extract weights from the corresponding nodes.
    """
    ret = {}
    for key, nodes_list in nodes_dict.items():
        results = []
        
        for nodes in nodes_list:
            results.append(weight_func_dict[key](module, nodes, ))
        
        ret[key]= results
    return ret     
