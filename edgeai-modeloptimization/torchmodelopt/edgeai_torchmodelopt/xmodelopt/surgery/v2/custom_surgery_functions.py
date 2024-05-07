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

from torch import nn,Tensor
from torch.fx import symbolic_trace, Node
from torchvision import models,ops
import inspect,torch, operator,torchvision
from copy import deepcopy

from ....xnn.layers import resize_with_scale_factor

from . import replacer
from . import custom_modules


def replace_resize_with_scale_factor(model, pattern=None, verbose_mode=False):
    '''
    replaces all resize wih 'resize with scale factor only'
    self-made function is required as we have to modify keyword arguments
    '''

    traced_m=symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    pattern_m= nn.Upsample()
    traced_pattern= symbolic_trace(pattern_m)
    matches= replacer.straight_chain_searcher(traced_m,traced_pattern)
    
    for start,end in matches:
        with traced_m.graph.inserting_before(start):
            kwargs=dict(start.kwargs)
            kwargs.pop('antialias')         # removes unwanted keyword arguments
            new_node= traced_m.graph.call_function(resize_with_scale_factor,start.args,kwargs)
            start.replace_all_uses_with(new_node)
        traced_m.graph.erase_node(start)
    
    traced_m.graph.lint()
    traced_m.recompile()
    if verbose_mode:
        print('resize',len(matches))
    return traced_m

 
def _replace_pool_size_ge_5(model:nn.Module, pattern=None, pool_class=nn.MaxPool2d,pool_function=nn.functional.max_pool2d, verbose_mode=False):
    '''
    replaces all pool 2d module or function having kernel size greater than or equal to 5
    with a stack of pool2d modules having kernel size 3
    
    to have same output pixels original stride is added to last maxpool module
    '''

    traced_model = symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    modules=dict(traced_model.named_modules())
    
    no_of_pool=0
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            #for call module pool
            module=modules[node.target]
            if isinstance(module,pool_class):
                if module.kernel_size >4:
                    k_size=module.kernel_size 
                    stride=module.stride
                    padding=module.padding
                    replacement= nn.Sequential()
                    while k_size > 4:
                        # if k_size % 2 ==0: replacement.append(pool_class(kernel_size=2,stride=1,padding=(0,0,1,1)))
                        if k_size % 2 == 0:
                            replacement.append(pool_class(kernel_size=2, stride=1, padding=(1,1)))
                        else: replacement.append(pool_class(kernel_size=3,stride=1,padding=1))
                        k_size-=2
                    # replacement.append(pool_class(kernel_size=k_size,stride=stride,padding=1 if padding %2 !=0 else (0,0,1,1)))
                    replacement.append(
                        pool_class(kernel_size=k_size, stride=stride, padding=1 if padding % 2 != 0 else (1,1)))
                    replacer._replace_pattern(traced_model,node,node,replacement,no_of_pool)
                    no_of_pool+=1
        
        if node.target == pool_function:
            #for functional pool
            k_size=node.args[1]
            stride=node.kwargs['stride']
            padding=node.kwargs['padding']
            replacement= nn.Sequential()
            if k_size>4:
                while k_size > 4:
                    # if k_size % 2 ==0: replacement.append(pool_class(kernel_size=2,stride=1,padding=(0,0,1,1)))
                    if k_size % 2 == 0:
                        replacement.append(pool_class(kernel_size=2, stride=1, padding=(1,1)))
                    else: replacement.append(pool_class(kernel_size=3,stride=1,padding=1))
                    k_size-=2
                # replacement.append(pool_class(kernel_size=k_size,stride=stride,padding=1 if padding %2 !=0 else (0,0,1,1)))
                replacement.append(
                    pool_class(kernel_size=k_size, stride=stride, padding=1 if padding % 2 != 0 else (1,1)))
                new_node_name=f'replaced_{pool_class.__name__.lower()}_{no_of_pool}'
                traced_model.add_submodule(new_node_name,replacement)
                args=(node.args[0],)
                with traced_model.graph.inserting_before(node):
                    new_node=traced_model.graph.call_module(new_node_name,args,{})
                    node.replace_all_uses_with(new_node)
                traced_model.graph.erase_node(node)
         
    traced_model.graph.lint()
    traced_model.recompile()
    if verbose_mode:
        print(f'{pool_class.__name__.lower()}',no_of_pool)
    return traced_model


def replace_maxpool2d_kernel_size_ge_5(model:nn.Module, pattern=None, verbose_mode=False):
    return _replace_pool_size_ge_5(model, pattern= None, pool_class=nn.MaxPool2d,pool_function=nn.functional.max_pool2d, verbose_mode=verbose_mode)
    
def replace_avgpool2d_kernel_size_ge_5(model:nn.Module, pattern=None, verbose_mode=False):
    return _replace_pool_size_ge_5(model, pattern= None, pool_class=nn.AvgPool2d,pool_function=nn.functional.avg_pool2d, verbose_mode=verbose_mode)


def replace_conv2d_kernel_size_gt_7(model:nn.Module, pattern= None, verbose_mode=False):
    '''
    replaces all conv2d module or function having kernel size greater than or equal to 7
    with a stack of conv2d modules having kernel size 3
    
    to have same output pixels original stride is added to last conv module
    '''

    traced_model = symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    modules=dict(traced_model.named_modules())
    no_of_conv=0
    import math, random
    
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            #for call module conv
            module=modules[node.target]
            if isinstance(module,nn.Conv2d):
                if module.kernel_size[0] > 7:
                    in_channels=module.in_channels
                    out_channels=module.out_channels
                    k_size=module.kernel_size[0]
                    stride=module.stride[0]
                    padding=module.padding[0]
                    replacement= nn.Sequential()
                    while k_size > 7:
                        temp_out_channels= 2**(round(math.log2(in_channels))+random.choice([-1,0,1]))
                        replacement.append(custom_modules.ConvBNRModule(in_channels,temp_out_channels, kernel_size=3,stride=1,padding=1))
                        in_channels=temp_out_channels
                        k_size-=2
                    padding=min(2,padding)
                    replacement.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size,stride=stride,padding=padding))
                    replacer._replace_pattern(traced_model,node,node,replacement,no_of_conv)
                    no_of_conv+=1
        
        if node.target == nn.functional.conv2d:
            #for functional conv
            args=node.args
            weight_node=args[1]
            parent_name,name=replacer._get_parent_name(weight_node.target)
            parent_module=modules[parent_name]
            weight=parent_module.__getattr__(name)
            weight_shape=weight.shape
            stride=args[3][0]
            padding=args[4][0]
            in_channels=weight_shape[1]
            out_channels=weight_shape[0]
            k_size=weight_shape[2]
            replacement= nn.Sequential()
            while k_size > 7:
                temp_out_channels= 2**(round(math.log2(in_channels))+random.choice([-1,0,1]))
                replacement.append(custom_modules.ConvBNRModule(in_channels,temp_out_channels, kernel_size=3,stride=1,padding=1))
                in_channels=temp_out_channels
                k_size-=2
            padding=min(2,padding)
            replacement.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size,stride=stride,padding=padding))
            traced_model.add_submodule(f'replaced_conv_{no_of_conv}',replacement)
            args=(node.args[0],)
            with traced_model.graph.inserting_before(node):
                new_node=traced_model.graph.call_module(f'replaced_conv_{no_of_conv}',args,{})
                node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)
        
    traced_model.graph.lint()
    traced_model.recompile()
    if verbose_mode:
        print('conv changed', no_of_conv)
    return traced_model

def replace_conv2d_kernel_size_6(model:nn.Module, pattern= None, verbose_mode=False):
    '''
    replaces all conv2d module or function having kernel size greater than or equal to 7
    with a stack of conv2d modules having kernel size 3
    
    to have same output pixels original stride is added to last conv module
    '''

    traced_model = symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    modules=dict(traced_model.named_modules())
    no_of_conv=0
    
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            #for call module conv
            module=modules[node.target]
            if isinstance(module,nn.Conv2d):
                if module.kernel_size[0] == 6:
                    in_channels=module.in_channels
                    out_channels=module.out_channels
                    k_size=module.kernel_size[0]
                    stride=module.stride[0]
                    padding=module.padding[0]
                    replacement= nn.Sequential()
                    padding=min(2,padding)
                    replacement.append(nn.Conv2d(in_channels, out_channels, kernel_size=5,stride=stride,padding=padding))
                    replacer._replace_pattern(traced_model,node,node,replacement,no_of_conv)
                    # no_of_conv+=1
        
        if node.target == nn.functional.conv2d:
            #for functional conv
            args=node.args
            weight_node=args[1]
            parent_name,name=replacer._get_parent_name(weight_node.target)
            parent_module=modules[parent_name]
            weight=parent_module.__getattr__(name)
            weight_shape=weight.shape
            stride=args[3][0]
            padding=args[4][0]
            in_channels=weight_shape[1]
            out_channels=weight_shape[0]
            k_size=weight_shape[2]
            replacement= nn.Sequential()
            padding=min(2,padding)
            replacement.append(nn.Conv2d(in_channels, out_channels, kernel_size=5,stride=stride,padding=padding))
            traced_model.add_submodule(f'replaced_conv_{no_of_conv}',replacement)
            args=(node.args[0],)
            with traced_model.graph.inserting_before(node):
                new_node=traced_model.graph.call_module(f'replaced_conv_{no_of_conv}',args,{})
                node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)
        
    traced_model.graph.lint()
    traced_model.recompile()
    if verbose_mode:
        print('conv changed', no_of_conv)
    return traced_model

def replace_cnblock(model:nn.Module, pattern=None, verbose_mode=False):
    traced_model = symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    t_modules= dict(traced_model.named_modules())
    from torchvision.models.convnext import CNBlock
    pattern = symbolic_trace(CNBlock(34,0.125,0.000001))
    matched=replacer.straight_chain_searcher(traced_model,pattern)
    for start,end in matched:
        ptr = start
        # get the first call_module pattern
        attribute_list = []
        while ptr != end and ptr.op == "get_attr":
            attribute_list.append(ptr)
            ptr = ptr.next
        start_conv = t_modules[ptr.target]
        if type(start_conv) == nn.Conv2d:
            dim=start_conv.in_channels
        else: break
        replacement = custom_modules.ReplacementCNBlock(dim)
        replacer._replace_pattern(traced_model, ptr, end, replacement)
        for node in attribute_list:
            traced_model.graph.erase_node(node)
    if verbose_mode:
        print('cnblock',len(matched))
    return traced_model


def replace_layer_norm(model:nn.Module, pattern=None, verbose_mode=False):
    traced_model=remove_identiy(model)
    no_of_layer_norm=0
    t_modules= dict(traced_model.named_modules())

    for node in traced_model.graph.nodes:
        module=None
        replacement=None
        if node.target== nn.functional.layer_norm:
            arg= node.args[0]
            args=[]
            for arg1 in arg.args:
                if type(arg1) == Node:
                    args.append(arg1)
            while len(arg.users)==1 or len(args) == 1 :
                if arg.op in ['get_attr','placeholder']: break
                arg = arg.args[0]
                if (arg.op == 'call_method' and arg.target=='mean') or (arg.target==nn.functional.adaptive_avg_pool2d) or (arg.op == 'call_module' and type(t_modules[arg.target]) == nn.AdaptiveAvgPool2d):
                    break
                args=[]
                for arg1 in arg.args:
                    if type(arg1) == Node:
                        args.append(arg1)
            replacement=None
            prev=arg
            if (prev.op == 'call_method' and prev.target=='mean') or (prev.target==nn.functional.adaptive_avg_pool2d) or (prev.op == 'call_module' and type(t_modules[prev.target]) == nn.AdaptiveAvgPool2d):
                replacement = nn.Identity()
            else:
                num_features=node.args[1][0]
                replacement=custom_modules.ReplaceBatchNorm2d(num_features)
            args=(node.args[0],)
            arg=args[0]
            args_arg=arg.args[0]
            replacement=deepcopy(replacement)
            new_node_name= type(replacement).__name__+str(no_of_layer_norm)
            traced_model.add_submodule(new_node_name,replacement)
            t_modules.update({new_node_name:replacement})
            ptr = node.prev
            with traced_model.graph.inserting_before(node):
                new_node= traced_model.graph.call_module(new_node_name,args,{})
                node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)
            no_of_layer_norm +=1
            while ptr.op == 'get_attr':
                temp = ptr
                ptr=ptr.prev
                traced_model.graph.erase_node(temp)

        elif node.op == 'call_module':
            module=t_modules[node.target]
            if type(module) == nn.LayerNorm:
                arg= node.args[0]
                args=[]
                for arg1 in arg.args:
                    if type(arg1) == Node:
                        args.append(arg1)
                while len(arg.users)==1 or len(args) == 1 :
                    if arg.op in ['get_attr','placeholder']: break
                    arg = arg.args[0]
                    if (arg.op == 'call_method' and arg.target=='mean') or (arg.target==nn.functional.adaptive_avg_pool2d) or (arg.op == 'call_module' and type(t_modules[arg.target]) == nn.AdaptiveAvgPool2d):
                        break
                    args=[]
                    for arg1 in arg.args:
                        if type(arg1) == Node and node.op != 'gett_attr':
                            args.append(arg1)
                replacement=None
                prev=arg
                if (prev.op == 'call_method' and prev.target=='mean') or (prev.target==nn.functional.adaptive_avg_pool2d) or (prev.op == 'call_module' and type(t_modules[prev.target]) == nn.AdaptiveAvgPool2d):
                    replacement = nn.Identity()
                else:
                    num_features=module.normalized_shape[0]
                    replacement= custom_modules.ReplaceBatchNorm2d(num_features)
                parent_name,name=replacer._get_parent_name(node.target)
                replacement=deepcopy(replacement)
                t_modules[node.target]=replacement
                t_modules[parent_name].__setattr__(name,replacement)
                no_of_layer_norm+=1
    traced_model.graph.lint()
    traced_model.recompile()
    if verbose_mode:
        print('layernorm',no_of_layer_norm)
    permute_nodes=[]
    traced_model=remove_identiy(traced_model)
    for node in traced_model.graph.nodes:
        if (node.op == 'call_method' and node.target == 'permute') or (node.target == torch.permute):
            if len(node.users) ==1:
                permute_nodes.append(node)
    i= len(permute_nodes)-1
    while i>=0:
        node=permute_nodes[i]
        arg=node.args[0]
        if len(arg.users) >1:
            i-=1
            continue
        changes_in_dim=[]
        if type(node.args[1]) == int:
            changes_in_dim=[node.args[1:]]
        elif len(node.args)==2:
            changes_in_dim=[node.args[1]]
        while (arg.op == 'call_method' and arg.target == 'permute')or (arg.target == torch.permute):
            args_arg=arg.args[0]
            if type(arg.args[1]) == int:
                changes_in_dim.append(arg.args[1:])
            elif len(arg.args)==2:
                changes_in_dim.append(arg.args[1])
            arg.replace_all_uses_with(args_arg)
            permute_nodes.remove(arg)
            i-=1
            traced_model.graph.erase_node(arg)
            arg=args_arg
        dim=[0,1,2,3]
        j=len(changes_in_dim)-1
        if j>0:
            while j>=0:
                change_in_dim=changes_in_dim[j]
                new_dim=[0,0,0,0]
                for k in range(len(dim)):
                    new_dim[k]=dim[change_in_dim[k]]
                dim=new_dim
                j-=1
            if dim == [0,1,2,3]:
                node.replace_all_uses_with(arg)
                traced_model.graph.erase_node(node)                
            else:
                if type(node.args[1]) == int:
                    node.args=(arg,*dim)
                elif len(node.args)==2:
                    node.args=(arg,dim)
            traced_model.graph.lint()
            traced_model.recompile()
        i-=1
    return traced_model


#not effective so not implemented
def replace_se_layer(model:nn.Module, pattern=None, verbose_mode=False):
    traced_model=remove_identiy(model)
    modules=dict(traced_model.named_modules())
     
    matched=[]
    nodes=[]
    for node in traced_model.graph.nodes:
        nodes.append(node)
    i=0
    activation_func=(nn.functional.relu,
                    nn.functional.relu6,
                    nn.functional.rrelu,
                    nn.functional.hardsigmoid,
                    nn.functional.sigmoid,
                    nn.functional.hardswish,
                    nn.functional.silu,
                    nn.functional.leaky_relu,
                    nn.functional.gelu,
                    nn.functional.hardtanh,)
                                     
    while i< len(nodes):
        node=nodes[i]
        if ((node.op == 'call_module' and isinstance(modules[node.target],nn.AdaptiveAvgPool2d)) 
        or node.target in ('mean',nn.functional.adaptive_avg_pool2d,)):
            node_1= nodes[i+1]
            if ((node_1.op == 'call_module' and isinstance(modules[node_1.target],nn.Conv2d) )
                or node_1.target in (nn.functional.conv2d,)):
                node_2=nodes[i+2]
                if ((node_2.op == 'call_module' and inspect.getmodule(type(modules[node_2.target]))==nn.modules.activation)
                or node_2.target in activation_func):
                    node_3=nodes[i+3]
                    if ((node_3.op == 'call_module' and isinstance(modules[node_3.target],nn.Conv2d) )
                        or node_3.target in (nn.functional.conv2d,)):
                        node_4=nodes[i+4]
                        if ((node_4.op == 'call_module' and inspect.getmodule(type(modules[node_4.target]))==nn.modules.activation)
                or node_4.target in activation_func):
                            node_5=nodes[i+5]
                            if node.target in (torch.mul,operator.mul,):
                                matched.append((node,node_5))
                                i+=6
                            else: i+=5
                    else: i+=4
                else: i+=3
            else: i+=2
        else: i+=1
     
    replacer._replace_all_matches(traced_model,matched,nn.Identity())
    if verbose_mode:
        print('se',len(matched))
    return traced_model


def remove_identiy(model:nn.Module, pattern=None, verbose_mode=False):
    # model=deepcopy(model)
    traced_model=symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    modules= dict(traced_model.named_modules())
    n=0
    nodes=[]
    for node in traced_model.graph.nodes:
        if (node.op == 'call_module'):
            if type(modules[node.target]) == nn.Identity:
                nodes.append(node)
    for node in nodes:
        try:
            node.replace_all_uses_with(node.args[0])
            copy_found=False
            for node_1 in nodes:
                if node!=node_1 and node.target==node_1.target:
                   copy_found=True
            if not copy_found:
                parent_name,name=replacer._get_parent_name(node.target)           
                modules[parent_name].__delattr__(name)
                modules.pop(node.target)
            traced_model.graph.erase_node(node)
            n+=1
        except Exception as e:
            if verbose_mode:
                print(n,e)
    traced_model.graph.lint()
    traced_model.recompile()
    if verbose_mode:
        print('Identity removed',n)
    return traced_model
