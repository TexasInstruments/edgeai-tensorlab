from torch import nn
from torch.fx import symbolic_trace
from . import replacer
from . import custom_modules
    
from edgeai_torchtoolkit.v1.xnn.layers import resize_with_scale_factor


def replace_resize_with_scale_factor(model):
    '''
    replaces all resize wih 'resize with scale factor only'
    self-made function is required as we have to modify keyword arguments
    '''

    traced_m =  symbolic_trace(model)
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
    return traced_m

def replace_maxpool2d_kernel_size_ge_5(model:nn.Module):
    '''
    replaces all maxpool2d module or function having kernel size greater than or equal to 5
    with a stack of maxpool2d modules having kernel size 3
    
    to have same output pixels original stride is added to last maxpool module
    '''
    
    traced_model=symbolic_trace(model)
    modules=dict(traced_model.named_modules())
    
    no_of_max_pool=0
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            #for call module maxpool
            module=modules[node.target]
            if isinstance(module,nn.MaxPool2d):
                if module.kernel_size >3:
                    k_size=module.kernel_size 
                    stride=module.stride
                    replacement= nn.Sequential()
                    while k_size > 3:
                        replacement.append(nn.MaxPool2d(kernel_size=3,stride=1,padding=1))
                        k_size-=2
                    replacement.append(nn.MaxPool2d(kernel_size=k_size,stride=stride,padding=1))
                    replacer._replace_pattern(traced_model,node,node,replacement,no_of_max_pool)
                    no_of_max_pool+=1
        
        if node.target == nn.functional.max_pool2d:
            #for functional maxpool
            k_size=node.args[1]
            stride=node.kwargs['stride']
            replacement= nn.Sequential()
            while k_size > 3:
                replacement.append(nn.MaxPool2d(kernel_size=3,stride=1,padding=1))
                k_size-=2
            replacement.append(nn.MaxPool2d(kernel_size=k_size,stride=stride,padding=1))
            traced_model.add_submodule(f'replaced_maxpool_{no_of_max_pool}',replacement)
            args=(node.args[0],)
            with traced_model.graph.inserting_before(node):
                new_node=traced_model.graph.call_module(f'replaced_maxpool_{no_of_max_pool}',args,{})
                node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)
        
        traced_model.graph.lint()
        traced_model.recompile()
    return traced_model


def replace_avgpool2d_kernel_size_ge_5(model:nn.Module):
    '''
    replaces all avgpool2d module or function having kernel size greater than or equal to 5
    with a stack of avgpool2d modules having kernel size 3
    
    to have same output pixels original stride is added to last avgpool module
    '''

    traced_model=symbolic_trace(model)
    modules=dict(traced_model.named_modules())
    no_of_avg_pool=0
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            #for call module avgpool
            module=modules[node.target]
            if isinstance(module,nn.AvgPool2d):
                if module.kernel_size >3:
                    k_size=module.kernel_size 
                    stride=module.stride
                    replacement= nn.Sequential()
                    while k_size > 3:
                        replacement.append(nn.AvgPool2d(kernel_size=3,stride=1,padding=1))
                        k_size-=2
                    replacement.append(nn.AvgPool2d(kernel_size=k_size,stride=stride,padding=1))
                    replacer._replace_pattern(traced_model,node,node,replacement,no_of_avg_pool)
                    no_of_avg_pool+=1
        
        if node.target == nn.functional.avg_pool2d:
            #for functional avgpool
            k_size=node.args[1]
            stride=node.kwargs['stride']
            replacement= nn.Sequential()
            while k_size > 3:
                replacement.append(nn.AvgPool2d(kernel_size=3,stride=1,padding=1))
                k_size-=2
            replacement.append(nn.AvgPool2d(kernel_size=k_size,stride=stride,padding=1))
            traced_model.add_submodule(f'replaced_avgpool_{no_of_avg_pool}',replacement)
            args=(node.args[0],)
            with traced_model.graph.inserting_before(node):
                new_node=traced_model.graph.call_module(f'replaced_avgpool_{no_of_avg_pool}',args,{})
                node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)
        
        traced_model.graph.lint()
        traced_model.recompile()
    return traced_model


def replace_conv2d_kernel_size_ge_7(model:nn.Module):
    '''
    replaces all conv2d module or function having kernel size greater than or equal to 7
    with a stack of conv2d modules having kernel size 3
    
    to have same output pixels original stride is added to last conv module
    '''

    traced_model=symbolic_trace(model)
    modules=dict(traced_model.named_modules())
    no_of_conv=0
    import math, random
    
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            #for call module conv
            module=modules[node.target]
            if isinstance(module,nn.Conv2d):
                if module.kernel_size[0] >5:
                    in_channels=module.in_channels
                    out_channels=module.out_channels
                    k_size=module.kernel_size[0]
                    stride=module.stride[0]
                    replacement= nn.Sequential()
                    while k_size > 5:
                        temp_out_channels= 2**(round(math.log2(in_channels))+random.choice([-1,0,1]))
                        replacement.append(custom_modules.cusConvBNRModule(in_channels,temp_out_channels, kernel_size=3,stride=1,padding=1))
                        in_channels=temp_out_channels
                        k_size-=2
                    replacement.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size,stride=stride,padding=1))
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
            in_channels=weight_shape[1]
            out_channels=weight_shape[0]
            k_size=weight_shape[2]
            replacement= nn.Sequential()
            while k_size > 5:
                temp_out_channels= 2**(round(math.log2(in_channels))+random.choice([-1,0,1]))
                replacement.append(custom_modules.ConvBNRModule(in_channels,temp_out_channels, kernel_size=3,stride=1,padding=1))
                in_channels=temp_out_channels
                k_size-=2
            replacement.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size,stride=stride,padding=1))
            traced_model.add_submodule(f'replaced_conv_{no_of_conv}',replacement)
            args=(node.args[0],)
            with traced_model.graph.inserting_before(node):
                new_node=traced_model.graph.call_module(f'replaced_conv_{no_of_conv}',args,{})
                node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)
        
        traced_model.graph.lint()
        traced_model.recompile()
    return traced_model

