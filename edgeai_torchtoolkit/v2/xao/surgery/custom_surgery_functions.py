from torch import nn,Tensor
from torch.fx import symbolic_trace
from . import replacer
from torchvision import models,ops
from . import custom_modules
import inspect,torch, operator
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
                        replacement.append(custom_modules.ConvBNRModule(in_channels,temp_out_channels, kernel_size=3,stride=1,padding=1))
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


def replace_layer_norm_and_permute(model:nn.Module):
    traced_model=symbolic_trace(model)
    traced_pattern_1=symbolic_trace(models.convnext.LayerNorm2d(96))
    matches_1= replacer.straight_chain_searcher(traced_model,traced_pattern_1)
    main_modules=dict(traced_model.named_modules())
    no_of_layer_norm=0
    for start,end in matches_1:
        num_features=0
        ptr=start
        while ptr!=end.next:
            if ptr.target == nn.functional.layer_norm:
                num_features=ptr.args[1][0]
                break
            if ptr.op == 'call_module':
                module=main_modules[ptr.target]
                if type(module)==nn.LayerNorm:
                    num_features=module.normalized_shape
                    break
            ptr=ptr.next
        new_node_name= f'replaced_layer_norm_2d_{no_of_layer_norm}'
        no_of_layer_norm+=1
        replacement =nn.BatchNorm2d(num_features)
        prev_node=start.args[0]
        if prev_node.op == 'call_module':
            if type(main_modules[prev_node.target]) == nn.AdaptiveAvgPool2d:
                replacement=nn.Identity()
        traced_model.add_submodule(new_node_name,replacement)
        with traced_model.graph.inserting_before(start):
            new_node = traced_model.graph.call_module(new_node_name,(start.args[0],),{})
            end.replace_all_uses_with(new_node)
        ptr=end
        while ptr!=start.prev:
            temp = ptr
            ptr=ptr.prev
            traced_model.graph.erase_node(temp)
    traced_model.graph.lint()
    traced_model.recompile()
    traced_pattern_2=symbolic_trace(models.convnext.CNBlock(96,3,0.4,nn.LayerNorm).block)
    matches_2= replacer.straight_chain_searcher(traced_model,traced_pattern_2)
    # print((traced_pattern_2.graph))
    no_of_layer_norm=0
    for start,end in matches_2:
        dim = main_modules[start.target].in_channels
        replacement=nn.Sequential(
            custom_modules.ConvBNRModule(dim,dim,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(dim,dim,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(dim),
            ops.misc.Permute([0, 2, 3, 1]),
            nn.Linear(dim,4*dim),
            nn.ReLU(),
            nn.Linear(4*dim,dim),
            ops.misc.Permute([0, 3, 1, 2])
        )
        parent_name,name =replacer._get_parent_name(start.target)
        parent_module=main_modules[parent_name]
        parent_module.__setattr__(name,replacement)
        main_modules[start.target]=replacement
        newNode=start
        ptr=start.next

        #if pattern has more than one operational nodes, removes all extra nodes
        if start!=end:
            while ptr!=end:
                if (ptr.op=='call_module'):
                    parent_name,name= replacer._get_parent_name(ptr.target)
                    parent_module=main_modules[parent_name]
                    parent_module.__delattr__(name)

                ptr.replace_all_uses_with(newNode)
                temp=ptr.next  
                traced_model.graph.erase_node(ptr)
                ptr=temp

            if (end.op=='call_module'):
                parent_name,name= replacer._get_parent_name(end.target)
                parent_module=main_modules[parent_name]
                parent_module.__delattr__(name)

            end.replace_all_uses_with(newNode)
            traced_model.graph.erase_node(end)
    traced_model.graph.lint()
    traced_model.recompile()
    # print(traced_model.graph)
    return traced_model
        
def replace_se_layer(model:nn.Module):
    traced_model=symbolic_trace(model)
    matched=[]
    nodes=[]
    for node in traced_model.graph.nodes:
        nodes.append(node)
    modules=dict(traced_model.named_modules())
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
        start, end= None,None
        if ((node.op == 'call_module' and isinstance(modules[node.target],nn.AdaptiveAvgPool2d)) 
        or node.target in ('mean',nn.functional.adaptive_avg_pool2d,)):
            print(node)
            start=node
            node_1= nodes[i+1]
            print(1,node_1)
            if ((node_1.op == 'call_module' and isinstance(modules[node_1.target],nn.Conv2d) )
                or node_1.target in (nn.functional.conv2d,)):
                print(node_1)
                node_2=nodes[i+2]
                print(2,node_2)
                if ((node_2.op == 'call_module' and inspect.getmodule(type(modules[node_2.target]))==nn.modules.activation)
                or node_2.target in activation_func):
                    print(node_2)
                    node_3=nodes[i+3]
                    print(3,node_3)
                    if ((node_3.op == 'call_module' and isinstance(modules[node_3.target],nn.Conv2d) )
                        or node_3.target in (nn.functional.conv2d,)):
                        print(node_3)
                        node_4=nodes[i+4]
                        print(4,node_4)
                        if ((node_4.op == 'call_module' and inspect.getmodule(type(modules[node_4.target]))==nn.modules.activation)
                or node_4.target in activation_func):
                            print(node_4)
                            node_5=nodes[i+5]
                            print(5,node_5)
                            if node.target in (torch.mul,operator.mul,):
                                print(node_5)
                                matched.append((node,node_5))
                                i+=6
                            else: i+=5
                    else: i+=4
                else: i+=3
            else: i+=2
        else: i+=1 
    print(matched)
