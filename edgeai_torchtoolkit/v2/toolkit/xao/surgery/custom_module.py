import torch, copy
from torch import nn 
from . import replacer
from torch.fx import symbolic_trace
class SEModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32, out_channels= 16,kernel_size=3,),
            nn.Hardsigmoid()
        )
    
    def forward(self,x):
        return torch.mul(self.sequence(x),x)

class SEModule1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.SiLU(),
            nn.Conv2d(in_channels= 32, out_channels= 16,kernel_size=3,),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        return torch.mul(self.sequence(x),x)

class InstaModule(nn.Module):
    def __init__(self,preDefinedLayer:nn.Module) -> None:
        super().__init__()
        self.model=preDefinedLayer
    def forward(self,x):
        return self.model(x)
    
class Focus(nn.Module):
    def forward(self,x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x

class ConvBNRModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding) -> None:
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.bn=nn.BatchNorm2d(out_channels)
        self.act=nn.ReLU()
    
    def forward(self,x,*args):
        return self.act(self.bn(self.conv(x)))

    
from edgeai_torchtoolkit.v1.toolkit.xnn. layers.resize_blocks import resize_with_scale_factor

def replace_resize_with_scale_factor(model):
    traced_m =  symbolic_trace(model)
    pattern_m= nn.Upsample()

    traced_pattern= symbolic_trace(pattern_m)
    matches= replacer.straight_chain_searcher(traced_m,traced_pattern)
    for start,end in matches:
        with traced_m.graph.inserting_before(start):
            kwargs=dict(start.kwargs)
            kwargs.pop('antialias')
            new_node= traced_m.graph.call_function(resize_with_scale_factor,start.args,kwargs)
            start.replace_all_uses_with(new_node)
        traced_m.graph.erase_node(start)
    traced_m.recompile()
    return traced_m

def replace_maxpool2d_k_gt5(model:nn.Module):
    traced_model=symbolic_trace(model)
    modules=dict(traced_model.named_modules())
    no_of_max_pool=0
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
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
            k_size=node.args[1]
            stride=node.kwargs['stride']
            replacement= nn.Sequential()
            padding=1 
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
def replace_avgpool2d_k_gt5(model:nn.Module):
    traced_model=symbolic_trace(model)
    modules=dict(traced_model.named_modules())
    no_of_avg_pool=0
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
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
            k_size=node.args[1]
            stride=node.kwargs['stride']
            replacement= nn.Sequential()
            padding=1 
            while k_size > 3:
                replacement.append(nn.AvgPool2d(kernel_size=3,stride=1,padding=1))
                k_size-=2
            replacement.append(nn.AvgPool2d(kernel_size=k_size,stride=stride,padding=1))
            traced_model.add_submodule(f'replaced_maxpool_{no_of_avg_pool}',replacement)
            args=(node.args[0],)
            with traced_model.graph.inserting_before(node):
                new_node=traced_model.graph.call_module(f'replaced_maxpool_{no_of_avg_pool}',args,{})
                node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)
        
        traced_model.graph.lint()
        traced_model.recompile()
    return traced_model

def replace_conv2d_k_gt5(model:nn.Module):
    traced_model=symbolic_trace(model)
    modules=dict(traced_model.named_modules())
    no_of_conv=0
    import math, random
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            if isinstance(module,nn.Conv2d):
                module=modules[node.target]
                if module.kernel_size >3:
                    in_channels=module.in_channels
                    out_channels=module.out_channels
                    k_size=module.kernel_size[0]
                    stride=module.stride[0]
                    replacement= nn.Sequential()
                    while k_size > 3:
                        temp_out_channels= 2**(round(math.log2(in_channels))+random.choice([-1,0,1]))
                        replacement.append(ConvBNRModule(in_channels,temp_out_channels, kernel_size=3,stride=1,padding=1))
                        in_channels=temp_out_channels
                        k_size-=2
                    replacement.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size,stride=stride,padding=1))
                    replacer._replace_pattern(traced_model,node,node,replacement,no_of_conv)
                    no_of_conv+=1
        
        if node.target == nn.functional.conv2d:
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
            padding=1 
            while k_size > 3:
                temp_out_channels= 2**(round(math.log2(in_channels))+random.choice([-1,0,1]))
                replacement.append(ConvBNRModule(in_channels,temp_out_channels, kernel_size=3,stride=1,padding=1))
                in_channels=temp_out_channels
                k_size-=2
            replacement.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size,stride=stride,padding=1))
            traced_model.add_submodule(f'replaced_maxpool_{no_of_conv}',replacement)
            args=(node.args[0],)
            with traced_model.graph.inserting_before(node):
                new_node=traced_model.graph.call_module(f'replaced_maxpool_{no_of_conv}',args,{})
                node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)
        
        traced_model.graph.lint()
        traced_model.recompile()
    return traced_model

