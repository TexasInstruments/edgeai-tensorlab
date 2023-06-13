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

class InstaModule(nn.Module):
    def __init__(self,preDefinedLayer:nn.Module) -> None:
        super().__init__()
        self.model=preDefinedLayer
    def forward(self,x):
        return self.model(x)

    
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