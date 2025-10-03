import torch
import onnx_graphsurgeon as gs
from . import utils

#TODO add support for noop_with_empty_axes
def torch_reduce_max(x, axes, keepdims=True, noop_with_empty_axes = False):
    return torch.amax(x, dim=axes, keepdim=keepdims)

def add_reduce_max_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<=len(node.inputs)<= 2, f'{node.name} with operator {node.op} should have between 1 and 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter , list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    
    keepdims = node.attrs.get('keepdims', 0) == 1
    noop_with_empty_axes  = node.attrs.get('noop_with_empty_axes', 0) == 1
    kwargs = dict(keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
    if 'axes' in node.attrs:
        axes = node.attrs['axes'] 
        kwargs['axes'] = axes
    torch_nodes[node.name] = torch_graph.call_function(torch_reduce_max, tuple(args),  kwargs, name=node.name)
    
    
