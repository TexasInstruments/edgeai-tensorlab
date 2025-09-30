import torch
import onnx_graphsurgeon as gs
from . import utils

def add_reshape_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<= len(node.inputs) <= 2, f'{node.name} with operator {node.op} should have 1 or 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter,list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict(
        
    )
    #TODO add support for allowzero
    allowzero = node.attrs.get('allowzero', 0) == 1
    if 'shape' in node.attrs:
        kwargs['shape'] = node.attrs['shape']

    torch_nodes[node.name] = torch_graph.call_function(torch.reshape, tuple(args),  kwargs, name=node.name)
    for attr in node.attrs:
        if attr in kwargs:
            continue
        torch_nodes[node.name].meta[attr] = node.attrs[attr]
        