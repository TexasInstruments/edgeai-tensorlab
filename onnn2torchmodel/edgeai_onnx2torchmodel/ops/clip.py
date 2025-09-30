import torch
import onnx_graphsurgeon as gs
from . import utils

def add_clip_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [ torch.Tensor for inp in node.inputs]
    assert 3 >= len(node.inputs) >= 1, f'{node.name} with operator {node.op} should have between 1 and 3 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = {}
    if 'min' in node.attrs:
        kwargs['min'] = node.attrs['min']
    if 'max' in node.attrs:
        kwargs['max'] = node.attrs['max']
    torch_nodes[node.name] = torch_graph.call_function(torch.clip, tuple(args),  name=node.name)
    for attr in node.attrs:
        if attr in kwargs:
            continue
        torch_nodes[node.name].meta[attr] = node.attrs[attr]