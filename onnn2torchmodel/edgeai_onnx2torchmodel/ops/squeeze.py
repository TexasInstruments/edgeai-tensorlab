import torch
import onnx_graphsurgeon as gs
from . import utils

def add_squeeze_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<= len(node.inputs) <= 2, f'{node.name} with operator {node.op} should have between 1 and 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter,list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = {}
    if 'axes' in node.attrs:
        kwargs['dim'] = node.attrs['axes']
    torch_nodes[node.name] = torch_graph.call_function(torch.squeeze, tuple(args), kwargs, name=node.name)