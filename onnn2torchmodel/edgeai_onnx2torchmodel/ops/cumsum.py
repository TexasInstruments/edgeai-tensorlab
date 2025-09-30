import torch
import onnx_graphsurgeon as gs
from . import utils

def add_cumsum_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter, list]
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    reverse = node.attrs.get('reverse', 0) == 1
    if reverse: 
        raise NotImplementedError(f'for {node.name} with operator {node.op} and reverse is not implemented')
    exclusive = node.attrs.get('exclusive', 0) == 1
    if exclusive: 
        raise NotImplementedError(f'for {node.name} with operator {node.op} and exclusive is not implemented')
    torch_nodes[node.name] = torch_graph.call_function(torch.cumsum, tuple(args),  name=node.name)