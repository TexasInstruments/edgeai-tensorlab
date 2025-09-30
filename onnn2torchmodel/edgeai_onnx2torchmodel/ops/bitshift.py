import torch
import onnx_graphsurgeon as gs
from . import utils

def add_bitshift_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    direction = node.attrs.get('direction')
    if direction == 'RIGHT':
        func = torch.bitwise_right_shift
    elif direction == 'LEFT':
        func = torch.bitwise_left_shift
    torch_nodes[node.name] = torch_graph.call_function(func, tuple(args),  name=node.name)