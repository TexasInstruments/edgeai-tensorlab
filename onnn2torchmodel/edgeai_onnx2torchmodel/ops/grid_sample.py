import torch
import onnx_graphsurgeon as gs
from . import utils

def add_grid_sample_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    mode = node.attrs.get('mode','linear')
    padding_mode = node.attrs.get('padding_mode', 'zeros')
    align_corners = node.attrs.get('align_corners',0) == 1
    if mode in ('linear', 'cubic'):
        mode = 'bi'+mode
    kwargs = dict(
        mode=mode,
        align_corners=align_corners,
        padding_mode=padding_mode
    )
    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.grid_sample, tuple(args),  kwargs, name=node.name)
