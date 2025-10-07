import onnx_graphsurgeon as gs
import torch
from . import utils

def add_affine_grid_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph, torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if node.inputs[0].shape else torch.Tensor, list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for (inp,t) in zip(node.inputs, types)]
    kwargs = dict(
        align_corners = node.attrs.get('align_corners', 0) == 1,
    )
    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.affine_grid, tuple(args),  kwargs, name=node.name)
    for attr in node.attrs:
        if attr in kwargs:
            continue
        torch_nodes[node.name].meta[attr] = node.attrs[attr]