import onnx_graphsurgeon as gs
import torch
from . import utils

onnx_2_torch = {
    'ArgMax': torch.argmax,
    'ArgMin': torch.argmin
}

def add_argop_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict(
        dim = node.attrs.get('axis', 0),
        keepdim = node.attrs.get('keepdims', 0) == 1
    )
    torch_nodes[node.name] = torch_graph.call_function(onnx_2_torch[node.op], tuple(args),  kwargs, name=node.name)
    for attr in node.attrs:
        if attr in kwargs:
            continue
        torch_nodes[node.name].meta[attr] = node.attrs[attr]