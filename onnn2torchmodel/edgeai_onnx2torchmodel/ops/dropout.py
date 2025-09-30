import torch
import onnx_graphsurgeon as gs
from . import utils

def torch_dropout(x, ratio=0.5, training=True,):
    return torch.nn.functional.dropout(x, p=ratio, training=training)

def add_dropout_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    seed = node.attrs.get('seed', None)
    torch_nodes[node.name] = torch_graph.call_function(torch_dropout, tuple(args),  name=node.name)
    torch_nodes[node.name].meta['seed'] = seed