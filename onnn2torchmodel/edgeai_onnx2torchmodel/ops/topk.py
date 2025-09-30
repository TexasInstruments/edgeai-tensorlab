import torch
import onnx_graphsurgeon as gs
from . import utils

def torch_topk(x, k, axes=-1, largest=True, sorted=True):
    return torch.topk(x, k, dim=axes, largest=largest, sorted=sorted)


def add_topk_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter,list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict(node.attrs)
    torch_nodes[node.name] = torch_graph.call_function(torch_topk, tuple(args),  kwargs, name=node.name)