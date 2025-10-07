import torch
import onnx_graphsurgeon as gs
from . import utils

def torch_inf(x, det_neg=True, det_pos=True):
    is_neg_inf = x == -float('inf')
    is_pos_inf = x == float('inf')
    x = torch.zeros_like(x).bool()
    if det_neg:
        x = torch.logical_or(x, is_neg_inf)
    if det_pos:
        x = torch.logical_or(x, is_pos_inf)
    return x

def add_is_inf_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    det_neg = node.attrs.get('detect_negative', 1)==1
    det_pos = node.attrs.get('detect_positive', 1)==1
    if det_neg and det_pos:
        torch_nodes[node.name] = torch_graph.call_function(torch.isinf, tuple(args), name=node.name)
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.isinf, tuple(args), dict(det_neg=det_neg, det_pos=det_pos), name=node.name)