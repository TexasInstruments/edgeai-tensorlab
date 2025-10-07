import onnx_graphsurgeon as gs
import torch
from . import utils

onnx_to_torch = {
    'Abs': torch.abs,
    'Acos': torch.acos,
    'Acosh': torch.acosh,
    'Asin': torch.asin,
    'Asinh': torch.asinh,
    'Atan': torch.atan,
    'Atanh': torch.atanh,
    'BitwiseNot': torch.bitwise_not,
    'Ceil': torch.ceil,
    'Cos': torch.cos,
    'Cosh': torch.cosh,
    'Det' : torch.det,
    'Erf': torch.erf,
    'Log': torch.log,
    'Relu': torch.relu,
    'Sigmoid': torch.sigmoid,
    'Sin': torch.sin,
    'Sinh': torch.sinh,
    'Tan': torch.tan,
    'Tanh': torch.tanh
}

def add_node_2_torch_graph_1ip_1op(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    inp = node.inputs[0]
    if node.op in onnx_to_torch:
        args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, torch.nn.Parameter if inp.shape else torch.Tensor)]
        torch_nodes[node.name] = torch_graph.call_function(onnx_to_torch[node.op], tuple(args),  name=node.name)
    else:
        raise NotImplementedError