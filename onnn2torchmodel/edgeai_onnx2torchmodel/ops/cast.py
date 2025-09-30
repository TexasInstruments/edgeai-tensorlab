import torch
import onnx_graphsurgeon as gs
from . import utils
import onnx
from onnx import TensorProto

# TODO add support for round and saturate
def add_cast_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    dtype = node.attrs.get('to')
    round_mode = node.attrs.get('rounding_mode','up')
    saturate = node.attrs.get('saturate', 1) == 1
    dtype = utils.onnx_2_torch_type_mapping[dtype]
    torch_nodes[node.name] = torch_graph.call_method('to', tuple(args),  dict(dtype=dtype), name=node.name)

def add_cast_like_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [ torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    round_mode = node.attrs.get('rounding_mode','up')
    saturate = node.attrs.get('saturate', 1) == 1
    torch_nodes[node.name] = torch_graph.call_method('to', tuple(args),   name=node.name)