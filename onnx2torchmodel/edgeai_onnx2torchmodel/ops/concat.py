import torch
import onnx_graphsurgeon as gs
from . import utils

def add_concat_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    dim = node.attrs.get('axis')
    torch_nodes[node.name] = torch_graph.call_function(torch.concat, tuple([args]),  dict(dim=dim), name=node.name)

def add_concat_from_sequence_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    dim = node.attrs.get('axis')
    new_axis = node.attrs.get('new_axis', 0) == 1
    func = torch.stack if new_axis else torch.concat
    torch_nodes[node.name] = torch_graph.call_function(func, tuple([args]),  dict(dim=dim), name=node.name)

def torch_expand(x:torch.Tensor, shape):
    return x.expand(shape)

def add_expand_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    torch_nodes[node.name] = torch_graph.call_function(torch_expand, tuple([args]),   name=node.name)
