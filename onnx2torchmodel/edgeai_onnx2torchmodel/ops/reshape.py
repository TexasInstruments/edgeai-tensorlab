import torch
import onnx_graphsurgeon as gs
from . import utils

def torch_reshape(x, shape, allowzero=False):
    if isinstance(shape, torch.Tensor):
        shape = shape.tolist()
    if allowzero:
        return torch.reshape(x, shape)
    for i, s in enumerate(shape):
        if s == -1:
            break
        if s == 0:
            shape[i] = x.shape[i]
    return torch.reshape(x, shape)


def add_reshape_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<= len(node.inputs) <= 2, f'{node.name} with operator {node.op} should have 1 or 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter,list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict(
        
    )
    #TODO add support for allowzero
    allowzero = node.attrs.get('allowzero', 0) == 1
    if 'shape' in node.attrs:
        kwargs['shape'] = node.attrs['shape']
    kwargs['allowzero'] = allowzero

    torch_nodes[node.name] = torch_graph.call_function(torch_reshape, tuple(args),  kwargs, name=node.name)
    for attr in node.attrs:
        if attr in kwargs:
            continue
        torch_nodes[node.name].meta[attr] = node.attrs[attr]

def torch_flatten(x:torch.Tensor, axis):
    if axis == 0:
        return x.reshape(1, -1)
    shape = list(x.shape)
    prod = torch.prod(torch.tensor(shape[:axis]))
    return x.reshape(prod, -1)

def add_flatten_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module ):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', 1)
    torch_nodes[node.name] = torch_graph.call_function(torch_flatten, tuple(args),  dict(axis=axis), name=node.name)
