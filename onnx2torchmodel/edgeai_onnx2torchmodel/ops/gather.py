import torch
import onnx_graphsurgeon as gs
from operator import getitem
from . import utils

def torch_gather(x, indices, axis=0):
    if indices.shape:
        return torch.index_select(x, axis, indices)
    slices = [slice(None) for _ in range(x.ndim)]
    indices = indices.cpu().numpy().tolist()
    if axis < 0:
        axis += x.ndim
    if indices<0:
        indices += x.shape[axis]
    slices[axis] = indices
    shape = [x.shape[i] for i in range(x.ndim) if i != axis]
    return getitem(x, tuple(slices))

def add_gather_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', 0)
    torch_nodes[node.name] = torch_graph.call_function(torch_gather, tuple(args),  dict(axis=axis), name=node.name)

