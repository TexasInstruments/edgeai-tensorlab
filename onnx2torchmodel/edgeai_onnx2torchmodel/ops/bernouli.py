import torch
import onnx_graphsurgeon as gs
from . import utils

#TODO fix and find usage
def add_bernouli_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    dtype = node.attrs.get('dtype', 1)
    kwargs = dict(dtype=utils.onnx_2_torch_type_mapping[dtype])
    seed = node.attrs.get('seed', None)
    if seed is not None:
        raise NotImplementedError(f'for {node.name} with operator {node.op} : seed is not implemented')
    torch_nodes[node.name] = torch_graph.call_function(torch.bernoulli, tuple(args),  kwargs, name=node.name)