import torch
import onnx_graphsurgeon as gs
from . import utils

# TODO fix and find usage
def add_blackman_window_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    types = [list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    output_dtype = node.attrs.get('output_datatype')
    output_dtype = utils.onnx_2_torch_type_mapping[output_dtype]
    periodic = node.attrs.get('periodic', None)
    if periodic is not None:
        raise NotImplementedError(f'for {node.name} with operator {node.op} : periodic is not implemented')
    periodic = periodic == 1
    kwargs = dict(dtype=output_dtype,)
    torch_nodes[node.name] = torch_graph.call_function(torch.blackman_window, tuple(args),  name=node.name)