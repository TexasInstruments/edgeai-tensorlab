import torch
import onnx_graphsurgeon as gs
from . import utils

# TODO add_col2im_2_torch_graph
def add_col2im_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    torch_nodes[node.name] = torch_graph.call_function(torch.col2im, tuple(args),  name=node.name)