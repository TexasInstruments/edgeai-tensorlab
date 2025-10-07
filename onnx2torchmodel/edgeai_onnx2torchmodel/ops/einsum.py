import torch
import onnx_graphsurgeon as gs
from . import utils

def add_einsum_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    equation = node.attrs.get('equation')
    torch_nodes = torch_graph.call_function(torch.einsum, tuple([equation]+args), name=node.name)
    