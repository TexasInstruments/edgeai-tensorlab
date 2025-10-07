import torch
import onnx_graphsurgeon as gs
from . import utils

def add_hamming_window_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_hann_window_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")