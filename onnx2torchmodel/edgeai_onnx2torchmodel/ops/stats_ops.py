import torch
import onnx_graphsurgeon as gs
from . import utils

def apply_func_to_all_tensors(func, tensors):
    if len(tensors) == 1:
        return func(tensors[0])
    return [func(tensor) for tensor in tensors]

onnx_2_torch = {
    'Max': torch.max,
    'Min': torch.min,
    'Mean': torch.mean,
}

def add_stat_op_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert node.op in onnx_2_torch, f'{node.name} with operator {node.op} is not implemented'
    func = onnx_2_torch[node.op]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, torch.Tensor) for inp in node.inputs]
    torch_nodes[node.name] = torch_graph.call_function(apply_func_to_all_tensors, (func, args), name=node.name)
    

