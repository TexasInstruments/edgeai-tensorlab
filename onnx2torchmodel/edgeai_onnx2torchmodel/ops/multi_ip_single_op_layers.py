import onnx_graphsurgeon as gs
import torch
from . import utils

onnx_to_torch = {
    'Add': torch.add,
    'Mul': torch.mul,
    'Sub': torch.sub,
    'Div': torch.div,
    'And': torch.logical_and,
    'BitwiseAnd': torch.bitwise_and,
    'BitwiseOr': torch.bitwise_or,
    'BitwiseXor': torch.bitwise_xor,
    'MatMul': torch.matmul
}

def add_node_2_torch_graph_multi_ip_1op(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    if node.op in onnx_to_torch:
        assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
        types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
        args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
        torch_nodes[node.name] = torch_graph.call_function(onnx_to_torch[node.op], tuple(args),  name=node.name)
    else:
        raise NotImplementedError (f"{node.name} with operator {node.op} is not implemented")