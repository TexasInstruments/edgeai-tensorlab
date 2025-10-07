import torch
import onnx_graphsurgeon as gs
from . import utils

def add_celu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 1.0)
    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.celu, tuple(args), dict(alpha=alpha), name=node.name)

def add_elu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 1.0)
    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.elu, tuple(args), dict(alpha=alpha), name=node.name)

def torch_hardsigmoid(x, alpha=0.2, beta=0.5):
    return torch.clip(alpha*x+beta, 0, 1)

def add_hardsigmoid_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 0.5)
    beta = node.attrs.get('beta', 0.5)
    torch_nodes[node.name] = torch_graph.call_function(torch_hardsigmoid, tuple(args), dict(alpha=alpha, beta=beta), name=node.name)

def add_hardswish_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    torch_nodes[node.name] = torch_graph.call_function(torch_hardsigmoid, tuple(args),dict(alpha=1/6, beta=0.5), name=node.name)

def torch_hardmax(input_tensor, axis=-1):
    if axis < 0:
        axis = input_tensor.dim() + axis
    
    # Find indices of maximum values along the specified axis
    max_values, _ = torch.max(input_tensor, dim=axis, keepdim=True)
    
    # Create a mask where elements equal the maximum along the axis
    max_mask = (input_tensor == max_values).to(torch.float32)
    
    # Use cumulative sum to identify the first occurrence of maximum
    # The first max will have a value of 1, and subsequent max values will be > 1
    cumsum = torch.cumsum(max_mask, dim=axis)
    
    # First max has cumsum=1, so we can create the hardmax mask
    hardmax_mask = (cumsum == 1) & (max_mask == 1)
    
    return hardmax_mask.to(torch.float32)

def add_hardmax_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', -1)
    torch_nodes[node.name] = torch_graph.call_function(torch_hardmax, tuple(args), dict(axis=axis), name=node.name)

def add_leakyrelu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 0.01)
    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.leaky_relu, tuple(args), dict(negative_slope=alpha), name=node.name)

def add_log_softmax_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', -1)
    torch_nodes[node.name] = torch_graph.call_function(torch.log_softmax, tuple(args), dict(dim=axis), name=node.name)

