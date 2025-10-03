import torch
import onnx_graphsurgeon as gs
from . import utils


def add_batchnorm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 5, f'{node.name} with operator {node.op} should have 5 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, utils.Buffer, utils.Buffer]
    inp, weight, bias, mean, var = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    epsilon = node.attrs.get('epsilon', 1e-5)
    momentum = node.attrs.get('momentum', 0.9)
    training_mode = node.attrs.get('training_mode', 0) == 1

    kwargs = dict(
        eps = epsilon,
        momentum = momentum,
        training = training_mode
    )
    

    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.batch_norm, [inp, mean, var, weight, bias], kwargs, name=node.name)

def add_instance_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]
    inp, weight, bias = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    epsilon = node.attrs.get('epsilon', 1e-5)
    kwargs = dict(
        eps = epsilon
    )
    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.instance_norm, tuple([inp,None, None, weight, bias]), kwargs, name=node.name)

def torch_layer_norm(x, weight, bias, axis=-1, eps=1e-5):
    normalized_shape = x.shape[axis],
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def add_layer_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]
    inp, weight, bias = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    epsilon = node.attrs.get('epsilon', 1e-5)
    axis = node.attrs.get('axis', -1)

    kwargs = dict(
        axis = axis,
        eps = epsilon
    )
    torch_nodes[node.name] = torch_graph.call_function(torch_layer_norm, tuple([inp, weight, bias]), kwargs, name=node.name)