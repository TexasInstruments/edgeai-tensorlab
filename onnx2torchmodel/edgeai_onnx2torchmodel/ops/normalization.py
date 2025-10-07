import torch
import onnx_graphsurgeon as gs
from . import utils
import warnings


def torch_batch_norm(inp, mean, var, weight, bias, num_outputs=1, **kwargs ):
    output = torch.nn.functional.batch_norm(inp, mean, var, weight, bias, **kwargs)
    if num_outputs == 1:
        return output
    raise ValueError(f'num_outputs should be 1, but got {num_outputs}')

def add_batchnorm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 5, f'{node.name} with operator {node.op} should have 5 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, utils.Buffer, utils.Buffer]
    inp, weight, bias, mean, var = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    epsilon = node.attrs.get('epsilon', 1e-5)
    momentum = node.attrs.get('momentum', 0.9)
    training_mode = node.attrs.get('training_mode', 0) == 1
    # if training_mode:
    #     raise NotImplementedError(f'node {node.name} with operator {node.op} in training mode is not implemented yet')
    num_outputs = len(node.outputs)
    for i in range(-1, -num_outputs-1, -1):
        i = i+num_outputs
        out = node.outputs[i]
        if i<1:
            continue
        if out.outputs:
            raise NotImplementedError(f'node {node.name} with operator {node.op} has multiple outputs which are used further in the model, but not implemented yet')
        node.outputs.remove(out)
        node.attrs['training_mode'] = 0
        if out in state.graph.outputs:
            state.graph.outputs.remove(out)
            warnings.warn(f'{out.name} output is removed from the graph because it is not used in model and needed to be removed for {node.name}({node.op}) ')
    if len(node.outputs) != 1:
        raise NotImplementedError(f'node {node.name} with operator {node.op} has multiple outputs which are used further in the model, but not implemented yet')
    
    kwargs = dict(
        eps = epsilon,
        momentum = 1-momentum,
        training = training_mode,
        num_outputs = len(node.outputs)
    )
    

    torch_nodes[node.name] = torch_graph.call_function(torch_batch_norm, tuple([inp, mean, var, weight, bias]), kwargs, name=node.name)

def add_instance_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]
    inp, weight, bias = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    epsilon = node.attrs.get('epsilon', 1e-5)
    kwargs = dict(
        eps = epsilon
    )
    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.instance_norm, tuple([inp,None, None, weight, bias]), kwargs, name=node.name)

def torch_layer_norm(x, weight, bias, axis=-1, eps=1e-5, num_outputs=1):
    normalized_shape = x.shape[axis],
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def add_layer_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]
    inp, weight, bias = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    epsilon = node.attrs.get('epsilon', 1e-5)
    axis = node.attrs.get('axis', -1)
    num_outputs = len(node.outputs)
    for i in range(-1, -num_outputs-1, -1):
        i = i+num_outputs
        out = node.outputs[i]
        if i<1:
            continue
        if out.outputs:
            raise NotImplementedError(f'node {node.name} with operator {node.op} has multiple outputs which are used further in the model, but not implemented yet')
        node.outputs.remove(out)
        if out in state.graph.outputs:
            state.graph.outputs.remove(out)
            warnings.warn(f'{out.name} output is removed from the graph because it is not used in model and needed to be removed for {node.name}({node.op}) ')
    if len(node.outputs) != 1:
        raise NotImplementedError(f'node {node.name} with operator {node.op} has multiple outputs which are used further in the model, but not implemented yet')
    kwargs = dict(
        axis = axis,
        eps = epsilon,
        num_outputs = len(node.outputs)
    )
    torch_nodes[node.name] = torch_graph.call_function(torch_layer_norm, tuple([inp, weight, bias]), kwargs, name=node.name)

def add_group_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]
    inp, weight, bias = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    num_groups = node.attrs.get('num_groups')
    epsilon = node.attrs.get('epsilon', 1e-5)
    kwargs = dict(
        # num_groups = num_groups,
        eps = epsilon
    )
    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.group_norm, tuple([inp, num_groups, weight, bias]), kwargs, name=node.name)
    for attr in node.attrs:
        if attr in kwargs:
            continue
        torch_nodes[node.name].meta[attr] = node.attrs[attr]

def add_lp_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_mean_variance_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")
    
