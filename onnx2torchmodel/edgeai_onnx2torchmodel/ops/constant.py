import torch
import onnx_graphsurgeon as gs
from . import utils

def add_constant_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 0, f'input {node.inputs} of {node.name} has more than 0 inputs. Constant node should have 0 inputs'
    if (sp_val:= node.attrs.get('sparse_value',None)) is not None:
        if isinstance(sp_val, gs.Constant):
            sp_val = sp_val.values
        try:
            val = torch.nn.Parameter(torch.tensor(sp_val))
        except:
            val = torch.tensor(sp_val)
    elif (val:= node.attrs.get('value',None)) is not None:
        if isinstance(val, gs.Constant):
            val = val.values
        try:
            val = torch.nn.Parameter(torch.tensor(val))
        except:
            val = torch.tensor(val)
    elif (val:= node.attrs.get('value_float',None)) is not None:
        val = torch.nn.Parameter(torch.tensor(val))
    elif (val:= node.attrs.get('value_int',None)) is not None:
        val = (torch.tensor(val))
    elif (val:= node.attrs.get('value_floats',None)) is not None:
        val = torch.nn.Parameter(torch.tensor(val))
    elif (val:= node.attrs.get('value_ints',None)) is not None:
        val = (torch.tensor(val))
    elif (val:= node.attrs.get('value_string',None)) is not None:
        val = val
    elif (val:= node.attrs.get('value_strings',None)) is not None:
        val = val
    else:
        raise ValueError(f'node {node.name} has no value')
    if isinstance(val, torch.nn.Parameter):
        torch_module.register_parameter(node.name, val)
    elif isinstance(val, torch.Tensor):
        torch_module.register_buffer(node.name, val)
    else:
        setattr(torch_module, node.name, val)
    
    torch_nodes[node.name] = torch_graph.get_attr(node.name)

def torch_costant_of_shape(shape, value=0.0):
    return torch.ones(shape)*value

def add_constant_of_shape_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    value = node.attrs.get('value',0.0)
    shape = utils.get_input_from_node(node.inputs[0], torch_graph, torch_nodes, torch_module, list)
    torch_nodes[node.name] = torch_graph.call_function(torch_costant_of_shape, (shape,), dict(value=value), name=node.name)

