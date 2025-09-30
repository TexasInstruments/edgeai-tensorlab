import torch
import onnx_graphsurgeon as gs
from operator import getitem
from onnx import TensorProto
import numpy as np

def get_input_from_node(inp:gs.Variable|gs.Constant, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module, attr_type:type=None, **kwargs):
    if inp.name in torch_nodes:
        return torch_nodes[inp.name]
    if isinstance(inp, gs.Constant):
        attr_type = attr_type or torch.Tensor
        if attr_type not in (list, tuple, set, torch.Tensor, torch.nn.Parameter, torch.nn.Buffer):
            raise ValueError(f"Unsupported type {attr_type}")
        val = inp.values
        if attr_type == list:
            val = val.tolist()
        elif attr_type in (tuple, set):
            val = attr_type(val)
        if attr_type in (torch.Tensor, torch.nn.Parameter, torch.nn.Buffer):
            val = torch.from_numpy(val)
        try:
            if attr_type in (torch.Tensor, list, tuple, set):
                setattr(torch_module, inp.name, val)
            elif attr_type == torch.nn.Buffer:
                torch_module.register_buffer(inp.name, val,)
            elif attr_type == torch.nn.Parameter:
                val = attr_type(val)
                torch_module.register_parameter(inp.name, val)
        except Exception as e:
            raise ValueError(f"Failed to register {inp.name} in the root module with Exception {e}. # Fix it")
        torch_nodes[inp.name] = torch_graph.get_attr(inp.name)
        return torch_nodes[inp.name]
    if inp.shape is None and inp.dtype is None:
        return None
    if len(inp.inputs) == 0:
        # graph input case 
        # already handled in get_torch_graph_module
        raise ValueError (f"Graph input {inp.name} not handled")
    inp_node = inp.inputs[0]
    if len(inp_node.outputs) == 1:
        return torch_nodes[inp_node.name]
    else:
        index = inp_node.outputs.index(inp)
        torch_nodes[inp.name] = torch_graph.call_function(getitem, tuple([torch_nodes[inp_node.name], index]), name=inp.name)
        return torch_nodes[inp.name]


onnx_2_torch_type_mapping = {
    TensorProto.FLOAT16 : torch.float16,
    TensorProto.FLOAT : torch.float32,
    TensorProto.DOUBLE : torch.float64,
    TensorProto.INT8 : torch.int8,
    TensorProto.UINT8 : torch.uint8,
    TensorProto.INT16 : torch.int16,
    TensorProto.UINT16 : torch.uint16,
    TensorProto.INT32 : torch.int32,
    TensorProto.UINT32 : torch.uint32,
    TensorProto.INT64 : torch.int64,
    TensorProto.UINT64 : torch.uint64,
    TensorProto.BOOL : torch.bool,
}